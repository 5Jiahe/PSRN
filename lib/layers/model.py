from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from layers.lan_enc import RNNEncoder, RNNEncoder_2, PhraseAttention
from layers.lan_dec import RNNDncoder,SubjectDecoder, LocationDecoder, RelationDecoder
from layers.vis_enc import LocationEncoder, SubjectEncoder, RelationEncoder, PairEncoder
from layers.reconstruct_loss import AttributeReconstructLoss, LangLangReconstructLoss, VisualLangReconstructLoss, ReconstructionLoss

class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        assert isinstance(bottom, Variable), 'bottom must be variable'

        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled

class RelationEncoder(nn.Module):
    def __init__(self, opt):
        super(RelationEncoder, self).__init__()
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.lfeat_normalizer    = Normalize_Scale(5, opt['visual_init_norm'])
        self.fc = nn.Linear(opt['fc7_dim']+5, opt['jemb_dim'])

    def forward(self, cxt_feats, cxt_lfeats, obj_attn):
        # cxt_feats.shape = (ann_num,2048), cxt_lfeats=(ann_num,ann_num,5), obj_attn=(sent_num,ann_num),
        # dist=(ann_num,ann_num,1), wo_obj_idx.shape=(sent_num)

        sent_num = obj_attn.size(0)
        ann_num = cxt_feats.size(0)
        batch = sent_num * ann_num

        # cxt_feats
        cxt_feats = cxt_feats.unsqueeze(0).expand(sent_num, ann_num, self.fc7_dim)  # cxt_feats.shape = (sent_num，ann_num,2048)
        obj_attn = obj_attn.unsqueeze(1)  # obj_attn=(sent_num, 1, ann_num)
        cxt_feats = torch.bmm(obj_attn, cxt_feats)  # cxt_feats_fuse.shape = (sent_num，1,2048)
        cxt_feats = self.fc7_normalizer(cxt_feats.contiguous().view(sent_num, -1))
        cxt_feats = cxt_feats.unsqueeze(1).expand(sent_num, ann_num, self.fc7_dim)

        cxt_lfeats = cxt_lfeats.unsqueeze(0).expand(sent_num, ann_num, ann_num, 5)
        cxt_lfeats = cxt_lfeats.contiguous().view(batch, ann_num, 5)
        obj_attn = obj_attn.unsqueeze(1).expand(sent_num, ann_num, 1, ann_num)
        obj_attn = obj_attn.contiguous().view(batch, 1, ann_num)
        cxt_lfeats = torch.bmm(obj_attn, cxt_lfeats) # (batch, 1, 5)
        cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.squeeze(1))
        cxt_lfeats = cxt_lfeats.view(sent_num, ann_num, -1)

        cxt_feats_fuse = torch.cat([cxt_feats, cxt_lfeats], 2)

        return cxt_feats_fuse

class SimAttention_test(nn.Module):
    def __init__(self, vis_dim, words_dim, jemb_dim):
        super(SimAttention_test, self).__init__()
        self.embed_dim = 300
        self.words_dim = words_dim
        self.feat_fuse = nn.Sequential(nn.Linear(words_dim + vis_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, 1))

        self.keep_dim = nn.Sequential(nn.Linear(words_dim, vis_dim),nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_emb, vis_feats):
        sent_num, ann_num  = vis_feats.size(0), vis_feats.size(1)
        # cosine similarity
        word_emb = word_emb.unsqueeze(1).expand(sent_num, ann_num, self.words_dim)
        word_emb = self.keep_dim(word_emb)
        cos_sim = F.cosine_similarity(vis_feats, word_emb, dim = 2)

        return cos_sim

class SimAttention_2layers(nn.Module):
    def __init__(self, vis_dim, words_dim, jemb_dim):
        super(SimAttention_2layers, self).__init__()
        self.embed_dim = 300
        self.words_dim = words_dim
        self.feat_fuse = nn.Sequential(nn.Linear(words_dim + vis_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_emb, vis_feats):
        sent_num, ann_num  = vis_feats.size(0), vis_feats.size(1)
        word_emb = word_emb.unsqueeze(1).expand(sent_num, ann_num, self.words_dim)
        sim_attn = self.feat_fuse(torch.cat([word_emb, vis_feats], 2))
        sim_attn = sim_attn.squeeze(2)

        return sim_attn

class Score(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim):
        super(Score, self).__init__()

        self.feat_fuse = nn.Sequential(nn.Linear(vis_dim+lang_dim, jemb_dim),
                                      nn.ReLU(),
                                      nn.Linear(jemb_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.lang_dim = lang_dim
        self.vis_dim = vis_dim

    def forward(self, visual_input, lang_input):

        sent_num, ann_num = visual_input.size(0), visual_input.size(1)
        lang_input = lang_input.unsqueeze(1).expand(sent_num, ann_num, self.lang_dim)
        lang_input = nn.functional.normalize(lang_input, p=2, dim=2)
        ann_attn = self.feat_fuse(torch.cat([visual_input, lang_input], 2))
        ann_attn = self.softmax(ann_attn.view(sent_num, ann_num))
        ann_attn = ann_attn.unsqueeze(2)

        return ann_attn

class PSRN(nn.Module):
    def __init__(self, opt):
        super(PSRN, self).__init__()
        self.num_layers = opt['rnn_num_layers']
        self.hidden_size = opt['rnn_hidden_size']
        self.num_dirs = 2 if opt['bidirectional'] > 0 else 1
        self.jemb_dim = opt['jemb_dim']
        self.word_vec_size = opt['word_vec_size']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.sub_filter_type = opt['sub_filter_type']
        # self.filter_thr = opt['sub_filter_thr']
        self.word_emb_size = opt['word_emb_size']
        # EARN opt
        self.lang_res_weight = opt['lang_res_weight']
        self.vis_res_weight = opt['vis_res_weight']
        self.att_res_weight = opt['att_res_weight']
        self.loss_combined = opt['loss_combined']
        self.loss_divided = opt['loss_divided']
        self.use_weight = opt['use_weight']
        self.earn_sentence_weight = opt['earn_sentence_weight']

        # language rnn encoder
        self.rnn_encoder = RNNEncoder(vocab_size=opt['vocab_size'],
                                      word_embedding_size=opt['word_embedding_size'],
                                      word_vec_size=opt['word_vec_size'],
                                      hidden_size=opt['rnn_hidden_size'],
                                      bidirectional=opt['bidirectional'] > 0,
                                      input_dropout_p=opt['word_drop_out'],
                                      dropout_p=opt['rnn_drop_out'],
                                      n_layers=opt['rnn_num_layers'],
                                      rnn_type=opt['rnn_type'],
                                      variable_lengths=opt['variable_lengths'] > 0)


        ''''
        self.mlp = nn.Sequential(
            nn.Linear((self.hidden_size * self.num_dirs + self.jemb_dim * 2+self.fc7_dim+self.pool5_dim), self.jemb_dim),
            nn.ReLU())
        '''
        self.sub_encoder = SubjectEncoder(opt)
        self.loc_encoder = LocationEncoder(opt)
        self.rel_encoder = RelationEncoder(opt)

        self.sub_score = Score(self.pool5_dim + self.fc7_dim, opt['word_vec_size'],
                               opt['jemb_dim'])
        self.loc_score = Score(25 + 5, opt['word_vec_size'],
                               opt['jemb_dim'])
        self.obj_score = Score(self.fc7_dim + 5, opt['word_vec_size'],
                               opt['jemb_dim'])

        self.sub_sim_attn = SimAttention_2layers(self.pool5_dim + self.fc7_dim, self.word_emb_size, self.jemb_dim)
        self.obj_sim_attn = SimAttention_2layers(self.fc7_dim, self.word_emb_size, self.jemb_dim)
        self.att_res_weight = opt['att_res_weight']
        self.mse_loss = nn.MSELoss()
        self.sub_decoder = SubjectDecoder(opt)
        self.loc_decoder = LocationDecoder(opt)
        self.obj_decoder = RelationDecoder(opt)

        self.att_res_loss = AttributeReconstructLoss(opt)
        self.vis_res_loss = VisualLangReconstructLoss(opt)
        self.lang_res_loss = LangLangReconstructLoss(opt)
        self.rec_loss = ReconstructionLoss(opt)

        #APMR
        self.weight_fc = nn.Linear(self.num_layers * self.num_dirs * self.hidden_size, 3)
        self.feat_fuse = nn.Sequential(
            nn.Linear(self.fc7_dim + self.pool5_dim + 25 + 5 + self.fc7_dim + 5, opt['jemb_dim']))

        self.sub_attn_apmr = PhraseAttention(self.hidden_size * self.num_dirs)
        self.loc_attn = PhraseAttention(self.hidden_size * self.num_dirs)
        self.obj_attn_apmr = PhraseAttention(self.hidden_size * self.num_dirs)

        self.visual_emb = nn.Sequential(
            nn.Linear(self.fc7_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, opt['word_emb_size']),
        )

        self.pair_emb = nn.Sequential(
            nn.Linear(opt['pair_feat_size'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, opt['word_emb_size']),
        )

        self.pair_encoder = PairEncoder(opt)

        self.pair_attn = SimAttention_test(opt['pair_feat_size'], self.word_emb_size*3, self.jemb_dim)
        self.sub_attn = SimAttention_test(self.fc7_dim, self.word_emb_size, self.jemb_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.mse_loss = nn.MSELoss()

    def forward(self, pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats,
                select_ixs,sub_sim, obj_sim,sub_emb, obj_emb,enc_labels,dec_labels,att_labels,att_weights,
                sub_wordembs, sub_classembs, obj_wordembs, rel_wordembs,
                ann_pool5, ann_fc7, ann_fleats, labels, mode, RDC,
                sub_word_index,obj_word_index):

        sent_num = pool5.size(0)
        ann_num = pool5.size(1)

        # KTMR
        sub_fuseembs = 0.1 * sub_wordembs + 0.9 * sub_classembs
        pair_wordembs = torch.cat([sub_fuseembs, obj_wordembs, rel_wordembs], 1)
        pair_feats, expand_1_pool5, expand_1_fc7, expand_1_fleats, expand_0_pool5, expand_0_fc7, expand_0_fleats = self.pair_encoder(pool5, fc7, ann_pool5, ann_fc7, ann_fleats)
        pair_attn = self.pair_attn(pair_wordembs, pair_feats)
        # sub attention
        sub_attn = self.sub_attn(sub_fuseembs, expand_1_fc7)
        # obj attention
        obj_attn = self.sub_attn(obj_wordembs, expand_0_fc7)

        #################################################################
        ### feat * attn #################################################
        #################################################################
        # pair_feat * attn
        re_pair_feats = torch.matmul(pair_attn.view(sent_num, 1, ann_num*ann_num), pair_feats)
        re_pair_feats = re_pair_feats.reshape([sent_num, -1])
        # sub_feat * attn
        re_sub_feats = torch.matmul(sub_attn.view(sent_num, 1, ann_num*ann_num), expand_1_fc7)
        re_sub_feats = re_sub_feats.reshape([sent_num, -1])
        # obj_feat * attn
        re_obj_feats = torch.matmul(obj_attn.view(sent_num, 1, ann_num * ann_num), expand_0_fc7)
        re_obj_feats = re_obj_feats.reshape([sent_num, -1])

        #################################################################
        ### re-construct ################################################
        #################################################################
        sub_result = self.visual_emb(re_sub_feats)
        obj_result = self.visual_emb(re_obj_feats)
        rel_result = self.pair_emb(re_pair_feats)

        # sub loss
        sub_loss = self.mse_loss(sub_result, sub_fuseembs)
        sub_loss_sum = torch.sum(sub_loss)

        # obj_loss
        obj_loss = self.mse_loss(obj_result, obj_wordembs)
        obj_loss_sum = torch.sum(obj_loss)

        # rel loss
        rel_loss = self.mse_loss(rel_result, rel_wordembs)
        rel_loss_sum = torch.sum(rel_loss)

        # loss sum origin
        final_attn = 2 * sub_attn + 1 * obj_attn + 1 * pair_attn
        loss_sum = 3*sub_loss_sum + 2*obj_loss_sum + 1 * rel_loss_sum

        #L1
        if mode == "stage1":
            return final_attn, obj_attn, loss_sum, sub_loss_sum, obj_loss_sum, rel_loss_sum, loss_sum, None

        # APMR
        # 句子特征mask掉triads信息
        context, hidden, embedded = self.rnn_encoder(labels)
        weights = F.softmax(self.weight_fc(hidden))  # (sent_num, 3)

        context_mask_sub = context.clone()
        context_mask_obj = context.clone()

        sub_attn_lan, sub_phrase_emb = self.sub_attn_apmr(context, embedded, labels)
        loc_attn_lan, loc_phrase_emb = self.loc_attn(context, embedded, labels)
        obj_attn_lan, obj_phrase_emb = self.obj_attn_apmr(context, embedded, labels)

        if mode == "stage3" and sent_num != 1: #only replce in stage3 train, not eval
            sub_attn_lan, sub_phrase_emb_recon = self.sub_attn_apmr(context_mask_sub, embedded,labels)
            obj_attn_lan, obj_phrase_emb_recon = self.obj_attn_apmr(context_mask_obj, embedded, labels)

        add_sub_attn = torch.zeros(sent_num, sub_attn_lan.size(1)).to(torch.device("cuda:0"))
        add_obj_attn = torch.zeros(sent_num, sub_attn_lan.size(1)).to(torch.device("cuda:0"))

        for ij in range(sent_num):#every sent
            if sub_word_index[ij] >= 0:
                add_sub_attn[ij, sub_word_index[ij]] = torch.max(sub_attn_lan[ij])

            if obj_word_index[ij] > 0:
                add_obj_attn[ij,obj_word_index[ij]] = torch.max(obj_attn_lan[ij])

        sub_attn_lan = sub_attn_lan + add_sub_attn
        obj_attn_lan = obj_attn_lan + add_obj_attn

        sub_phrase_emb_recon = torch.bmm(sub_attn_lan.unsqueeze(1), embedded).squeeze(1)
        obj_phrase_emb_recon = torch.bmm(obj_attn_lan.unsqueeze(1), embedded).squeeze(1)

        # subject feats
        sub_feats = self.sub_encoder(pool5, fc7)  # (sent_num, ann_num, 2048+1024)
        sub_attn_1 = self.sub_sim_attn(sub_emb, sub_feats)
        sub_loss = self.mse_loss(sub_attn_1, sub_sim)
        loc_feats = self.loc_encoder(lfeats, dif_lfeats)  # (sent_num, ann_num, 5+25)

        # object attention
        cxt_fc7_att = cxt_fc7.unsqueeze(0).expand(sent_num, ann_num, self.fc7_dim)
        cxt_fc7_att = nn.functional.normalize(cxt_fc7_att, p=2, dim=2)
        obj_attn_1 = self.obj_sim_attn(obj_emb, cxt_fc7_att)
        obj_loss = self.mse_loss(obj_attn_1, obj_sim)

        # object feats
        obj_feats = self.rel_encoder(cxt_fc7, cxt_lfeats, obj_attn_1)  # (sent_num, ann_num, 2048+5) (sent_num, ann_num)
        sub_ann_attn = self.sub_score(sub_feats, sub_phrase_emb)  # (sent_num, ann_num, 1)
        loc_ann_attn = self.loc_score(loc_feats, loc_phrase_emb)  # (sent_num, ann_num, 1)
        obj_ann_attn = self.obj_score(obj_feats, obj_phrase_emb)  # (sent_num, ann_num, 1)

        if mode == "stage3":
            sub_ann_attn = self.sub_score(sub_feats, sub_phrase_emb_recon)  # (sent_num, ann_num, 1)
            loc_ann_attn = self.loc_score(loc_feats, loc_phrase_emb)  # (sent_num, ann_num, 1)
            obj_ann_attn = self.obj_score(obj_feats, obj_phrase_emb_recon)

        if RDC:
            from layers.rel_deter import RD_Criteria
            score_R = RD_Criteria(labels, lfeats, dif_lfeats)
            loc_ann_attn *= score_R

        weights_expand = weights.unsqueeze(1).expand(sent_num, ann_num, 3)
        total_ann_score = (weights_expand * torch.cat([sub_ann_attn, loc_ann_attn, obj_ann_attn], 2)).sum(2)  # (sent_num, ann_num)

        sub_phrase_recons = self.sub_decoder(sub_feats, total_ann_score)  # (sent_num, 512)
        loc_phrase_recons = self.loc_decoder(loc_feats, total_ann_score)  # (sent_num, 512)
        obj_phrase_recons = self.obj_decoder(obj_feats, total_ann_score)  # (sent_num, 512)

        loss = 0
        if self.vis_res_weight > 0:
            vis_res_loss = self.vis_res_loss(sub_phrase_emb, sub_phrase_recons, loc_phrase_emb,
                                             loc_phrase_recons, obj_phrase_emb, obj_phrase_recons, weights)
            if mode == "stage3":
                vis_res_loss = self.vis_res_loss(sub_phrase_emb_recon, sub_phrase_recons, loc_phrase_emb,
                                             loc_phrase_recons, obj_phrase_emb_recon, obj_phrase_recons, weights)
            loss = self.vis_res_weight * vis_res_loss

        if self.lang_res_weight > 0:
            lang_res_loss = self.lang_res_loss(sub_phrase_emb, loc_phrase_emb, obj_phrase_emb, enc_labels, dec_labels)
            if mode == "stage3":
                lang_res_loss = self.lang_res_loss(sub_phrase_emb_recon, loc_phrase_emb, obj_phrase_emb_recon, enc_labels, dec_labels)
            loss += self.lang_res_weight *0.1* lang_res_loss

        # combined_loss
        loss = self.loss_divided * loss

        ann_score = total_ann_score.unsqueeze(1)
        fuse_feats = torch.cat([sub_feats, loc_feats, obj_feats], 2)  # (sent_num, ann_num, 2048+1024+512+512)

        fuse_feats = torch.bmm(ann_score, fuse_feats)
        fuse_feats = fuse_feats.squeeze(1)
        fuse_feats = self.feat_fuse(fuse_feats)
        rec_loss = self.rec_loss(fuse_feats, enc_labels, dec_labels)
        loss += self.loss_combined *0.1* rec_loss
        loss_1 = loss.clone()

        loss = loss + sub_loss + obj_loss

        if self.att_res_weight > 0:
            att_scores, att_res_loss = self.att_res_loss(sub_feats, total_ann_score, att_labels, select_ixs, att_weights)
            loss += self.att_res_weight * att_res_loss

        earn_loss = 0.03 * loss
        total_ann_score = total_ann_score.repeat_interleave(ann_num, dim=1)

        if mode == "stage2":
            loss_sum = 0.08 * loss
            sub_loss_sum = sub_loss
            obj_loss_sum = obj_loss
            rel_loss_sum = loss_1
            final_attn += 0.1 * total_ann_score

            return final_attn, total_ann_score, loss_sum, sub_loss_sum, obj_loss_sum, rel_loss_sum, earn_loss, None

        if mode =="stage3":
            sub_ann_attn = sub_ann_attn.squeeze(2).repeat_interleave(ann_num, dim=1)
            obj_ann_attn = obj_ann_attn.squeeze(2).repeat_interleave(ann_num, dim=1)
            loc_ann_attn = loc_ann_attn.squeeze(2).repeat_interleave(ann_num, dim=1)

            loss_sub = ((sub_attn - sub_ann_attn)**2).mean()
            loss_obj = ((obj_attn - obj_ann_attn) ** 2).mean()
            loss_rel = F.kl_div(F.log_softmax(loc_ann_attn, dim=0), F.softmax(pair_attn, dim=0), reduction='batchmean')

            loss_sum = loss + sub_loss_sum + obj_loss_sum + rel_loss_sum + loss_sub + loss_obj + loss_rel

            another_score = final_attn.clone()
            final_attn += (0.3 * total_ann_score)
            another_score += 0.25 * total_ann_score

        return final_attn, another_score, loss_sum, lang_res_loss, loss_sub, loss_obj, loss_rel, None

