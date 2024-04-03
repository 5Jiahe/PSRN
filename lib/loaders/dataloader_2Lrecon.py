"""
data_json has 
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import io
import _init_paths
import os.path as osp
import numpy as np
import h5py
import json
import random

import torch
from torch.autograd import Variable

from loaders.loader import Loader
from mrcn import inference_no_imdb
import functools
from sim_utils import com_simV2

# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

class DataLoader(Loader):

    def __init__(self, data_json, data_h5, sub_obj_wds, similarity, opt):
        # parent loader instance
        Loader.__init__(self, data_json, sub_obj_wds, similarity, data_h5)
        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.num_atts = len(self.att_to_ix)
        self.att_to_cnt = self.info['att_to_cnt']

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}

        #earn
        self.vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
        self.UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
        self.num_vocab = 72704
        self.embed_dim = 300
        self.embedmat_path = 'cache/word_embedding/embed_matrix.npy'
        self.embedding_mat = np.load(self.embedmat_path)

        for image_id, image in self.Images.items():
            # we use its ref's split (there is assumption that each image only has one split)
            split = self.Refs[image['ref_ids'][0]]['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split] += [image_id]
        for k, v in self.split_ix.items():
            print('assigned %d images to split %s' %(len(v), k))
    # load vocabulary file
    def load_vocab_dict_from_file(self, dict_file):
        if (sys.version_info > (3, 0)):
            with open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        else:
            with io.open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        vocab_dict = {words[n]: n for n in range(len(words))}
        return vocab_dict

    def get_key(self, dict, value):
        return [k for (k, v) in dict.items() if v == value]

    def get_word_id(self, dict, key):
        if key == '<UNK>':
            return 3
        try:
            return [v for (k, v) in dict.items() if k == key][0]
        except:
            # print("error keys:", key)
            return 3

    def load_vocab_dict_from_file(self, dict_file):
        if (sys.version_info > (3, 0)):
            with open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        else:
            with io.open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        vocab_dict = {words[n]: n for n in range(len(words))}
        return vocab_dict

    # get word location in vocabulary
    def words2vocab_indices(self, words, vocab_dict):
        if isinstance(words, str):
            vocab_indices = [vocab_dict[words] if words in vocab_dict else vocab_dict[self.UNK_IDENTIFIER]]
        else:
            vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[self.UNK_IDENTIFIER])
                             for w in words]
        return vocab_indices

    # get ann category
    def get_ann_category(self, image_id):
        image = self.Images[image_id]
        ann_ids = image['ann_ids']
        vocab_dict = self.load_vocab_dict_from_file(self.vocab_file)
        ann_cats = []

        for ann_id in ann_ids:
            ann = self.Anns[ann_id]
            cat_id = ann['category_id']
            cat = self.ix_to_cat[cat_id]
            cat = cat.split(' ')
            cat = self.words2vocab_indices(cat, vocab_dict)
            ann_cats += [cat]
        return ann_cats

    # get category for each sentence
    def get_sent_category(self, image_id):
        image = self.Images[image_id]
        ref_ids = image['ref_ids']
        vocab_dict = self.load_vocab_dict_from_file(self.vocab_file)
        sub_wds = []
        obj_wds = []
        for ref_id in ref_ids:
            ref = self.Refs[ref_id]
            sent_ids = ref['sent_ids']
            for sent_id in sent_ids:
                att_wds = self.sub_obj_wds[str(sent_id)]
                sub_wd = att_wds['r1']
                sub_wd = self.words2vocab_indices(sub_wd, vocab_dict)
                obj_wd = att_wds['r6']
                obj_wd = self.words2vocab_indices(obj_wd, vocab_dict)
                sub_wds += [sub_wd]
                obj_wds += [obj_wd]
        none_idx = self.words2vocab_indices('none', vocab_dict)


        sent_att = {}
        sent_att['sub_wds'] = sub_wds
        sent_att['obj_wds'] = obj_wds
        return sent_att, none_idx

    def prepare_mrcn(self, head_feats_dir, args):
        """
        Arguments:
          head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
          args: imdb_name, net_name, iters, tag
        """
        self.head_feats_dir = head_feats_dir
        self.mrcn = inference_no_imdb.Inference(args)
        assert args.net_name == 'res101'
        self.pool5_dim = 1024
        self.fc7_dim = 2048

    # load different kinds of feats
    def loadFeats(self, Feats):
        # Feats = {feats_name: feats_path}
        self.feats = {}
        self.feat_dim = None
        for feats_name, feats_path in Feats.items():
            if osp.isfile(feats_path):
                self.feats[feats_name] = h5py.File(feats_path, 'r')
                self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
                assert self.feat_dim == self.fc7_dim
                print('FeatLoader loading [%s] from %s [feat_dim %s]' %(feats_name, feats_path, self.feat_dim))

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def resetIterator(self, split):
        self.iterators[split]=0

    # expand list by seq per ref, i.e., [a,b], 3 -> [aaabbb]
    def expand_list(self, L, n):
        out = []
        for l in L:
            out += [l] * n
        return out

    def image_to_head(self, image_id):
        """Returns
        head: float32 (1, 1024, H, W)
        im_info: float32 [[im_h, im_w, im_scale]]
        """
        feats_h5 = osp.join(self.head_feats_dir, str(image_id)+'.h5')
        feats = h5py.File(feats_h5, 'r')
        head, im_info = feats['head'], feats['im_info']
        return np.array(head), np.array(im_info)

    def fetch_neighbour_ids(self, ann_id):
        """
        For a given ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not include itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
        """
        ann = self.Anns[ann_id]
        x,y,w,h = ann['box']
        rx, ry = x+w/2, y+h/2

        @functools.cmp_to_key
        def compare(ann_id0, ann_id1):
            x,y,w,h = self.Anns[ann_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x,y,w,h = self.Anns[ann_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer to farmer
            if (rx-ax0)**2+(ry-ay0)**2 <= (rx-ax1)**2+(ry-ay1)**2:
                return -1
            else:
                return 1

        image = self.Images[ann['image_id']]

        ann_ids = list(image['ann_ids'])
        ann_ids = sorted(ann_ids, key=compare)

        st_ann_ids, dt_ann_ids = [], []
        for ann_id_else in ann_ids:
            if ann_id_else != ann_id:
                if self.Anns[ann_id_else]['category_id'] == ann['category_id']:
                    st_ann_ids += [ann_id_else]
                else:
                    dt_ann_ids +=[ann_id_else]
        return st_ann_ids, dt_ann_ids

    def fetch_grid_feats(self, boxes, net_conv, im_info):
        """returns -pool5 (n, 1024, 7, 7) -fc7 (n, 2048, 7, 7)"""
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
        return pool5, fc7

    def compute_lfeats(self, ann_ids):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.empty((len(ann_ids), 5), dtype=np.float32)
        for ix, ann_id in enumerate(ann_ids):
            ann = self.Anns[ann_id]
            image = self.Images[ann['image_id']]
            x, y ,w, h = ann['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)],np.float32)
        return lfeats

    def compute_dif_lfeats(self, ann_ids, topK=5):
        # return ndarray float32 (#ann_ids, 5*topK)
        dif_lfeats = np.zeros((len(ann_ids), 5*topK), dtype=np.float32)
        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx,rcy,rw,rh = rbox[0]+rbox[2]/2,rbox[1]+rbox[3]/2,rbox[2],rbox[3]
            st_ann_ids, _ =self.fetch_neighbour_ids(ann_id)
            # candidate box
            for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cbox = self.Anns[cand_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats


    def fetch_attribute_label(self, ref_ids):
        """Return
    - labels    : Variable float (N, num_atts)
    - select_ixs: Variable long (n, )
    """
        labels = np.zeros((len(ref_ids), self.num_atts))
        select_ixs = []
        for i, ref_id in enumerate(ref_ids):
            # pdb.set_trace()
            ref = self.Refs[ref_id]
            if len(ref['att_wds']) > 0:
                select_ixs += [i]
                for wd in ref['att_wds']:
                    labels[i, self.att_to_ix[wd]] = 1

        return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

    def fetch_cxt_feats(self, ann_ids, opt):
        """
        Return
        - cxt_feats : ndarray (#ann_ids, fc7_dim)
        - cxt_lfeats: ndarray (#ann_ids, ann_ids, 5)
        - dist: ndarray (#ann_ids, ann_ids, 1)
        Note we only use neighbouring "different" (+ "same") objects for computing context objects, zeros padded.
        """

        cxt_feats = np.zeros((len(ann_ids), self.fc7_dim), dtype=np.float32)
        cxt_lfeats = np.zeros((len(ann_ids), len(ann_ids), 5), dtype=np.float32)
        dist = np.zeros((len(ann_ids), len(ann_ids), 1), dtype=np.float32)

        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            for j, cand_ann_id in enumerate(ann_ids):
                cand_ann = self.Anns[cand_ann_id]
                # fc7_feats
                cxt_feats[i, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
                cbox = cand_ann['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                ccx, ccy = cbox[0]+cbox[2]/2, cbox[1]+cbox[3]/2
                dist[i,j,:] = np.array(abs(rcx-ccx)+abs(rcy-ccy))
                cxt_lfeats[i,j,:] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return cxt_feats, cxt_lfeats, dist

    def extract_ann_features(self, image_id, opt):
        """Get features for all ann_ids in an image"""
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        # fetch image features
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch ann features
        ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

        # absolute location features
        lfeats = self.compute_lfeats(ann_ids)
        lfeats = Variable(torch.from_numpy(lfeats).cuda())

        return ann_fc7, ann_pool5, lfeats

    def extract_ann_features_EARN(self, image_id, opt, obj_sim):
        """Get features for all ann_ids in an image"""
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        # fetch image features
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch ann features
        ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

        # absolute location features
        lfeats = self.compute_lfeats(ann_ids)
        lfeats = Variable(torch.from_numpy(lfeats).cuda())

        # relative location features
        dif_lfeats = self.compute_dif_lfeats(ann_ids)
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())

        # fetch context_fc7 and context_lfeats
        cxt_fc7, cxt_lfeats, dist = self.fetch_cxt_feats(ann_ids, opt)
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        dist = Variable(torch.from_numpy(dist).cuda())

        return ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist

    def load_vocab_dict_from_file(self, dict_file):
        if (sys.version_info > (3, 0)):
            with open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        else:
            with io.open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        vocab_dict = {words[n]: n for n in range(len(words))}
        return vocab_dict

    def compute_rel_lfeats(self, sub_ann_id, obj_ann_id):
        if sub_ann_id == -1 or obj_ann_id == -1:
            rel_lfeats = torch.zeros(5)
        else:
            rbox = self.Anns[sub_ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            cbox = self.Anns[obj_ann_id]['box']
            cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
            rel_lfeats = np.array(
                [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            rel_lfeats = torch.Tensor(rel_lfeats)
        return rel_lfeats

    # get batch of data
    def getBatch(self, split, opt, embedding_mat, vocab_dict):
        # options
        split_ix = self.split_ix[split]
        # self.split_ix   {'train': [98304, 52461, 524291, 425988, 458762, 194438,    98304 is image_ID
        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']

        ri = self.iterators[split]
        ri_next = ri+1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]

        ann_ids = self.Images[image_id]['ann_ids']
        ann_num = len(ann_ids)
        ref_ids = self.Images[image_id]['ref_ids']

        img_ref_ids = []
        img_sent_ids = []
        gd_ixs = []
        gd_boxes = []
        for ref_id in ref_ids:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                img_ref_ids += [ref_id]
                img_sent_ids += [sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]
                gd_boxes += [ref['box']]
        img_sent_num = len(img_sent_ids)

        sub_sim = torch.Tensor(self.similarity[str(image_id)]['sub_sim']).cuda().squeeze(2)
        obj_sim = torch.Tensor(self.similarity[str(image_id)]['obj_sim']).cuda().squeeze(2)
        sub_emb = torch.Tensor(self.similarity[str(image_id)]['sub_emb']).cuda()
        obj_emb = torch.Tensor(self.similarity[str(image_id)]['obj_emb']).cuda()

        #DTWREG ANN FEATS
        #ann_fc7, ann_pool5, lfeats = self.extract_ann_features(image_id, opt)
        #EARN ann feats
        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist = \
            self.extract_ann_features_EARN(image_id, opt, obj_sim)

        pool5 = ann_pool5.unsqueeze(0).expand(img_sent_num, ann_num, self.pool5_dim, 7, 7)
        pool5.detach()
        fc7 = ann_fc7.unsqueeze(0).expand(img_sent_num, ann_num, self.fc7_dim, 7, 7)
        fc7.detach()

        ann_fleats = lfeats
        lfeats = lfeats.unsqueeze(0).expand(img_sent_num, ann_num, 5)
        lfeats.detach()
        dif_lfeats = dif_lfeats.unsqueeze(0).expand(img_sent_num, ann_num, TopK*5)
        dif_lfeats.detach()
        cxt_fc7.detach()
        cxt_lfeats.detach()
        dist.detach() #(ann_ids, ann_ids, 1)

        sub_sim.detach()
        obj_sim.detach()
        sub_emb.detach()
        obj_emb.detach()


        att_labels, select_ixs = self.fetch_attribute_label(img_ref_ids)

        labels = np.vstack([self.fetch_seq(sent_id) for sent_id in img_sent_ids])

        labels = Variable(torch.from_numpy(labels).long().cuda())
        max_len = (labels != 0).sum(1).max().data.data
        labels = labels[:, :max_len]

        start_words = np.ones([labels.size(0), 1], dtype=int)*(self.word_to_ix['<BOS>'])
        start_words = Variable(torch.from_numpy(start_words).long().cuda())
        enc_labels = labels.clone()
        enc_labels = torch.cat([start_words, enc_labels], 1)

        zero_pad = np.zeros([labels.size(0), 1], dtype=int)
        zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
        dec_labels = labels.clone()
        dec_labels = torch.cat([dec_labels, zero_pad], 1)

        ###### new data #######
        unkown_triads_sents = []
        sub_wordids = []
        sub_classwordids = []
        obj_wordids = []
        rel_wordids = []

        sub_classembs = []
        sub_wordembs = []
        obj_wordembs = []
        rel_wordembs = []

        sub_words = []
        sub_word_index = []
        obj_words = []
        obj_word_index = []

        ########################

        sent_i = 0

        for sent_id in img_sent_ids:
            sub_wordid = self.fetch_sub(sent_id)
            sub_word = self.get_key(vocab_dict, sub_wordid)
        # print(sub_wordid)
        # print(sub_word)
            try:
                sub_word_id = self.word_to_ix[sub_word[0]]
            except KeyError:
                sub_word_id = 1996
            # print(sentence[i])
            # print("unk sub word:", sub_word)
            sub_wordids.append(sub_word_id)
            sub_words.append(sub_word)
            sub_wordemb = embedding_mat[sub_wordid]
        # print(sub_wordemb.ndim)
        # print(sub_wordemb.shape)
            sub_wordembs.append(sub_wordemb)
            if sub_word_id in labels[sent_i]:
                index = labels[sent_i].tolist().index(sub_word_id)
            else:
                index = -1  # triads在原句中查找不到
            sub_word_index.append(index)

        # sub_classid = sent_sub_classid[unicode(sent_id)]
        # sub_classids.append(sub_classid)
            sub_classwordid = self.fetch_cls(sent_id)
            sub_classwordids.append(sub_classwordid)
            sub_classemb = embedding_mat[sub_classwordid]
        # print(sub_classemb.shape)
            sub_classembs.append(sub_classemb)

            # obj_nounid = sent_obj_nounid[unicode(sent_id)]
            # obj_nounids.append(obj_nounid)
            obj_wordid = self.fetch_obj(sent_id)
            obj_word = self.get_key(vocab_dict, obj_wordid)
            obj_words.append(obj_word)
            try:
                obj_word_id = self.word_to_ix[obj_word[0]]
            except KeyError:
                obj_word_id = 1996
                # print(sentence[i])
                # print("unk obj word:", obj_word)
            obj_wordids.append(obj_word_id)
            obj_wordemb = embedding_mat[obj_wordid]
            obj_wordembs.append(obj_wordemb)
            if obj_word_id in labels[sent_i]:
                index = labels[sent_i].tolist().index(obj_word_id)
            else:
                index = -1  # triads在原句中查找不到
            obj_word_index.append(index)

            # rel_prepid = sent_rel_prepid[unicode(sent_id)]
            # rel_prepids.append(rel_prepid)
            rel_wordid = self.fetch_rel(sent_id)
            rel_wordids.append(rel_wordid)
            rel_wordemb = embedding_mat[rel_wordid]
            rel_wordembs.append(rel_wordemb)
            #     # print("")
            #     # print("current print(sent_id): ", sent_id)
            #     try:
            #         sent_info = sent_extract[str(sent_id)]
            #         sub_word = sent_info["sent_sub_word"]
            #         obj_word = sent_info["sent_obj_word"]
            #         rel_word = sent_info["sent_rel_word"]
            #         cls_word = sent_info["sent_cls_word"]
            #         sent = sent_info["sentences"]
            #         negs = sent_info["sent_sub_negs"]
            #
            #     except KeyError:
            #         print("!!! neg file not has sent ",sent_id)
            #         sent_info = []
            #         sub_word = "<UNK>"
            #         obj_word = "<UNK>"
            #         rel_word = "self"
            #         cls_word = "<UNK>"
            #         sent = "19"
            #         negs = {}

            # ## negs
            # # sub_wordid = self.get_word_id(vocab_dict,sub_word)
            # # sub_wordids.append(sub_wordid)
            # sub_wordid = sub_word
            # sub_wordemb = embedding_mat[sub_wordid]
            # sub_wordembs.append(sub_wordemb)
            #
            # # sub_classwordid = self.get_word_id(vocab_dict,cls_word)
            # # sub_classwordids.append(sub_classwordid)
            # sub_classwordid = cls_word
            # sub_classemb = embedding_mat[sub_classwordid]
            # sub_classembs.append(sub_classemb)
            #
            # # obj_wordid = self.get_word_id(vocab_dict,obj_word)
            # # obj_wordids.append(obj_wordid)
            # obj_wordid =  obj_word
            # obj_wordemb = embedding_mat[obj_wordid]
            # obj_wordembs.append(obj_wordemb)
            #
            # # rel_wordid = self.get_word_id(vocab_dict,rel_word)
            # # rel_wordids.append(rel_wordid)
            # rel_wordid = rel_word
            # rel_wordemb = embedding_mat[rel_wordid]
            # rel_wordembs.append(rel_wordemb)

            if sub_wordid == 3 and obj_wordid == 3:
                unkown_triads_sent = True
            else:
                unkown_triads_sent = False

            unkown_triads_sents.append(unkown_triads_sent)

            sent_i += 1

            # print(sub_wordid, obj_wordid)
            # print("unkown_triads_sents:", unkown_triads_sents)

        # print(sub_wordids,obj_wordids,unkown_triads_sents)
        data = {}
        data['labels'] = labels
        data['enc_labels'] = enc_labels
        data['dec_labels'] = dec_labels
        data['ref_ids'] = ref_ids
        data['sent_ids'] = img_sent_ids
        data['gd_ixs'] = gd_ixs
        data['gd_boxes'] = gd_boxes
        data['att_labels'] = att_labels
        data['select_ixs'] = select_ixs
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        #################### new data ############################

        data['unkown_triads_sent'] = (torch.Tensor(np.array(unkown_triads_sents,dtype=bool)).cuda()).long()
        data['sub_wordids'] = (torch.Tensor(np.array(sub_wordids,dtype=int)).cuda()).long()
        data['sub_wordembs'] = torch.Tensor(np.array(sub_wordembs)).cuda()
        # print(img_sent_num, ann_num)
        # print("data['sub_wordids']:", data['sub_wordids'], data['sub_wordids'].size())

        data['obj_wordids'] = (torch.Tensor(np.array(obj_wordids)).cuda()).long()
        data['obj_wordembs'] = torch.Tensor(np.array(obj_wordembs)).cuda()

        data['rel_wordids'] = (torch.Tensor(np.array(rel_wordids)).cuda()).long()
        data['rel_wordembs'] = torch.Tensor(np.array(rel_wordembs)).cuda()

        data['sub_classwordids'] = (torch.Tensor(np.array(sub_classwordids)).cuda()).long()
        data['sub_classembs'] = torch.Tensor(np.array(sub_classembs)).cuda()

        data['ann_pool5'] = ann_pool5
        data['ann_fc7'] = ann_fc7
        data['ann_fleats'] = ann_fleats

        ###earn data###
        data['sim'] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}
        data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                         'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'dist': dist}
        data['sub_index'] = sub_word_index
        data['obj_index'] = obj_word_index

        # print("labels: " + str(type(labels)))
        # print("enc_labels: " + str(type(enc_labels)))
        # print("dec_labels: " + str(type(dec_labels)))
        # print("ref_ids: " + str(type(ref_ids)))
        # print("sent_ids: " + str(type(img_sent_ids)) )
        # print("gd_ixs: " + str(type(gd_ixs)))
        # print("gd_boxes: " + str(type(gd_boxes)))
        # print("fc7: " + str(type(fc7)))
        # print("att_labels: " + str(type(att_labels)) + str(att_labels))
        # print("select_ixs: " + str(type(select_ixs)))
        # print("sub_wordids: " + str(type(sub_wordids)))
        # print("ann_pool5: " + str(type(ann_pool5)))
        # print("ann_fleats: " + str(type(ann_fleats)))
        #########################################################
        return data

    def get_attribute_weights(self, scale = 10):
        # weights = \lamda * 1/sqrt(cnt)
        cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
        cnts = np.array(cnts)
        weights = 1 / cnts ** 0.5
        weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        weights = weights * (scale - 1) + 1
        return torch.from_numpy(weights).float()

    def decode_attribute_label(self, scores):
        """- scores: Variable (cuda) (n, num_atts) after sigmoid range [0, 1]
           - labels:list of [[att, sc], [att, sc], ...
        """
        scores = scores.data.cpu().numpy()
        N = scores.shape[0]
        labels = []
        for i in range(N):
            label = []
            score = scores[i]
            for j, sc in enumerate(list(score)):
                label += [(self.ix_to_att[j], sc)]
                labels.append(label)
        return labels

    def getTestBatch(self, split, opt, embedding_mat, vocab_dict):
        wrapped = False
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        ri = self.iterators[split]
        ri_next = ri + 1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        sent_ids = []
        gd_ixs = []
        gd_boxes = []
        att_refs = []
        img_sent_ids = []

        for ref_id in image['ref_ids']:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids += [sent_id]
                img_sent_ids += [sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]
                gd_boxes += [ref['box']]
                att_refs += [ref_id]

        expand_ann_ids = []
        for ann_id in ann_ids:
          for i in range(len(ann_ids)):
            expand_ann_ids.append(ann_id)

        #earn similiraty test
        ann_cats = self.get_ann_category(image_id)
        sent_att, none_idx = self.get_sent_category(image_id)
        self.com_sim = com_simV2.ComSim(self.embedding_mat)
        sub_sim, sub_emb = self.com_sim.cal_sim(ann_cats, sent_att['sub_wds'], none_idx)  # (sent_num, ann_num)
        obj_sim, obj_emb = self.com_sim.cal_sim(ann_cats, sent_att['obj_wds'], none_idx)  # (sent_num, ann_num)

        #APMR ann feats
        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist = \
            self.extract_ann_features_EARN(image_id, opt, obj_sim)

        pool5 = ann_pool5.unsqueeze(0)
        pool5.detach()
        fc7 = ann_fc7.unsqueeze(0)
        fc7.detach()

        ann_fleats = lfeats
        lfeats = lfeats.unsqueeze(0)
        lfeats.detach()
        dif_lfeats = dif_lfeats.unsqueeze(0)
        dif_lfeats.detach()
        cxt_fc7.detach()
        cxt_lfeats.detach()
        dist.detach()  # (ann_ids, ann_ids, 1)

        sub_emb.detach()
        obj_emb.detach()
        sub_sim.detach()
        obj_sim.detach()

        labels = np.vstack([self.fetch_seq(sent_id) for sent_id in sent_ids])
        labels = Variable(torch.from_numpy(labels).long().cuda())
        max_len = (labels!=0).sum(1).max().data.data
        labels = labels[:, :max_len]
        decoded_sent_strs = []
        for i in range(labels.size(0)):
            label = labels[i].tolist()
            sent_str = ' '.join([self.ix_to_word[int(i)] for i in label if i != 0])
            decoded_sent_strs.append(sent_str)

        # print("ix_to_word:", decoded_sent_strs)
        # print("labels:", labels)

        start_words = np.ones([labels.size(0), 1], dtype=int)*(self.word_to_ix['<BOS>'])
        start_words = Variable(torch.from_numpy(start_words).long().cuda())
        enc_labels = labels.clone()
        enc_labels = torch.cat([start_words, enc_labels], 1)

        zero_pad = np.zeros([labels.size(0), 1], dtype=int)
        zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
        dec_labels = labels.clone()
        dec_labels = torch.cat([dec_labels, zero_pad], 1)

        att_labels, select_ixs = self.fetch_attribute_label(att_refs)

        ####### new data #########
        unkown_triads_sents = []

        #sub_nounids = []
        sub_wordids = []
        sub_wordembs = []

        #sub_classids = []
        sub_classwordids = []
        sub_classembs = []

        #obj_nounids = []
        obj_wordids = []
        obj_wordembs = []

        #rel_prepids = []
        rel_wordids = []
        rel_wordembs = []

        sub_words = []
        sub_word_index = []
        obj_words = []
        obj_word_index = []

        # ori_ori
        sent_i = 0

        for sent_id in img_sent_ids:
            sub_wordid = self.fetch_sub(sent_id)
            sub_word = self.get_key(vocab_dict, sub_wordid)
            # print(sub_wordid)
            # print(sub_word)
            try:
                sub_word_id = self.word_to_ix[sub_word[0]]
            except KeyError:
                sub_word_id = 1996
            # print(sentence[i])
            # print("unk sub word:", sub_word)
            sub_wordids.append(sub_word_id)
            sub_words.append(sub_word)
            sub_wordemb = embedding_mat[sub_wordid]
            # print(sub_wordemb.ndim)
            # print(sub_wordemb.shape)
            sub_wordembs.append(sub_wordemb)
            if sub_word_id in labels[sent_i]:
                index = labels[sent_i].tolist().index(sub_word_id)
            else:
                index = -1  # triads在原句中查找不到
            sub_word_index.append(index)

            # sub_classid = sent_sub_classid[unicode(sent_id)]
            # sub_classids.append(sub_classid)
            sub_classwordid = self.fetch_cls(sent_id)
            sub_classwordids.append(sub_classwordid)
            sub_classemb = embedding_mat[sub_classwordid]
            # print(sub_classemb.shape)
            sub_classembs.append(sub_classemb)

            # obj_nounid = sent_obj_nounid[unicode(sent_id)]
            # obj_nounids.append(obj_nounid)
            obj_wordid = self.fetch_obj(sent_id)
            obj_word = self.get_key(vocab_dict, obj_wordid)
            obj_words.append(obj_word)
            try:
                obj_word_id = self.word_to_ix[obj_word[0]]
            except KeyError:
                obj_word_id = 1996
                # print(sentence[i])
                # print("unk obj word:", obj_word)
            obj_wordids.append(obj_word_id)
            obj_wordemb = embedding_mat[obj_wordid]
            obj_wordembs.append(obj_wordemb)
            if obj_word_id in labels[sent_i]:
                index = labels[sent_i].tolist().index(obj_word_id)
            else:
                index = -1  # triads在原句中查找不到
            obj_word_index.append(index)

            # rel_prepid = sent_rel_prepid[unicode(sent_id)]
            # rel_prepids.append(rel_prepid)
            rel_wordid = self.fetch_rel(sent_id)
            rel_wordids.append(rel_wordid)
            rel_wordemb = embedding_mat[rel_wordid]
            rel_wordembs.append(rel_wordemb)
            #     # print("")
            #     # print("current print(sent_id): ", sent_id)
            #     try:
            #         sent_info = sent_extract[str(sent_id)]
            #         sub_word = sent_info["sent_sub_word"]
            #         obj_word = sent_info["sent_obj_word"]
            #         rel_word = sent_info["sent_rel_word"]
            #         cls_word = sent_info["sent_cls_word"]
            #         sent = sent_info["sentences"]
            #         negs = sent_info["sent_sub_negs"]
            #
            #     except KeyError:
            #         print("!!! neg file not has sent ",sent_id)
            #         sent_info = []
            #         sub_word = "<UNK>"
            #         obj_word = "<UNK>"
            #         rel_word = "self"
            #         cls_word = "<UNK>"
            #         sent = "19"
            #         negs = {}

            # ## negs
            # # sub_wordid = self.get_word_id(vocab_dict,sub_word)
            # # sub_wordids.append(sub_wordid)
            # sub_wordid = sub_word
            # sub_wordemb = embedding_mat[sub_wordid]
            # sub_wordembs.append(sub_wordemb)
            #
            # # sub_classwordid = self.get_word_id(vocab_dict,cls_word)
            # # sub_classwordids.append(sub_classwordid)
            # sub_classwordid = cls_word
            # sub_classemb = embedding_mat[sub_classwordid]
            # sub_classembs.append(sub_classemb)
            #
            # # obj_wordid = self.get_word_id(vocab_dict,obj_word)
            # # obj_wordids.append(obj_wordid)
            # obj_wordid =  obj_word
            # obj_wordemb = embedding_mat[obj_wordid]
            # obj_wordembs.append(obj_wordemb)
            #
            # # rel_wordid = self.get_word_id(vocab_dict,rel_word)
            # # rel_wordids.append(rel_wordid)
            # rel_wordid = rel_word
            # rel_wordemb = embedding_mat[rel_wordid]
            # rel_wordembs.append(rel_wordemb)

            if sub_wordid == 3 and obj_wordid == 3:
                unkown_triads_sent = True
            else:
                unkown_triads_sent = False

            unkown_triads_sents.append(unkown_triads_sent)

            sent_i += 1

        # print(sub_wordids, obj_wordids, unkown_triads_sents)
        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = ann_ids
        data['sent_ids'] = sent_ids
        data['gd_ixs'] = gd_ixs
        data['gd_boxes'] = gd_boxes

        data['labels'] = labels
        data['enc_labels'] = enc_labels
        data['dec_labels'] = dec_labels
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        data['att_labels'] = att_labels
        data['select_ixs'] = select_ixs

        ######################### new data #####################
        data['unkown_triads_sent'] = (torch.Tensor(np.array(unkown_triads_sents, dtype=bool)).cuda()).long()

        #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()

        data['sub_wordids'] = (torch.Tensor(np.array(sub_wordids)).cuda()).long() #(sent_num)
        # print(img_sent_num,ann_num)
        # print("data['sub_wordids']:", data['sub_wordids'], data['sub_wordids'].size())
        data['sub_wordembs'] = torch.Tensor(np.array(sub_wordembs)).cuda()

        data['obj_wordids'] = (torch.Tensor(np.array(obj_wordids)).cuda()).long()
        data['obj_wordembs'] = torch.Tensor(np.array(obj_wordembs)).cuda()

        data['rel_wordids'] = (torch.Tensor(np.array(rel_wordids)).cuda()).long()
        data['rel_wordembs'] = torch.Tensor(np.array(rel_wordembs)).cuda()

        data['sub_classwordids'] = (torch.Tensor(np.array(sub_classwordids)).cuda()).long()
        data['sub_classembs'] = torch.Tensor(np.array(sub_classembs)).cuda()

        data['ann_pool5'] = ann_pool5
        data['ann_fc7'] = ann_fc7
        data['ann_fleats'] = ann_fleats

        data['expand_ann_ids'] = expand_ann_ids
        # print("pool5:",pool5.size())
        data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                         'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'dist': dist}
        # print("loader2 sub_sim:", sub_sim.size())
        data['sim'] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}

        data['sub_index'] = sub_word_index
        # print("==========================sub_word_index:", sub_word_index)
        data['obj_index'] = obj_word_index
        #######################################################

        return data

