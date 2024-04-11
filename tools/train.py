from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import os.path as osp
import json
import time
import random
import sys
from lib.loaders.dataloader_2Lrecon import DataLoader
from layers.model import PSRN
import evals.utils as model_utils
import evals.eval as eval_utils
import numpy as np
from opt import parse_opt
from Config import *
import io
import torch

def load_vocab_dict_from_file(dict_file):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict

def main(args):
    opt = vars(args)
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['exp_id'])
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    opt['learning_rate'] = learning_rate
    opt['eval_every'] = 3000
    opt['learning_rate_decay_start'] = learning_rate_decay_start
    opt['learning_rate_decay_every'] = learning_rate_decay_every
    opt['pair_feat_size'] = pair_feat_size
    opt['word_emb_size'] = word_emb_size
    opt['class_size'] = class_size
    opt['noun_candidate_size'] = noun_candidate_size
    opt['prep_candidate_size'] = prep_candidate_size

    # set random seed
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])
    data_root = r"E:\guoxingyue\reg\DTWREG-master\DTWREG-master"
    # set up loader
    data_json = osp.join(data_root, 'cache/prepro', opt['dataset_splitBy'], 'data_triads.json')
    data_h5 = osp.join(data_root, 'cache/prepro', opt['dataset_splitBy'], 'data.h5')
    sub_obj_wds = osp.join(data_root, 'cache/sub_obj_wds', opt['dataset_splitBy'], 'sub_obj.json')
    similarity = osp.join(data_root, 'cache/similarity', opt['dataset_splitBy'], 'similarity.json')
    loader_start_time = time.perf_counter()
    loader = DataLoader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, similarity=similarity, opt=opt)
    loader_end_time = time.perf_counter()
    print("loader_time:",loader_end_time-loader_start_time)

    # prepare feats
    feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    head_feats_dir = osp.join(data_root, 'cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)

    loader.prepare_mrcn(head_feats_dir, args)

    ann_feats = osp.join(data_root, 'cache/feats', opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
    loader.loadFeats({'ann': ann_feats})

    # set up model
    opt['vocab_size'] = loader.vocab_size
    opt['fc7_dim'] = loader.fc7_dim
    opt['pool5_dim'] = loader.pool5_dim
    opt['num_atts'] = loader.num_atts

    opt['resume'] = True
    opt['resume'] = False
    if opt['resume']:
        checkpoint_path = osp.join('output/refcocog_google/stage1', 'mrcn_cmr_with_st.pth')
        checkpoint = torch.load(checkpoint_path)
        opt = checkpoint['opt']
        model = PSRN(opt)
        model.load_state_dict(checkpoint['model'].state_dict())
        print("model resumes form ", checkpoint_path)
        mode = "stage2"

    else:
        model = PSRN(opt)
        mode = "stage1"

    opt['RDC'] = False
    infos = {}
    if opt['start_from'] is not None:
        pass
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)
    best_val_score = 0
    att_weights = loader.get_attribute_weights()

    if opt['gpuid'] >= 0:
        model.cuda()

    lr = opt['learning_rate']

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'],
                                 weight_decay=opt['weight_decay'])

    data_time, model_time = 0, 0
    start_time = time.time()

    f = open("./result", "w")
    f.close()
    embedmat_path = 'cache/word_embedding/embed_matrix.npy'
    embedding_mat = np.load(embedmat_path)

    vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
    f = open(vocab_file, "r")
    vocab_dict = load_vocab_dict_from_file(vocab_file)
    f.close()

    mode = "stage3"
    freeze_stage2 = [model.pair_encoder, model.pair_attn, model.sub_attn, model.visual_emb, model.pair_emb]
    freeze_stage3 = [model.pair_encoder, model.pair_attn, model.sub_attn, model.visual_emb, model.pair_emb,
                     model.sub_sim_attn, model.obj_sim_attn, model.sub_attn_apmr, model.loc_attn, model.obj_attn_apmr, model.sub_encoder,
                     model.loc_encoder, model.rel_encoder, model.loc_score, model.sub_decoder, model.rnn_encoder,
                     model.obj_decoder, model.feat_fuse]
    RDC = opt['RDC']


    while True:
        if mode == "stage2":
            for module in freeze_stage2:
                for param in module.parameters():
                    param.requires_grad = False

        if mode == "stage3":
            for module in freeze_stage3:
                for param in module.parameters():
                    param.requires_grad = False

        if mode == "stage1" and (epoch > 3):
            mode = "stage2"
            print("---Enter Stage 2---", mode)
            # update optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt['learning_rate']

        if mode == "stage2" and (epoch > 7):
            mode = "stage3"
            print("---Enter Stage 3---", mode)
            # update optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt['learning_rate']

        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()
        T = {}
        tic = time.time()
        data = loader.getBatch('train', opt, embedding_mat, vocab_dict)

        ####### new data  ################
        sub_wordembs = data['sub_wordembs']
        obj_wordembs = data['obj_wordembs']
        rel_wordembs = data['rel_wordembs']
        sub_classembs = data['sub_classembs']

        ann_pool5 = data['ann_pool5']
        ann_fc7 = data['ann_fc7']
        ann_fleats = data['ann_fleats']

        labels = data['labels']
        enc_labels = data['enc_labels']
        dec_labels = data['dec_labels']
        Feats = data['Feats']
        sub_sim = data['sim']['sub_sim']
        obj_sim = data['sim']['obj_sim']
        sub_emb = data['sim']['sub_emb']
        obj_emb = data['sim']['obj_emb']
        att_labels, select_ixs = data['att_labels'], data['select_ixs']

        sub_word_index = data['sub_index']
        obj_word_index = data['obj_index']

        T['data'] = time.time() - tic

        tic = time.time()
        scores, _, loss, sub_loss, obj_loss, rel_loss, earn_loss, _ = \
            model(Feats['pool5'], Feats['fc7'],Feats['lfeats'],
                                              Feats['dif_lfeats'], Feats['cxt_fc7'],Feats['cxt_lfeats'],
                                              select_ixs, sub_sim, obj_sim, sub_emb, obj_emb,
                                              enc_labels,dec_labels,att_labels,att_weights,
                                              sub_wordembs, sub_classembs, obj_wordembs, rel_wordembs,
                                              ann_pool5, ann_fc7, ann_fleats, labels, mode, RDC,
                                              sub_word_index, obj_word_index)

        try:
            loss.backward()

        except RuntimeError:
            continue

        model_utils.clip_gradient(optimizer, opt['grad_clip'])
        optimizer.step()

        T['model'] = time.time() - tic
        wrapped = data['bounds']['wrapped']

        data_time += T['data']
        model_time += T['model']

        total_time = (time.time() - start_time)/3600
        total_time = round(total_time, 2)

        if iter % opt['losses_log_every'] == 0:
            loss_history[iter] = loss.item()
            print('i[%s], e[%s], sub_loss=%.3f, obj_loss=%.3f, rel_loss=%.3f, earn_loss=%.3f, lr=%.2E, time=%.3f h, mode=%s' % (
            iter, epoch, sub_loss.item(), obj_loss.item(), rel_loss.item(), earn_loss, lr, total_time, mode))
            data_time, model_time = 0, 0

        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor = 0.1 ** frac
            lr = opt['learning_rate'] * decay_factor
            model_utils.set_lr(optimizer, lr)

        if (iter % opt['eval_every'] == 0) and (iter > 1) or iter == opt['max_iters']:
            print("begin eval------------------")
            val_loss, acc, predictions = eval_utils.eval_split(loader, model, 'val', opt, mode)
            val_loss_history[iter] = val_loss
            val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
            val_accuracies += [(iter, acc)]
            print('validation loss: %.4f' % val_loss)
            print('validation acc : %.4f%%\n' % (acc * 100.0))
            current_score = acc
            f = open("./result", "a")
            f.write(str(current_score) + "\n")
            f.close()

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                checkpoint_path = osp.join(checkpoint_dir, mode, opt['id'] + '.pth')
                checkpoint = {}
                checkpoint['model'] = model
                checkpoint['opt'] = opt
                checkpoint['optimizer'] = optimizer
                checkpoint['lr'] = lr
                checkpoint['epoch'] = epoch
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)

            if mode == "stage1" and (epoch > 1):
                mode = "stage2"
                print("---Enter Stage 2---", mode)
                # update optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt['learning_rate']
                model_utils.set_lr(optimizer, opt['learning_rate'])

            if mode == "stage2" and (epoch > 3):
                mode = "stage3"
                print("---Enter Stage 3---", mode)
                # update optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt['learning_rate']
                model_utils.set_lr(optimizer, opt['learning_rate'])

            # write json report
            infos['iter'] = iter
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['loss_history'] = loss_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['val_result_history'] = val_result_history

            with open(osp.join(checkpoint_dir, mode, opt['id'] + '.json'), 'w') as io:
                json.dump(infos, io)

        iter += 1
        if wrapped:
            epoch += 1
        if iter >= opt['max_iters'] and opt['max_iters'] > 0:
            print(str(best_val_score))
            break

if __name__ == '__main__':
    args = parse_opt()
    main(args)

