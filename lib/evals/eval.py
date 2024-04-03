from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import sys
import io



# IoU function
def computeIoU(box1, box2):
  # each box is of [x1, y1, w, h]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  return float(inter)/union

def load_vocab_dict_from_file(dict_file):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict

def eval_split(loader, model, split, opt, mode):
    # mode = 0
    opt['RDC'] = False
    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    acc_new = 0
    expressions = 0
    RDC_number = 1
    unk_triad_acc = 0
    sent_num = 0
    predictions = []
    finish_flag = False
    RDC = opt['RDC']
    model_time = 0

    embedmat_path = 'cache/word_embedding/embed_matrix.npy'
    embedding_mat = np.load(embedmat_path)

    vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
    f = open(vocab_file, "r")
    vocab_dict = load_vocab_dict_from_file(vocab_file)
    f.close()

    while True:
        data = loader.getTestBatch(split, opt, embedding_mat, vocab_dict)
        att_weights = loader.get_attribute_weights()
        sent_ids = data['sent_ids']
        Feats = data['Feats']
        labels = data['labels']
        enc_labels = data['enc_labels']
        dec_labels = data['dec_labels']
        image_id = data['image_id']
        ann_ids = data['ann_ids']
        att_labels, select_ixs = data['att_labels'], data['select_ixs']
        sim = data['sim']

        ######### new data  ############
        sub_wordembs = data['sub_wordembs']
        sub_classembs = data['sub_classembs']
        obj_wordembs = data['obj_wordembs']
        rel_wordembs = data['rel_wordembs']
        ann_pool5 = data['ann_pool5']
        ann_fc7 = data['ann_fc7']
        ann_fleats = data['ann_fleats']
        expand_ann_ids = data['expand_ann_ids']
        labels = data['labels']

        sub_word_index = data['sub_index']
        obj_word_index = data['obj_index']

        ################################
        for i, sent_id in enumerate(sent_ids):
            expressions+=1

            ########### new data #################
            sub_wordemb = sub_wordembs[i:i + 1]
            sub_classemb = sub_classembs[i:i + 1]
            obj_wordemb = obj_wordembs[i:i + 1]
            rel_wordemb = rel_wordembs[i:i + 1]

            #######################################
            enc_label = enc_labels[i:i + 1] # (1, sent_len)
            max_len = (enc_label != 0).sum().item()
            enc_label = enc_label[:, :max_len]  # (1, max_len)
            dec_label = dec_labels[i:i + 1]
            dec_label = dec_label[:, :max_len]

            label = labels[i:i + 1]
            max_len = (label != 0).sum().item()
            label = label[:, :max_len]  # (1, max_len)

            label_int = [tensor.int().item() for tensor in label[0]]

            RDC_list = set([79,494,2264,2647,
                               20,604,
                               913,196,253,84,
                               1052,391,676,73,
                               352,821,1609,35,360,840,
                               186,42,656,1425,915,
                               992,2598])
            intersection = RDC_list.intersection(set(label_int))
            if intersection:
                RDC_TEMP=True
                RDC_number += 1
            else:
                RDC_TEMP=False

            sub_sim = sim['sub_sim'][i:i+1]
            obj_sim = sim['obj_sim'][i:i+1]
            sub_emb = sim['sub_emb'][i:i+1]
            obj_emb = sim['obj_emb'][i:i+1]

            att_label = att_labels[i:i + 1]
            if i in select_ixs:
                select_ix = torch.LongTensor([0]).cuda()
            else:
                select_ix = torch.LongTensor().cuda()

            sub_word_index_current = sub_word_index[i:i+1]
            obj_word_index_current = obj_word_index[i:i+1]

            tic = time.time()

            scores, ann_scores, loss, sub_loss, obj_loss, rel_loss, earn_loss, sub_idx = \
                                                        model(Feats['pool5'], Feats['fc7'], Feats['lfeats'],
                                                               Feats['dif_lfeats'], Feats['cxt_fc7'], Feats['cxt_lfeats'],
                                                               select_ix, sub_sim, obj_sim,sub_emb,obj_emb,
                                                               enc_label, dec_label, att_label, att_weights,
                                                               sub_wordemb, sub_classemb, obj_wordemb, rel_wordemb,
                                                               ann_pool5, ann_fc7, ann_fleats, label, mode, RDC,
                                                               sub_word_index_current, obj_word_index_current)
            scores = scores.squeeze(0)
            ann_scores = ann_scores.squeeze(0)

            loss = loss.item()

            pred_ix = torch.argmax(scores)
            pred_ix_new = torch.argmax(ann_scores)

            pred_ann_id = expand_ann_ids[pred_ix]
            pred_ann_id_NEW = expand_ann_ids[pred_ix_new]

            gd_ix = data['gd_ixs'][i]
            loss_sum += loss
            loss_evals += 1

            pred_box = loader.Anns[pred_ann_id]['box']
            pred_box_new = loader.Anns[pred_ann_id_NEW]['box']
            gd_box = data['gd_boxes'][i]
            sent_num += 1

            IoU = computeIoU(pred_box, gd_box)
            IoU_new = computeIoU(pred_box_new, gd_box)

            # print("当前图片：%s，当前描述：%s，预测box：%s，真实box：%s，IoU:%s, 句子： %s" % (str(image_id),str(sent_id), str(pred_box), str(gd_box), IoU, str()))
            if opt['use_IoU'] > 0:
                if IoU >= 0.5:
                    acc += 1
                    if RDC_TEMP:
                        acc_new += 1

            else:
                if pred_ix == gd_ix:
                    acc += 1

            entry = {}
            entry['image_id'] = image_id
            entry['sent_id'] = sent_id
            entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0]  # gd-truth sent
            entry['gd_ann_id'] = data['ann_ids'][gd_ix]
            entry['pred_ann_id'] = pred_ann_id
            entry['pred_score'] = scores.tolist()[pred_ix]
            entry['IoU'] = IoU
            entry['ann_ids'] = ann_ids

            predictions.append(entry)
            toc = time.time()
            model_time += (toc - tic)

            if num_sents > 0  and loss_evals >= num_sents:
                finish_flag = True
                break
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if verbose:
            # if ix0 % 100 == 0 or ix0 >1498:

            if ix0 % 200 == 0 or ix0 > 4645:
                print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), acc_new=%.2f%%' % \
                    (split, ix0, ix1, acc*100.0/loss_evals, loss, acc_new*100.0/RDC_number),"RDC_number",RDC_number,expressions)

        if ix0 > 602:
            break

        model_time = 0

        if finish_flag or data['bounds']['wrapped']:
            break

    return loss_sum / loss_evals, acc / loss_evals, predictions