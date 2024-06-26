from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import time
import argparse
import _init_paths
from layers.model import PSRN
from loaders.dataloader_2Lrecon import DataLoader
import evals.eval as eval_utils
import torch

def load_model(checkpoint_path, opt):
    tic = time.time()
    model = PSRN(opt)
    checkpoint = torch.load(checkpoint_path)
    print("checkpoint", checkpoint)

    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    model.cuda()
    print('model loaded in %.2f seconds' % (time.time() - tic))
    return model

def evaluate(params):
    data_root = r"E:\guoxingyue\reg\DTWREG-master\DTWREG-master"
    # data_root = ""
    model_prefix = 'output/refcocog_google/mrcn_cmr_with_st'
    model_prefix = osp.join(data_root, model_prefix)
    infos = json.load(open(model_prefix + '.json'))
    model_opt = infos['opt']
    print("model_opt", model_opt)

    model_path = 'output/refcocog_google/mrcn_cmr_with_st.pth'
    model_path = osp.join(data_root, model_path)
    print(model_path)
    model = load_model(model_path, model_opt)

    # set up loader
    data_json = osp.join(data_root, 'cache/prepro', params['dataset_splitBy'], 'data_triads.json')
    data_h5 = osp.join(data_root, 'cache/prepro', params['dataset_splitBy'], 'data.h5')
    sub_obj_wds = osp.join(data_root, 'cache/sub_obj_wds', model_opt['dataset_splitBy'], 'sub_obj.json')
    similarity = osp.join(data_root, 'cache/similarity', model_opt['dataset_splitBy'], 'similarity.json')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, similarity=similarity, opt=model_opt)

    # loader's feats
    feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag'])
    args.imdb_name = model_opt['imdb_name']
    args.net_name = model_opt['net_name']
    args.tag = model_opt['tag']
    args.iters = model_opt['iters']
    loader.prepare_mrcn(head_feats_dir=osp.join('cache/feats/', model_opt['dataset_splitBy'], 'mrcn', feats_dir),
                        args=args)
    ann_feats = osp.join('cache/feats', model_opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag']))
    # load ann features
    loader.loadFeats({'ann': ann_feats})

    # check model_info and params
    assert model_opt['dataset'] == params['dataset']
    assert model_opt['splitBy'] == params['splitBy']

    # evaluate on the split,
    split = params['split']
    model_opt['num_sents'] = params['num_sents']
    model_opt['verbose'] = params['verbose']

    val_loss, acc, predictions = eval_utils.eval_split(loader, model, split, model_opt, mode="stage3")

    print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
          (params['dataset_splitBy'], params['split'], len(predictions), acc * 100.))

    # save
    out_dir = osp.join('results', params['dataset_splitBy'], 'easy')
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = osp.join(out_dir, params['id'] + '_' + params['split'] + '.json')
    with open(out_file, 'w') as of:
        json.dump({'predictions': predictions, 'acc': acc}, of)

    # write to results.txt
    f = open('output/refcoco_unc/easy_results.txt', 'a')
    f.write('[%s]: [%s][%s], id[%s]\'s acc is %.2f%%\n' % \
            (params['id'], params['dataset_splitBy'], params['split'], params['id'], acc * 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco',
                        help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')
    parser.add_argument('--id', type=str, default='0', help='model id name')
    parser.add_argument('--num_sents', type=int, default=-1,
                        help='how many sentences to use when periodically evaluating the loss? (-1=all)')
    parser.add_argument('--verbose', type=int, default=1, help='if we want to print the testing progress')
    args = parser.parse_args()
    params = vars(args)

    # make other options
    params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
    evaluate(params)


