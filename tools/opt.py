from pprint import pprint
import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--dataset', type=str, default='refcocog', help='name of dataset')
    parser.add_argument('--splitBy', type=str, default='google', help='who splits this dataset')
    # parser.add_argument('--splitBy', type=str, default='unc', help='who splits this dataset')
    parser.add_argument('--start_from', type=str, default=None, help='continuing training from saved model')
    # FRCN setting
    parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
    parser.add_argument('--net_name', default='res101', help='net_name: res101 or vgg16')
    parser.add_argument('--iters', default=1250000, type=int, help='iterations we trained for faster R-CNN')
    parser.add_argument('--tag', default='notime', help='on default tf, don\'t change this!')
    # Filter Settings
    parser.add_argument('--RDC', default=True, type=bool, help='whether RDC')
    parser.add_argument('--sub_filter_type', default='thr', type=str, help='subject filter type: none, thr or soft')
    # parser.add_argument('--sub_filter_thr', default=0.4, type=float, help='subject filter threshold, working only when sub_filter_type is thr')
    parser.add_argument('--sub_filter_thr', default=0.6, type=float,
                        help='subject filter threshold, working only when sub_filter_type is thr')
    parser.add_argument('--sub_filter_num', default=5, type=int, help='subject filter number')
    parser.add_argument('--obj_filtered_num', default=2, type=int, help='object filter number')
    parser.add_argument('--sub_obj_dist', default=300.,  type=float, help='the distance threshold between sub and obj')
    parser.add_argument('--net_type', default='addobj', type=str, help='network type: baseline, addloc, addobj')
    parser.add_argument('--dist_pel', default=1, type=int, help='distance penalty')
    # Visual Encoder Setting
    parser.add_argument('--visual_sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--visual_fuse_mode', type=str, default='concat', help='concat or mul')
    parser.add_argument('--visual_init_norm', type=float, default=20, help='norm of each visual representation')
    parser.add_argument('--visual_use_bn', type=int, default=-1, help='>0: use bn, -1: do not use bn in visual layer')    
    parser.add_argument('--visual_use_cxt', type=int, default=1, help='if we use contxt')
    parser.add_argument('--visual_cxt_type', type=str, default='frcn', help='frcn or res101')
    parser.add_argument('--visual_drop_out', type=float, default=0.2, help='dropout on visual encoder')
    parser.add_argument('--window_scale', type=float, default=2.5, help='visual context type')
    parser.add_argument('--with_visual_att', type=int, default=0, help='whether to use visual_att')
    # Visual Feats Setting
    parser.add_argument('--with_st', type=int, default=1, help='if incorporating same-type objects as contexts')
    parser.add_argument('--num_cxt', type=int, default=5, help='how many surrounding objects do we use')
    # Language Encoder Setting
    parser.add_argument('--word_embedding_size', type=int, default=512, help='the encoding size of each token')
    parser.add_argument('--word_vec_size', type=int, default=512, help='further non-linear of word embedding')
    parser.add_argument('--word_drop_out', type=float, default=0.5, help='word drop out after embedding')
    parser.add_argument('--bidirectional', type=int, default=1, help='bi-rnn')
    parser.add_argument('--rnn_hidden_size', type=int, default=512, help='hidden size of LSTM')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn, gru or lstm')
    parser.add_argument('--rnn_drop_out', type=float, default=0.2, help='dropout between stacked rnn layers')
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of layers in lang_encoder')
    parser.add_argument('--variable_lengths', type=int, default=1, help='use variable length to encode')
    # Language Decoder Setting
    parser.add_argument('--decode_bidirectional', type=int, default=0, help='decode_bi-rnn')
    # Joint Embedding setting
    parser.add_argument('--jemb_drop_out', type=float, default=0.1, help='dropout in the joint embedding')
    parser.add_argument('--jemb_dim', type=int, default=512, help='joint embedding layer dimension')
    # Loss Setting
    parser.add_argument('--att_weight', type=float, default=1.0, help='weight on attribute prediction')
    parser.add_argument('--visual_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (neg_ref, sent)')
    parser.add_argument('--lang_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (ref, neg_sent)')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for ranking loss')
    parser.add_argument('--att_res_weight', type=float, default=1.0, help='weight on attribute reconstruction loss')
    # EARN ADDING Loss Setting
    parser.add_argument('--lang_res_weight', type=float, default=1.0, help='weight on language reconstruction loss')
    parser.add_argument('--vis_res_weight', type=float, default=1.0, help='weight on visual reconstruction loss')
    parser.add_argument('--loss_combined', type=float, default=1.0, help='weight on loss_combined')
    parser.add_argument('--loss_divided', type=float, default=1.0, help='weight on loss_divided')
    parser.add_argument('--use_weight', type=int, default=1, help='whether to use weight for sub, loc, rel ')
    parser.add_argument('--earn_sentence_weight', type=int, default=0.2 , help='the earn loss on dtwreg origin loss')
    # Optimization: General
    parser.add_argument('--max_iters', type=int, default=200000, help='max number of iterations to run')
    parser.add_argument('--sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in number of images per batch')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--seq_per_ref', type=int, default=3, help='number of expressions per object during training')
    parser.add_argument('--learning_rate_decay_start', type=int, default=30000, help='at what iter to start decaying learning rate')
    parser.add_argument('--learning_rate_decay_every', type=int, default=200000, help='every how many iters thereafter to drop LR by half')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--learning_rate', type=float, default=1.26e-5, help='learning rate')
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for adam')
    # Evaluation/Checkpointing
    parser.add_argument('--num_sents', type=int, default=-1, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2000, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', type=str, default='output', help='directory to save models')
    parser.add_argument('--language_eval', type=int, default=0, help='Evaluate language as well (1 = yes, 0 = no)?')
    parser.add_argument('--losses_log_every', type=int, default=500, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1, help='Do we load previous best score when resuming training.')      
    parser.add_argument('--use_IoU', type=int, default=1, help='whether to use IoU or not')
    # misc
    parser.add_argument('--stage', type=int, default=1, help='stage for training.')
    parser.add_argument('--id', type=str, default='mrcn_cmr_with_st', help='an id identifying this run/job.')
    parser.add_argument('--seed', type=int, default=24, help='random number generator seed to use')
    parser.add_argument('--gpuid', type=int, default=1, help='which gpu to use, -1 = use CPU id')
    parser.add_argument('--exp_id', type=str, default='', help='experiment id')
    parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')

    # parse 
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args

if __name__ == '__main__':

    opt = parse_opt()
    print('opt[\'id\'] is ', opt['id'])




