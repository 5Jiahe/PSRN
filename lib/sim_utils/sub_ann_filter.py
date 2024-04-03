from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SubHardFilterNet(nn.Module):
    def __init__(self, opt):
        super(SubHardFilterNet, self).__init__()
        self.sub_filter_thr = opt['sub_filter_thr']
        self.sub_filtered_num = opt['sub_filter_num']

    def thr_filter(self, sub_sim):
        '''

        :param sub_sim: Tensor # (sent_num, ann_num)
        :return:
        sub_idx : Tensor # (sent_num, ann_num) 保留的的ann
        '''
        # 高于阈值的内容
        sub_idx = sub_sim.gt(self.sub_filter_thr)
        # 判断是否会过滤掉全部的subject ann
        all_filterd_idx = (sub_idx.sum(1).eq(0)) # (sent_num)
        # 如果全部过滤掉，则不进行过滤
        sub_idx[all_filterd_idx] = 1

        return sub_idx

    def num_filter(self, sub_sim):
        '''
        :param sub_sim: Tensor # (sent_num, ann_num)
        :return:
        sim: Tensor # (sent_num, filtered_ann_num)
        idx: Tensor # (sent_num, filtered_ann_num) 记录内容为第几个ann
        '''
        ann_num = sub_sim.size(1)
        while ann_num < self.sub_filtered_num:
            self.sub_filtered_num -= 1
        sim, idx = torch.topk(sub_sim, self.sub_filtered_num, dim=1)  # (sent_num, sub_filtered_num)
        return sim, idx


