from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ObjFilterNet(nn.Module):
    def __init__(self, opt):
        super(ObjFilterNet, self).__init__()
        self.obj_filtered_num = opt['obj_filtered_num']

    def filter_obj_ann(self, obj_sim):
        '''

        :param obj_sim: Tensor # (sent_num, ann_num)
        :return:
        sim: Tensor # (sent_num, filtered_ann_num)
        idx: Tensor # (sent_num, filtered_ann_num) 记录内容为第几个ann
        '''
        ann_num = obj_sim.size(1)
        while ann_num < self.obj_filtered_num:
            self.obj_filtered_num -= 1
        sim, idx = torch.topk(obj_sim, self.obj_filtered_num, dim=1) # (sent_num, ann_filtered_num)
        return sim, idx