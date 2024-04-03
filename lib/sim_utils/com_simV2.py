from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ComSim(nn.Module):
    def __init__(self, embedding_mat):
        super(ComSim, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_mat))
        self.cossim = nn.CosineSimilarity()

    def cal_sim(self, ann_cats, sen_cats, none_idx):
        ann_cat_embs = self.get_cat_emb(ann_cats)
        sen_cat_embs = self.get_cat_emb(sen_cats)
        cos_sims = []
        sen_wd_emb = []
        for sen_cat_emb in sen_cat_embs:
            sen_cos_sim = []
            sen_wd_emb.append(torch.stack(sen_cat_emb).sum(0).squeeze())
            for ann_cat_emb in ann_cat_embs:
                sen_cos_sim += [self.com_sim(sen_cat_emb, ann_cat_emb, none_idx)]
            cos_sims += [sen_cos_sim]
        return torch.Tensor(cos_sims).cuda(), torch.stack(sen_wd_emb).cuda()

    def com_sim(self, sen_cat_emb, ann_cat_emb, none_idx):
        max_sim = 0
        none_emb = self.embedding(torch.LongTensor(none_idx))
        for sen_wd  in sen_cat_emb:
            for ann_wd  in ann_cat_emb:
                if torch.equal(sen_wd, none_emb):
                    sim = torch.zeros(1)
                else:
                    sim = self.cossim(sen_wd, ann_wd)
                if sim >= max_sim:
                    max_sim = sim
        return max_sim

    def get_cat_emb(self, cats):
        cat_embs = []
        for cat in cats:
            cat_emb = []
            #  注意wds是否都是list
            for wds in cat:
                wd_embedding = self.embedding(torch.LongTensor([wds]))
                cat_emb += [wd_embedding]
            cat_embs += [cat_emb]
        return cat_embs