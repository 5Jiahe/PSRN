# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mrcn.model.config import cfg
from mrcn.model.bbox_transform import bbox_transform_inv, clip_boxes
import numpy.random as npr

import torch
from torch.autograd import Variable

def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
  """A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report
  Parameters:
  - rpn_cls_prob : (1, H, W, 2A) float Variable
  - rpn_bbox_pred: (1, H, W, 4A)
  - im_info      : [im_height, im_width, scale], ndarray (3, )
  - _feat_stride : 16
  - anchors      : (HWA, 4) float Variable 
  - num_anchors  : A = 9
  Returns:
  - blob         : (N_nms, 5) float Variable [0; x1y1x2h2]
  - scores       : (N_nms, ) float Variable
  """
  rpn_top_n = cfg.TEST.RPN_TOP_N
  im_info = im_info[0]

  scores = rpn_cls_prob[:, :, :, num_anchors:]

  rpn_bbox_pred = rpn_bbox_pred.view(-1, 4)
  scores = scores.contiguous().view(-1, 1)

  length = scores.size(0)
  if length < rpn_top_n:
    # Random selection, maybe unnecessary and loses good proposals
    # But such case rarely happens
    top_inds = torch.from_numpy(npr.choice(length, size=rpn_top_n, replace=True)).long().cuda()
  else:
    top_inds = scores.sort(0, descending=True)[1]
    top_inds = top_inds[:rpn_top_n]
    top_inds = top_inds.view(rpn_top_n)

  # Do the selection here
  anchors = anchors[top_inds].contiguous()
  rpn_bbox_pred = rpn_bbox_pred[top_inds].contiguous()
  scores = scores[top_inds].contiguous()

  # Convert anchors into proposals via bbox transformations
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

  # Clip predicted boxes to image
  proposals = clip_boxes(proposals, im_info[:2])

  # Output rois blob
  # Our RPN implementation only supports a single input image, so all
  # batch inds are 0
  batch_inds = Variable(proposals.data.new(proposals.size(0), 1).zero_())
  blob = torch.cat([batch_inds, proposals], 1)
  return blob, scores
