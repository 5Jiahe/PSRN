# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mrcn.model.config import cfg
from mrcn.model.bbox_transform import bbox_transform_inv, clip_boxes
from mrcn.model.nms_wrapper import nms
from mrcn.model.nms_wrapper import nms

import torch
from torch.autograd import Variable


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  Parameters:
  - rpn_cls_prob : (1, H, W, 2A) float Variable
  - rpn_bbox_pred: (1, H, W, 4A)
  - im_info      : [im_height, im_width, scale], ndarray (3, )
  - cfg_key      : train or test
  - _feat_stride : 16
  - anchors      : (HWA, 4) float Variable 
  - num_anchors  : A = 9
  Returns:
  - blob         : Variable (N_nms, 5) [0; x1y1x2h2]
  - scores       : Variable (N_nms, )
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N   # train 12000; test 6000
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # train 2000 ; test 300
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  im_info = im_info[0]
  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:] # (1, H, W, A) pos score only
  rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))  # (HWA, 4)
  scores = scores.contiguous().view(-1, 1)     # (HWA, 1)
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])  # (HWA, 4)

  # Pick the top region proposals
  scores, order = scores.view(-1).sort(descending=True)
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
    scores = scores[:pre_nms_topN].view(-1, 1)
  proposals = proposals[order.data, :]

  # Non-maximal suppression
  keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep,]

  # Only support single image as input
  batch_inds = Variable(proposals.data.new(proposals.size(0), 1).zero_())
  blob = torch.cat((batch_inds, proposals), 1)

  return blob, scores

