# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from mrcn.model.config import cfg
from mrcn.model.bbox_transform import bbox_transform
from mrcn.utils.bbox import bbox_overlaps

# from scipy.misc import imresize
from imageio import imread
import torch
from torch.autograd import Variable

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, gt_masks, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  Parametes:
  - rpn_rois  : Variable (N, 5) which is (0, x1y1x2y2)
  - rpn_scores: Variable (N, ) 
  - gt_boxes  : Variable (M, 5) which is [x1 y1 x2 y2 cls]
  - gt_masks  : ndarray uint8 (M, im_height, im_width) range {0,1}, note in NDARRAY
  - _num_classes: K
  Return:
  - rois        : (Nkp, 5)
  - roi_scores  : (Nkp, )
  - labels      : (Nkp, 1) range (0, clss)
  - bbox_targets: (Nkp, 4cls)
  - bbox_inside_weights : (Nkp, 4cls)
  - bbox_outside_weights: (Nkp, 4cls)
  - mask_targets        : (Npos, 14, 14) in Variable cuda [0, 1]
  Inputs and outputs are in torch.autograd.Variable (cuda)
  """
  
  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = Variable(rpn_rois.data.new(gt_boxes.size(0), 1))
    all_rois = torch.cat(
      (all_rois, torch.cat((zeros, gt_boxes[:, :-1]), 1))
    , 0)
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = torch.cat((all_scores, zeros), 0)

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))
  
  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights, mask_targets = _sample_rois(
    all_rois, all_scores, gt_boxes, gt_masks,
    fg_rois_per_image, rois_per_image, _num_classes)

  rois = rois.view(-1, 5)
  roi_scores = roi_scores.view(-1)
  labels = labels.view(-1, 1)
  bbox_targets = bbox_targets.view(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
  bbox_outside_weights = (bbox_inside_weights > 0).float()

  # move to Variable
  bbox_targets = Variable(bbox_targets)
  bbox_inside_weights = Variable(bbox_inside_weights)
  bbox_outside_weights = Variable(bbox_outside_weights)
  mask_targets = Variable(mask_targets)

  # return
  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, mask_targets
        

def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
  # Inputs are tensor

  clss = bbox_target_data[:, 0]
  bbox_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  bbox_inside_weights = clss.new(bbox_targets.shape).zero_()
  inds = (clss > 0).nonzero().view(-1)
  if inds.numel() > 0:
    clss = clss[inds].contiguous().view(-1,1)
    dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
    dim2_inds = torch.cat([4*clss, 4*clss+1, 4*clss+2, 4*clss+3], 1).long()
    bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
    bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""
  # Inputs are tensor

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return torch.cat(
    [labels.unsqueeze(1), targets], 1)


def _sample_rois(all_rois, all_scores, gt_boxes, gt_masks, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background examples.
  Return:
  - labels: (Nkp, )
  - rois  : (Nkp, 5), [0 x1 y1 x2 y2]
  - roi_scores  : (Nkp, )
  - bbox_targets: (Nkp, 4k)
  - bbox_inside_weights: (Nkp, 4k)
  """
  # overlaps: (rois x gt_boxes)
  all_rois_data = all_rois.data
  gt_boxes_data = gt_boxes.data
  overlaps = bbox_overlaps(all_rois_data[:, 1:5], gt_boxes_data[:, :4])
  max_overlaps, gt_assignment = overlaps.max(1)  # cuda tensor
  labels = gt_boxes[gt_assignment, [4]]  # cuda Variable

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.numel() > 0 and bg_inds.numel() > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(fg_rois_per_image), replace=False)).long().cuda()]
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.numel() < bg_rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long().cuda()]
  elif fg_inds.numel() > 0:
    to_replace = fg_inds.numel() < rois_per_image
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = rois_per_image
  elif fg_inds.numel() == 0:
    # we always make fg_inds.numel() > 0 
    zeros = Variable(all_rois.data.new(gt_boxes.size(0), 1))
    all_rois = torch.cat(
      (all_rois, torch.cat((zeros, gt_boxes[:, :-1]), 1))
    , 0)
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = torch.cat((all_scores, zeros), 0)
    return _sample_rois(all_rois, all_scores, gt_boxes, gt_masks, fg_rois_per_image, rois_per_image, num_classes)
  # elif bg_inds.numel() > 0:
  #   to_replace = bg_inds.numel() < rois_per_image
  #   bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
  #   fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = torch.cat([fg_inds, bg_inds], 0)
  # Select sampled values from various arrays:
  labels = labels[keep_inds].contiguous()
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds].contiguous()
  roi_scores = all_scores[keep_inds].contiguous()

  bbox_target_data = _compute_targets(
    rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data, labels.data)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  # Get masks, float (num_boxes, 14, 14) 
  # corresponding to the selected boxes
  mask_targets = torch.FloatTensor(fg_inds.numel(), cfg.MASK_SIZE, cfg.MASK_SIZE).cuda() 
  mix = 0
  for i in fg_inds.cpu().numpy().tolist():
    roi = all_rois_data[i] # tensor [xyxyc]
    cropped = gt_masks[gt_assignment[i], int(roi[2]):int(roi[4])+1, int(roi[1]):int(roi[3])+1] # uint8 {0,1}
    cropped = imresize(cropped, (cfg.MASK_SIZE, cfg.MASK_SIZE), interp='nearest')  # still uint8 {0,1}
    cropped = cropped.astype(np.float32)  # float32, range [0,1]
    mask_targets[mix,:,:] = torch.from_numpy(cropped).cuda()
    mix += 1
  assert mask_targets.max() <= 1.0001

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, mask_targets

