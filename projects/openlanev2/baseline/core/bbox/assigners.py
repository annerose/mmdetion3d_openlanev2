# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# assigners.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import HungarianAssigner, AssignResult
import numpy as np


@BBOX_ASSIGNERS.register_module()
class LaneHungarianAssigner(HungarianAssigner):

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                 bev_range=None,
                 normalize=False):
        super().__init__(cls_cost, reg_cost, iou_cost)
        self.bev_range = bev_range
        self.normalize = normalize

    def assign(self,
               lane_pred,
               cls_pred,
               gt_lanes,
               gt_labels,
               img_meta,
               gt_lanes_ignore=None,
               eps=1e-7):
        assert gt_lanes_ignore is None, \
            'Only case when gt_lanes_ignore is None is supported.'
        num_gts, num_lanes = gt_lanes.size(0), lane_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = lane_pred.new_full((num_lanes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = lane_pred.new_full((num_lanes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_lanes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and lanecost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        if self.normalize:
            gt_lanes_normalized = torch.zeros_like(gt_lanes)
            gt_lanes_normalized[..., 0::3] = (gt_lanes[..., 0::3] - self.bev_range[0]) / (self.bev_range[3] - self.bev_range[0])
            gt_lanes_normalized[..., 1::3] = (gt_lanes[..., 1::3] - self.bev_range[1]) / (self.bev_range[4] - self.bev_range[1])
            gt_lanes_normalized[..., 2::3] = (gt_lanes[..., 2::3] - self.bev_range[2]) / (self.bev_range[5] - self.bev_range[2])
        else:
            gt_lanes_normalized = gt_lanes

        reg_cost = self.reg_cost(lane_pred, gt_lanes_normalized)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        # print(f'------------------ cost type: {cost.dtype}')
        # avoid ValueError: matrix contains invalid numeric entries
        if torch.any(torch.isnan(cost)) or torch.any(torch.isinf(cost)):
            print('########  Error assigner.py: cost invalid numeric entries ....')
            print(cost)
            cost = torch.nan_to_num(cost, nan=0, posinf=0, neginf=0)
        
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            lane_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            lane_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
