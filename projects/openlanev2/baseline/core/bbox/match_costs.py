# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# match_costs.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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
# import einops as ein
from mmdet.core.bbox.match_costs.builder import MATCH_COST

# def cdist(x: torch.Tensor, y: torch.Tensor, p) -> torch.Tensor:
#     if x.dtype is torch.float16 and x.is_cuda:
#         x = ein.rearrange(x, "b l r -> b l () r")
#         y = ein.rearrange(y, "b l r -> b () l r")
#         return (x - y).norm(dim=-1, p=p)
#     return torch.cdist(x, y, p)

@MATCH_COST.register_module()
class LaneL1Cost:
    r"""
    Notes
    -----
    Adapted from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/match_costs/match_cost.py#L11.

    """
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, lane_pred, gt_lanes):
        
        # print(f'lane_pred: {lane_pred}')
        # print(f'gt_lanes: {gt_lanes}')

        if lane_pred.dtype == torch.float16:

            lane_cost = torch.cdist(lane_pred.float(), gt_lanes, p=1)
        else:
            lane_cost = torch.cdist(lane_pred, gt_lanes, p=1)
        # lane_cost = cdist(lane_pred, gt_lanes, p=1)
        return lane_cost * self.weight
