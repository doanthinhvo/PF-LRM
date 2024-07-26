"""
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import torch.nn as nn
import torch

class MonteCarloPoseLoss(nn.Module):

    def __init__(self, init_norm_factor=1.0, momentum=0.01):
        super(MonteCarloPoseLoss, self).__init__()
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            with torch.no_grad():
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)

        loss_tgt = cost_target
        loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

        loss_pose = loss_tgt + loss_pred  # (num_obj, )
        loss_pose[torch.isnan(loss_pose)] = 0
        loss_pose = loss_pose.mean() / self.norm_factor

        return loss_pose.mean()

class CDPN(nn.Module):
    def __init__(self, backbone, rot_head_net, trans_head_net):
        super(CDPN, self).__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.trans_head_net = trans_head_net
        self.monte_carlo_pose_loss = MonteCarloPoseLoss()

    def forward(self, x):                     # x.shape [bs, 3, 256, 256]
        features = self.backbone(x)           # features.shape [bs, 2048, 8, 8]
        cc_maps = self.rot_head_net(features) # joints.shape [bs, 1152, 64, 64]
        trans = self.trans_head_net(features)
        return cc_maps, trans
