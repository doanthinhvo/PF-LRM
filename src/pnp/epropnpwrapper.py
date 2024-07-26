import math
import torch
import numpy as np

import cv2
# from progress.bar import Bar
import os

import time
from .CDPN import MonteCarloPoseLoss
from .epropnp import EProPnP6DoF 
from .levenberg_marquardt import LMSolver, RSLMSolver
from .cost_fun import AdaptiveHuberPnPCost
from .camera import PerspectiveCamera
from .common import extrinsic_to_6dof

class EproPnPLossWrapper(torch.nn.Module):
    def __init__(self):
        super(EproPnPLossWrapper, self).__init__()
        
    def forward(self, x3d, x2d, w2d, instrinsics, input_extrinsics, device):
        x3d = x3d.view(-1, 400, 3).to(device)
        x2d = x2d.view(-1, 400, 2).to(device)
        w2d = w2d.view(-1, 400, 2).to(device)

        scale=torch.rand(1, 5, 2).to(device)
        cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
        epropnp = EProPnP6DoF(
            mc_samples=400, 
            num_iter=4,
            solver=LMSolver(
                dof=6,
                num_iter=5,
                init_solver=RSLMSolver(
                    dof=6,
                    num_points=16,
                    num_proposals=4,
                    num_iter=3))).to(device)
        pose_gt = extrinsic_to_6dof(input_extrinsics.view(-1, 3, 4)).to(device)
        cam_mats = instrinsics.view(-1, 3, 3).to(device) 
        camera = PerspectiveCamera(cam_mats=cam_mats)
        cost_fun.set_param(x2d, w2d)
        _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
            x3d, x2d, w2d, camera, cost_fun,
            pose_init=pose_gt, force_init_solve=True, with_pose_opt_plus=True)
        criterion = MonteCarloPoseLoss().to(device)
        loss_mc = criterion(pose_sample_logweights, cost_tgt, scale.detach().mean())
        return loss_mc