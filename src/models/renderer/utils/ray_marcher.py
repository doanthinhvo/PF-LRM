# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Modified by Jiale Xu
# The modifications are subject to the same license as the original.


"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MipRayMarcher2(nn.Module):
    def __init__(self, activation_factory):
        super().__init__()
        self.activation_factory = activation_factory

    def run_forward(self, colors, densities, depths, sample_coordinates, rendering_options, normals=None):
        dtype = colors.dtype
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2 # lay gia tri mau trung binh o giua 
        # torch.Size([1, 400, 47, 3])
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        # torch.Size([1, 400, 47, 1])
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        sample_coordinates_mid = (sample_coordinates[:, :, :-1] + sample_coordinates[:, :, 1:]) / 2 # Phải scale lại nếu không thì ko nhân được với delta

        # using factory mode for better usability
        densities_mid = self.activation_factory(rendering_options)(densities_mid).to(dtype)

        density_delta = densities_mid * deltas

        # (1 − exp(−σ kδ k)) in first equation of equation 3
        alpha = 1 - torch.exp(-density_delta).to(dtype)
        
        # cai nay de tinh tau_k o equation 3. 
        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        
        # wow, cumprod la cai tinh tau_k
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]   
        weights = weights.to(dtype)

        tau = torch.cumprod(alpha_shifted, -2)[:, :, :-1]  
        tau = tau.to(dtype)

        composite_rgb = torch.sum(weights * colors_mid, -2)
        composite_point = torch.sum(weights * sample_coordinates_mid, -2)
        weight_total = weights.sum(2)
        tau = tau[:, :, -1, :] # because in formular (7) it's big K, and in the formular of tau is already cumprod, so only take last value. 
        # composite_depth = torch.sum(weights * depths_mid, -2) / weight_total
        composite_depth = torch.sum(weights * depths_mid, -2)

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf')).to(dtype)
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        # rendered value scale is 0-1, comment out original mipnerf scaling
        # composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights, composite_point, tau


    def forward(self, colors, densities, depths, sample_coordinates, rendering_options):

        # Gop cac sample tren ray lai de thanh final color value cho ray do. 
        composite_rgb, composite_depth, weights, composite_point, tau = self.run_forward(colors, densities, depths, sample_coordinates, rendering_options)
        return composite_rgb, composite_depth, weights, composite_point, tau
        # composite_point is actually a coarse point cloud (400 points) of 1 view.
