# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointMLP(nn.Module):
    """
    MLP that processes image features and outputs point cloud, alphas, and weights.
    """
    def __init__(self, input_dim: int = 768, output_dim: int = 3, hidden_dim: int = 512, num_layers: int = 4):
        super(PointMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))  # Output point cloud with 3 dimensions (x, y, z)
        
        self.mlp = nn.Sequential(*layers)
        self.alpha_mlp = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)])
        self.weight_mlp = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)])
        
    def forward(self, x):
        # x: [batch_size, num_tokens, input_dim]
        point_cloud = self.mlp(x)
        alphas = self.alpha_mlp(x)
        weights = self.weight_mlp(x)

        return point_cloud, alphas, weights
