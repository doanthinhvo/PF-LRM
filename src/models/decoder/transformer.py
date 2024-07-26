# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PreprocessImageFeats(nn.Module):
    """
    Preprocess image features by increasing their dimension from 768 to 1024 using ConvTranspose2d.
    """
    def __init__(self, input_dim: int = 768, output_dim: int = 1024):
        super(PreprocessImageFeats, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        
    def forward(self, x):
        # x: [batch_size, num_tokens, input_dim]
        batch_size, num_tokens, input_dim = x.size()
        # Reshape to [batch_size, input_dim, num_tokens, 1] to use ConvTranspose2d
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv_transpose(x)
        # Reshape back to [batch_size, num_tokens, output_dim]
        x = x.squeeze(-1).permute(0, 2, 1)
        return x

class PostprocessImageFeats(nn.Module):
    """
    Postprocess image features by reducing their dimension from 1024 back to 768 using ConvTranspose2d.
    """
    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super(PostprocessImageFeats, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        
    def forward(self, x):
        # x: [batch_size, num_tokens, input_dim]
        batch_size, num_tokens, input_dim = x.size()
        # Reshape to [batch_size, input_dim, num_tokens, 1] to use ConvTranspose2d
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv_transpose(x)
        # Reshape back to [batch_size, num_tokens, output_dim]
        x = x.squeeze(-1).permute(0, 2, 1)
        return x


class BasicTransformerBlock(nn.Module):
    """
    Transformer block that uses only self-attention and an MLP.
    """
    def __init__(
        self, 
        inner_dim: int, 
        num_heads: int, 
        eps: float,
        attn_drop: float = 0., 
        attn_bias: bool = False,
        mlp_ratio: float = 4., 
        mlp_drop: float = 0.,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(inner_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x):
        # x: [N, L, D]
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TriplaneTransformer(nn.Module):
    """
    Transformer with condition that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(
        self, 
        inner_dim: int, 
        image_feat_dim: int,
        triplane_low_res: int, 
        triplane_high_res: int, 
        triplane_dim: int,
        num_layers: int, 
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, inner_dim) * (1. / inner_dim) ** 0.5)
        self.layers = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim=inner_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.deconv = nn.ConvTranspose2d(inner_dim, triplane_dim, kernel_size=2, stride=2, padding=0)

        self.preprocess_image_feats = PreprocessImageFeats(input_dim=image_feat_dim, output_dim=inner_dim)
        self.postprocess_image_feats = PostprocessImageFeats(input_dim=inner_dim, output_dim=image_feat_dim)


    def forward(self, image_feats):
        # image_feats: [N, L_cond, D_cond]
        # N: batch_size, L_cond: num_tokens of all images, D_cond: dim

        N = image_feats.shape[0]
        H = W = self.triplane_low_res
        L = 3 * H * W
        image_feats = self.preprocess_image_feats(image_feats) # [batch_size, num_tokens, dim]
        
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D] # [batch_size, num_tokens, dim]

        # concat x and image_feats
        x_concat = torch.cat([x, image_feats], dim=-2)

        # image_feats has shape torch.Size([1, 1203, 768])
        # x has shape torch.Size([1, 3072, 1024])
        # concat image_feats and x
        # image_feats = image_feats.unsqueeze(1).expand(-1, L, -1)
        # x = torch.cat([x, image_feats
        #                   ], dim=-1)

        for layer in self.layers:
            x_concat = layer(x_concat) # 
        x_concat = self.norm(x_concat)

        # seperate x_concat to x and image_feats
        x = x_concat[:, :L, :]
        image_feats = x_concat[:, L:, :]
        image_feats = self.postprocess_image_feats(image_feats)
        # separate each plane and apply deconv
        x = x.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.deconv(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()

        # TODO: Fix this.
        return x, image_feats
