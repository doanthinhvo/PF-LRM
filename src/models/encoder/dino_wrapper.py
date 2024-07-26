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


import torch.nn as nn
from transformers import ViTImageProcessor
from einops import rearrange, repeat
from .dino import ViTModel
import torch

class DinoWrapper(nn.Module):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.instrinsic_embeder = nn.Sequential(
            nn.Linear(4, self.model.config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=True)
        )
        self.reference_view_embedding = nn.Parameter(torch.randn(1, self.model.config.hidden_size))
        self.source_view_embedding = nn.Parameter(torch.randn(1, self.model.config.hidden_size))
        if freeze:
            self._freeze()

    # def forward(self, image, camera):
    def forward(self, image, instrinsics):
        # image: [B, N (num_images), C, H, W]
        # camera: [B, N num_images, D (dimension)]
        # RGB image with [0,1] scale and properly sized
        if image.ndim == 5:
            image = rearrange(image, 'b n c h w -> (b n) c h w')
        dtype = image.dtype
        inputs = self.processor(
            images=image.float(), 
            return_tensors="pt", 
            do_rescale=False, 
            do_resize=False,
        ).to(self.model.device).to(dtype)
        # embed camera
        bs = instrinsics.shape[0]
        N = instrinsics.shape[1] # Num views

        reference_embeddings = self.reference_view_embedding.expand(bs, 1, self.model.config.hidden_size)
        source_embeddings = self.source_view_embedding.expand(bs, N-1, self.model.config.hidden_size)

        reference_source_embeddings = torch.cat([reference_embeddings, source_embeddings], dim=1)        

        instrinsics_embeddings = self.instrinsic_embeder(instrinsics) # torch.Size([1, 3, 768])
        
        camera_embeddings = instrinsics_embeddings + reference_source_embeddings
        camera_embeddings = rearrange(camera_embeddings, 'b n d -> (b n) d')
        embeddings = camera_embeddings # torch.Size([3, 768])
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model(**inputs, adaln_input=embeddings, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state

        # skip the first token (CLS)
        last_hidden_states = last_hidden_states[:, 1:]
        return last_hidden_states

    def _freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
            processor = ViTImageProcessor.from_pretrained(model_name)
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
