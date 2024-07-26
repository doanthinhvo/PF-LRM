import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_lightning as pl
from einops import rearrange, repeat

from src.utils.train_util import instantiate_from_config, get_central_point_of_patch
from src.pnp.epropnpwrapper import EproPnPLossWrapper

class MVRecon(pl.LightningModule):
    def __init__(
        self,
        lrm_generator_config,
        lrm_path=None,
        input_size=256,
        render_size=192,
    ):
        super(MVRecon, self).__init__()

        self.input_size = input_size
        self.render_size = render_size

        # init modules
        self.lrm_generator = instantiate_from_config(lrm_generator_config)
        if lrm_path is not None:
            lrm_ckpt = torch.load(lrm_path)
            self.lrm_generator.load_state_dict(lrm_ckpt['weights'], strict=False)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.validation_step_outputs = []
        self.epropnpLoss = EproPnPLossWrapper()
    
    def on_fit_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        lrm_generator_input = {}
        render_gt = {} # for supervision

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        # input cameras and render cameras
        input_c2ws = batch['input_c2ws'].flatten(-2) # torch.Size([1, 3, 4, 4]) -> 1, 3, 16
        input_Ks = batch['input_Ks'].flatten(-2) # torch.Size([1, 3, 3, 3]) -> torch.Size([1, 3, 9])
        target_c2ws = batch['target_c2ws'].flatten(-2)
        target_Ks = batch['target_Ks'].flatten(-2)
        render_cameras_input = torch.cat([input_c2ws, input_Ks], dim=-1)
        render_cameras_target = torch.cat([target_c2ws, target_Ks], dim=-1)
        render_cameras = torch.cat([render_cameras_input, render_cameras_target], dim=1)

        input_extrinsics = input_c2ws[:, :, :12] # torch.Size([1, 3, 12])
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1) # torch.Size([1, 3, 4])
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1) # torch.Size([1, 3, 16])

        # add noise to input cameras
        cameras = cameras + torch.rand_like(cameras) * 0.04 - 0.02
        input_intrinsics = input_intrinsics + torch.rand_like(input_intrinsics) * 0.04 - 0.02

        lrm_generator_input['cameras_intrinsics'] = input_intrinsics.to(self.device)

        lrm_generator_input['cameras'] = cameras.to(self.device)
        lrm_generator_input['render_cameras'] = render_cameras.to(self.device)

        # target images
        target_images = torch.cat([batch['input_images'], batch['target_images']], dim=1)
        target_alphas = torch.cat([batch['input_alphas'], batch['target_alphas']], dim=1)

        # random crop
        render_size = np.random.randint(self.render_size, 513)
        target_images = v2.functional.resize(
            target_images, render_size, interpolation=3, antialias=True).clamp(0, 1)

        target_alphas = v2.functional.resize(
            target_alphas, render_size, interpolation=0, antialias=True)

        crop_params = v2.RandomCrop.get_params(
            target_images, output_size=(self.render_size, self.render_size))
        # crop_params = None
        target_images = v2.functional.crop(target_images, *crop_params)
        target_alphas = v2.functional.crop(target_alphas, *crop_params)[:, :, 0:1]

        lrm_generator_input['render_size'] = render_size
        lrm_generator_input['crop_params'] = crop_params

        render_gt['target_images'] = target_images.to(self.device)
        render_gt['target_alphas'] = target_alphas.to(self.device)
        render_gt['input_extrinsics'] = input_extrinsics.to(self.device)
        render_gt['input_intrinsics'] = input_Ks.to(self.device)

        return lrm_generator_input, render_gt
    
    def prepare_validation_batch_data(self, batch):
        lrm_generator_input = {}

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        input_c2ws = batch['input_c2ws'].flatten(-2)
        input_Ks = batch['input_Ks'].flatten(-2)

        input_extrinsics = input_c2ws[:, :, :12]
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        lrm_generator_input['cameras_intrinsics'] = input_intrinsics.to(self.device)
        lrm_generator_input['cameras'] = cameras.to(self.device)

        render_c2ws = batch['render_c2ws'].flatten(-2)
        render_Ks = batch['render_Ks'].flatten(-2)
        render_cameras = torch.cat([render_c2ws, render_Ks], dim=-1)

        lrm_generator_input['render_cameras'] = render_cameras.to(self.device)
        lrm_generator_input['render_size'] = 384
        lrm_generator_input['crop_params'] = None

        return lrm_generator_input
    
    def forward_lrm_generator(
        self, 
        images, 
        cameras, 
        render_cameras, 
        render_size=192, 
        crop_params=None, 
        chunk_size=1,
    ):
        planes, image_feats = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.forward_planes, 
            images, 
            cameras, 
            use_reentrant=False,
        )
        bs, num_images,num_tokens_per_image, dim = images.shape[0], images.shape[1], image_feats.shape[1] // images.shape[1], image_feats.shape[2]
        image_feats = image_feats.view(bs, num_images, num_tokens_per_image, dim)

        # predict pose features from first branch.
        pose_features = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.pointmlp,
            image_feats,
        )


        frames = []
        for i in range(0, render_cameras.shape[1], chunk_size):
            if i < images.shape[1]: 
                nerf_supervised = True
                render_size_nerf_supervised = images.shape[-1]
            else: 
                nerf_supervised = False
                render_size_nerf_supervised = None

            frames.append(
                torch.utils.checkpoint.checkpoint(
                    self.lrm_generator.synthesizer,
                    planes,
                    cameras=render_cameras[:, i:i+chunk_size],
                    render_size=render_size, 
                    crop_params=crop_params,
                    nerf_supervised=nerf_supervised,
                    render_size_nerf_supervised=render_size_nerf_supervised,
                    use_reentrant=False,
                )
            )

        frames = {
            k: torch.cat([r[k] for r in frames], dim=1)
            for k in frames[0].keys()
        }
        return frames, pose_features
    
    def forward(self, lrm_generator_input):
        # Input images 
        images = lrm_generator_input['images'] 
        # Input cameras intrinsics
        cameras = lrm_generator_input['cameras_intrinsics'] 

        render_cameras = lrm_generator_input['render_cameras']
        render_size = lrm_generator_input['render_size']
        crop_params = lrm_generator_input['crop_params'] 


        # pose_features predict from the first branch. 
        out, pose_features = self.forward_lrm_generator(
            images, 
            cameras, 
            render_cameras, 
            render_size=render_size, 
            crop_params=crop_params, 
            chunk_size=1,
        )
        central_patch_coordinates = get_central_point_of_patch(images.shape[-1], patch_size=16)
        # expand by input batch size
        central_patch_coordinates = central_patch_coordinates.expand(images.shape[0], images.shape[1], -1, -1)
        # For second branch
        render_images = torch.clamp(out['images_rgb'], 0.0, 1.0)
        render_alphas = torch.clamp(out['images_weight'], 0.0, 1.0)

        # For first branch
        point_pred = pose_features[0] # shape [B, N, 3]
        point_pred_alpha = pose_features[1] # shape [B, N, 3]
        point_pred_weight = pose_features[2] # shape [B, N, 3]

        
        gt_point_clouds_from_nerf = out['triplane_point_samples'][:, :3, :, :] # shape [B, N, 3]
        nerf_tau = out['tau'][:, :3, :, :] # shape [B, N, 1] # volume transmistance

        out = {
            'render_images': render_images,
            'render_alphas': render_alphas,
            'point_clouds': point_pred,
            'central_patch_coordinates': central_patch_coordinates,
            'point_clouds_alpha': point_pred_alpha,
            'point_clouds_weight': point_pred_weight,
            'gt_point_clouds_from_nerf': gt_point_clouds_from_nerf,
            'gt_tau_from_nerf': nerf_tau
        }
        print(f"render_images: {render_images.shape} render_alphas: {render_alphas.shape} point_clouds: {point_pred.shape} point_clouds_alpha: {point_pred_alpha.shape} point_clouds_weight: {point_pred_weight.shape} gt_point_clouds_from_nerf: {gt_point_clouds_from_nerf.shape} gt_tau_from_nerf: {nerf_tau.shape}")
        return out

    def training_step(self, batch, batch_idx):
        lrm_generator_input, render_gt = self.prepare_batch_data(batch)

        render_out = self.forward(lrm_generator_input)

        loss, loss_dict = self.compute_loss(render_out, render_gt)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            B, N, C, H, W = render_gt['target_images'].shape
            N_in = lrm_generator_input['images'].shape[1]

            input_images = v2.functional.resize(
                lrm_generator_input['images'], (H, W), interpolation=3, antialias=True).clamp(0, 1)
            input_images = torch.cat(
                [input_images, torch.ones(B, N-N_in, C, H, W).to(input_images)], dim=1)

            input_images = rearrange(
                input_images, 'b n c h w -> b c h (n w)')
            target_images = rearrange(
                render_gt['target_images'], 'b n c h w -> b c h (n w)')
            render_images = rearrange(
                render_out['render_images'], 'b n c h w -> b c h (n w)')
            target_alphas = rearrange(
                repeat(render_gt['target_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            render_alphas = rearrange(
                repeat(render_out['render_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')

            grid = torch.cat([
                input_images, 
                target_images, render_images, 
                target_alphas, render_alphas, 
            ], dim=-2)
            grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))

            save_image(grid, os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png'))

        return loss
    
    def compute_loss(self, render_out, render_gt):
        # NOTE: the rgb value range of OpenLRM is [0, 1]
        render_images = render_out['render_images']
        target_images = render_gt['target_images'].to(render_images)
        render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0

        loss_mse = 1.0 * F.mse_loss(render_images, target_images)
        loss_lpips = 2.0 * self.lpips(render_images, target_images)

        loss_C = loss_mse + loss_lpips
        loss_P = 1 * F.mse_loss(render_out['point_clouds'], render_out['gt_point_clouds_from_nerf'])
        loss_Alpha = 1 * F.mse_loss(render_out['point_clouds_alpha'], 1 - render_out['gt_tau_from_nerf'])
        loss_Y = 1 *  self.epropnpLoss(render_out['point_clouds'], render_out['central_patch_coordinates'], render_out['point_clouds_alpha'] * render_out['point_clouds_weight'], render_gt['input_intrinsics'],render_gt['input_extrinsics'], device=self.device)
        loss = loss_C + loss + loss_P + loss_Alpha + loss_Y 

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_mse': loss_mse})
        loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        lrm_generator_input = self.prepare_validation_batch_data(batch)

        render_out = self.forward(lrm_generator_input)
        render_images = render_out['render_images']
        render_images = rearrange(render_images, 'b n c h w -> b c h (n w)')

        self.validation_step_outputs.append(render_images)
    
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=-1)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            image_path = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png')

            grid = make_grid(all_images, nrow=1, normalize=True, value_range=(0, 1))
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        lr = self.learning_rate

        params = []

        params.append({"params": self.lrm_generator.parameters(), "lr": lr, "weight_decay": 0.01 })

        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.90, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=lr/10)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
