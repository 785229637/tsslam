from dataclasses import dataclass
from typing import Dict, Literal, Optional, List

from networkx import generate_pajek
import numpy as np
from scipy import ndimage
import torch
from einops import rearrange
from jaxtyping import Float
from torch import BoolTensor, Tensor, nn
import torch.nn.functional as F
from collections import OrderedDict
# from encoder_wb.PoseEstimation import PoseEstimationHead

from transplat.src.dataset.shims.bounds_shim import apply_bounds_shim
from transplat.src.dataset.shims.patch_shim import apply_patch_shim
from transplat.src.dataset.types import BatchedExample, DataShim
from transplat.src.geometry.projection import sample_image_grid
from transplat.src.model.types import Gaussians
from encoder_wb.backbone.backbone_multiview import BackboneMultiview
from encoder_wb.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from encoder_wb.matching.depth_predictor_trans import DepthPredictorTrans
from encoder_wb.visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from transplat.src.global_cfg import get_cfg

from transplat.src.depth_anything_v2.dpt import DepthAnythingV2

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderTransCfg:
    name: Literal["trans"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_cost_volume_refine: bool


class EncoderTrans(Encoder[EncoderTransCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorTrans
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderTransCfg) -> None:
        super().__init__(cfg)


        self.novelty_detector = nn.Sequential(
            nn.Conv2d(cfg.d_feature, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        

        self.flow_estimator = nn.Sequential(
            nn.Conv2d(cfg.d_feature*2, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )
        

        self.register_buffer('prev_features', None)

        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                self.backbone.load_state_dict(updated_state_dict, strict=False)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitb' # or 'vits', 'vitb', 'vitl'
        DA_size = model_configs[encoder]['features'] // 2

        self.da_model = DepthAnythingV2(**model_configs[encoder])
        self.da_model.load_state_dict(torch.load(f'transplat/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.da_model = self.da_model.eval()
        
        for param in self.da_model.parameters():
            param.requires_grad = False

        self.depth_predictor = DepthPredictorTrans(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            # wo_depth_refine=cfg.wo_depth_refine,
            # wo_cost_volume=cfg.wo_cost_volume,
            # wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            DA_size=DA_size,
        )

        # self.pose_head = PoseEstimationHead(cfg.d_feature*64*64).to("cuda:0")
        # self.pose_head = PoseEstimationHead(int(cfg.d_feature*144*144/2)).to("cuda:0")


    def _project_gaussians(self, means: Tensor, context: dict) -> Tensor:
        b, v, _, h, w = context["image"].shape
        

        c2w = context["extrinsics"]  # [B, V, 4, 4]
        w2c = torch.inverse(c2w)     # [B, V, 4, 4]
        K = context["intrinsics"][:, :, :3, :3]  # [B, V, 3, 3]
        

        homogeneous = torch.cat([means, torch.ones_like(means[..., :1])], dim=-1)
        cam_coords = torch.einsum('bvij,bgkj->bvgki', w2c[:, :, :3], homogeneous)
        

        pix_coords = torch.einsum('bvij,bvgkj->bvgki', K, cam_coords)
        pix_coords = pix_coords[..., :2] / pix_coords[..., 2:3]
        


        pix_coords_x = (pix_coords[..., 0] / w) * 2 - 1
        pix_coords_y = (pix_coords[..., 1] / h) * 2 - 1
        pix_coords = torch.stack([pix_coords_x, pix_coords_y], dim=-1)
        
        return pix_coords



    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def _generate_region_mask(
        self,
        features: Tensor,
        existing_means: Optional[Tensor],
        context: dict,
        h: int,
        w: int,
        device: torch.device
    ) -> Tensor:
        batch_size = features.size(0)
        
        if existing_means is not None:

            proj_coords = self._project_gaussians(existing_means, context)
            
            current_depth = features.mean(dim=1)  # [B, H, W]
            current_depth = F.interpolate(
                current_depth.unsqueeze(1), 
                size=(h, w), 
                mode='bilinear', 
                align_corners=True
            ).squeeze(1)
            
            proj_depth = F.grid_sample(
                current_depth.unsqueeze(1),
                proj_coords,
                align_corners=True
            ).squeeze(1)  # [B, H, W]
            
            depth_diff = (current_depth - proj_depth).detach()
            depth_diff_net = depth_diff.detach()
            

            threshold = torch.quantile(
                depth_diff_net.view(batch_size, -1),
                0.9, 
                dim=1
            ).view(batch_size, 1, 1, 1)
            
            mask = (depth_diff.unsqueeze(1) > threshold).float()  # [B, 1, H, W]
        
        elif self.prev_features is not None:

            resized_features = F.interpolate(features, scale_factor=1.0, mode='bilinear')
            resized_prev = F.interpolate(self.prev_features, scale_factor=1.0, mode='bilinear')
            
            flow = self.flow_estimator(torch.cat([resized_prev, resized_features], dim=1))  # [B, 2, H, W]
            flow_magnitude = torch.norm(flow, dim=1, keepdim=True)  # [B, 1, H, W]
            
            flow_mag_net = flow_magnitude.detach()
            threshold = torch.quantile(
                flow_mag_net.view(batch_size, -1),
                0.95,
                dim=1
            ).view(batch_size, 1, 1, 1)
            
            mask = (flow_magnitude > threshold).float()  # [B, 1, H, W]
            self.prev_features = features.detach()
        
        else:

            novelty_map = self.novelty_detector(features)  # [B, 1, H, W]
            novelty_net = novelty_map.detach()
            
            threshold = torch.quantile(
                novelty_net.view(batch_size, -1),
                0.75,
                dim=1
            ).view(batch_size, 1, 1, 1)
            
            mask = (novelty_map > threshold).float()  # [B, 1, H, W]
        

        mask = F.interpolate(
            mask, 
            size=(h, w), 
            mode='nearest'
        )
        
        mask = F.max_pool2d(
            mask, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        return mask
    


    def _calculate_importance(self, gaussians: Gaussians) -> Tensor:

        det = torch.det(gaussians.covariances).abs().clamp_min(1e-6)
        volume = 1 / det.sqrt()
        return volume * gaussians.opacities.sigmoid()

    def _sample_indices(self, importance: Tensor, gpp: int) -> Tensor:
        b, _, h, w = importance.shape
        flat_imp = importance.view(b, -1)
        

        probs = flat_imp / flat_imp.sum(dim=-1, keepdim=True)
        indices = torch.multinomial(probs, gpp, replacement=True)
        
        return indices.view(b, h, w, gpp)

    def _sparse_gaussian_generation(
        self,
        raw_gaussians: Float[Tensor, "b v r c"],
        mask: BoolTensor,
        existing: Optional[Gaussians],
        context: dict,
        h: int,
        w: int
    ) -> Gaussians:
        device = raw_gaussians.device
        b, v, r, c = raw_gaussians.shape
        


        spatial_mask = rearrange(mask, "b v h w -> b v (h w)")  # [b, v, r]
        active_mask = spatial_mask.any(dim=1)
        

        active_counts = active_mask.sum(dim=1)
        gpp = torch.clamp(active_counts // 50 + 1, 1, self.cfg.gaussians_per_pixel)
        

        with torch.no_grad():

            cov_params = raw_gaussians[..., 2:8].detach()
            importance = torch.norm(cov_params, dim=-1).sum(dim=1)  # [b, r]
            importance[~active_mask] = -float('inf')
            

            all_indices = []
            for batch_idx in range(b):
                batch_imp = importance[batch_idx]
                indices = torch.topk(batch_imp, gpp[batch_idx]).indices
                all_indices.append(indices + batch_idx * r)
            
            indices = torch.cat(all_indices)



        flat_gaussians = rearrange(raw_gaussians, "b v r c -> (b r v) c")
        flat_depths = rearrange(context["depths"], "b v r srf -> (b r v srf)")
        flat_opacities = rearrange(context["opacities"], "b v r srf -> (b r v srf)")
        

        selected_gaussians = flat_gaussians[indices]
        selected_depths = flat_depths[indices]
        selected_opacities = flat_opacities[indices]
        

        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        selected_xy = xy_ray.expand(b, v, r, 1, 2).reshape(-1, 2)[indices]


        extrinsics = rearrange(context["extrinsics"], "b v i j -> (b v) i j")
        intrinsics = rearrange(context["intrinsics"], "b v i j -> (b v) i j")
        
        return self.gaussian_adapter(
            extrinsics=extrinsics[indices // (v * r)],
            intrinsics=intrinsics[indices // (v * r)],
            coordinates=selected_xy,
            depths=selected_depths,
            opacities=selected_opacities,
            raw_gaussians=selected_gaussians,
            image_shape=(h, w)
        )

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def forward(
        self,
        context: dict,
        global_step: int,
        existing_gaussians: Optional[Gaussians] = None,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        pose_ture = False
    ) -> Gaussians:







        torch.autograd.set_detect_anomaly(True)

        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        intr_curr = context["intrinsics"][:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
        intr_curr[:, :, 0, :] *= float(w)
        intr_curr[:, :, 1, :] *= float(h)
        camk = torch.eye(4).view(1,1,4,4).repeat(intr_curr.shape[0], intr_curr.shape[1], 1, 1).to(intr_curr.device).float()
        camk[:,:,:3,:3] = intr_curr
        c2w = context["extrinsics"].clone().detach()

        camk = torch.inverse(camk)
        img2world = torch.matmul(c2w, camk)
        # img2world=None








        with torch.no_grad():
            da_images = self.normalize_images(context["image"])
            da_images = da_images[:,:,[2, 0, 1]]
            b, v, c, h, w = da_images.shape
            da_images = da_images.view(b*v, c, h, w)
            da_images = F.interpolate(da_images, (252, 252), mode="bilinear", align_corners=True)
            da_depth, out_feature = self.da_model.forward(da_images)
            da_depth = F.interpolate(da_depth[None], (h, w), mode="bilinear", align_corners=True)
            da_depth = da_depth.view(b, v, 1, h, w)
            # normalize to 0 - 1
            da_depth = da_depth.flatten(2)
            da_max = torch.max(da_depth, dim=-1, keepdim=True)[0]
            da_min = torch.min(da_depth, dim=-1, keepdim=True)[0]
            da_depth = (da_depth - da_min) / (da_max - da_min)
            da_depth = da_depth.reshape(b, v, 1, h, w)

        dino_feature = out_feature.view(b, v, out_feature.shape[1], out_feature.shape[2], out_feature.shape[3])

        trans_features, cnn_features , updated_pose,delta_T = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            camk=camk,
            c2w=c2w,
            dino_feature = dino_feature[0],
            pose_ture = pose_ture
        )
        out_ex=update_extrinsics(context["extrinsics"], updated_pose)
        


        # show depthanything result here
        '''
        import cv2
        lowres_da_depth = F.interpolate(
            da_depth[0],
            scale_factor=0.25,
            mode="bilinear",
            align_corners=True,
        )

        depth_vis = lowres_da_depth[0,0].cpu().numpy()
        depth_vis = depth_vis * 255
        depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('vis_lor.png', depth_rgb)

        image_vis = context["image"][0,0].permute(1, 2, 0)
        image_vis = image_vis.cpu().numpy()
        image_vis = image_vis * 255
        cv2.imwrite('image.png', image_vis)
        import pdb
        pdb.set_trace()
        '''

        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel

        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            out_ex,
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
            da_depth=da_depth,
            dino_feature=dino_feature,
        )


        # region_mask = self._generate_region_mask(
        #             features=trans_features[0],
        #             existing_means=existing_gaussians.means if existing_gaussians else None,

        #             h=h, w=w, device=device
        #         )
        # if existing_gaussians != None:
        if existing_gaussians != None:



            
            new_object_mask_tensor = generate_mask(context, cnn_features, depths)
            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(context["image"][0, 0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.title("Image 1")

            # plt.subplot(1, 2, 2)
            # plt.imshow(context["image"][0, 1].permute(1, 2, 0).detach().cpu().numpy())
            # plt.title("Image 2")

            # plt.figure()
            # plt.imshow(new_object_mask_tensor.squeeze().detach().cpu().numpy(), cmap='hot')
            # plt.title("Difference Mask")
            # plt.colorbar()
            # plt.show(block=True)
            mask = new_object_mask_tensor.reshape(-1, 1, 1)
            mask_bool = mask.view(-1).bool()
            xy_ray, _ = sample_image_grid((h, w), device)

            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )

            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

            xy_ray =  xy_ray[:, 1:2, mask_bool, :, :]
            gaussians = gaussians[:, 1:2, mask_bool, :, :]


            gpp = self.cfg.gaussians_per_pixel




            gaussians = self.gaussian_adapter.forward(
                rearrange(out_ex[0][1].unsqueeze(0).unsqueeze(0), "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"][0][1].unsqueeze(0).unsqueeze(0), "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths[:, 1:2, mask_bool, :, :],
                (self.map_pdf_to_opacity(densities, global_step) / gpp)[:, 1:2, mask_bool, :, :],
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
            )




        else:


            xy_ray, _ = sample_image_grid((h, w), device)

            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )

            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
            gpp = self.cfg.gaussians_per_pixel

            gpp = self.cfg.gaussians_per_pixel



            gaussians = self.gaussian_adapter.forward(
                rearrange(out_ex, "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
            )


 

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1



        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ).to("cuda"),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        ),out_ex ,delta_T

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            # TTRANS Do not patch now
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None




# def generate_mask_depth(context, da_depth, threshold=0.2):
#     """







#     """
#     device = context["image"].device
#     images = context["image"]  # [1, 2, 3, H, W]
#     b, v, _, h, w = images.shape
    

#     if b != 1 or v != 2:

    

#     img1 = images[0, 0]  # [3, H, W]
#     img2 = images[0, 1]  # [3, H, W]
    

#     depth1 = da_depth[0, 0, 0]  # [H, W]
#     depth2 = da_depth[0, 1, 0]  # [H, W]
    

#     color_diff = torch.mean(torch.abs(img1 - img2), dim=0)  # [H, W]

    

#     depth_diff = torch.abs(depth1 - depth2)
#     depth_mask = (depth_diff > threshold).float()
    


#     K1 = context["intrinsics"][0, 0, :3, :3].clone().detach()
#     K1[0] *= w
#     K1[1] *= h
#     w2c1 = torch.inverse(context["extrinsics"][0, 0])
#     w2c2 = torch.inverse(context["extrinsics"][0, 1])
    

#     rel_pose = w2c2 @ torch.inverse(w2c1)
    

#     combined_mask = (color_mask * depth_mask)
    

#     kernel = torch.ones(5, 5, device=device)
#     combined_mask = F.max_pool2d(combined_mask.unsqueeze(0), kernel_size=5, stride=1, padding=2)[0]
#     combined_mask = F.avg_pool2d(combined_mask.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    

#     filtered_mask = (combined_mask > 0.5).float()
    

#     new_object_mask = filtered_mask
    
#     return new_object_mask


def generate_mask(context, cnn_features,depth, threshold=0.9, depth_threshold=0.95):
    device = context["image"].device
    images = context["image"]  # [1, 2, 3, H, W]
    b, v, _, h, w = images.shape
    

    if b != 1 or v != 2:
        raise ValueError("只支持batch_size=1且两个视图的输入")
    

    img1, img2 = images[0, 0], images[0, 1]  # [3, H, W]
    feat1, feat2 = cnn_features[0, 0], cnn_features[0, 1]  # [128, 64, 64]
    


    feat1_flat = feat1.reshape(128, -1).permute(1, 0)  # [4096, 128]
    feat2_flat = feat2.reshape(128, -1).permute(1, 0)  # [4096, 128]
    feat1_norm = F.normalize(feat1_flat, p=2, dim=1)
    feat2_norm = F.normalize(feat2_flat, p=2, dim=1)
    

    sim_matrix = torch.mm(feat1_norm, feat2_norm.t())  # [4096, 4096]
    max_sim, _ = torch.max(sim_matrix, dim=0)
    

    unmatched_mask = (max_sim < threshold).reshape(64, 64).float()
    unmatched_mask = F.interpolate(unmatched_mask[None, None], size=(h, w), 
                                  mode='bilinear', align_corners=False)[0, 0]
    


    K1 = context["intrinsics"][0, 0, :3, :3].clone().detach()
    K1[0] *= w
    K1[1] *= h
    w2c1 = torch.inverse(context["extrinsics"][0, 0])
    w2c2 = torch.inverse(context["extrinsics"][0, 1])
    

    rel_pose = w2c2 @ torch.inverse(w2c1)
    T_cam2_cam1 = torch.inverse(rel_pose)
    

    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32)
    )
    uv_homo = torch.stack([x_coords + 0.5, y_coords + 0.5, torch.ones_like(x_coords)], dim=0)  # [3, H, W]
    uv_homo_flat = uv_homo.reshape(3, -1)  # [3, H*W]
    

    avg_depth = depth.mean()
    X_cam2 = avg_depth * (torch.inverse(K1) @ uv_homo_flat)
    X_cam2_homo = torch.cat([X_cam2, torch.ones(1, h*w, device=device)], dim=0)
    

    X_cam1_homo = T_cam2_cam1 @ X_cam2_homo
    X_cam1 = X_cam1_homo[:3]  # [3, H*W]
    

    x_proj = K1 @ X_cam1
    u_proj = x_proj[0] / x_proj[2]
    v_proj = x_proj[1] / x_proj[2]
    

    visible = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h) & (x_proj[2] > 0)
    motion_mask = (~visible).reshape(h, w).float()
    

    color_diff = torch.mean(torch.abs(img1 - img2), dim=0)
    color_mask = (color_diff > 0.3).float()
    

    combined_mask = unmatched_mask + motion_mask + color_mask
    combined_mask = torch.clamp(combined_mask, 0, 1)
    

    kernel = torch.ones(5, 5, device=device)
    combined_mask = F.max_pool2d(combined_mask[None, None], kernel_size=5, stride=1, padding=2)[0, 0]
    combined_mask = F.avg_pool2d(combined_mask[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    

    new_region_mask = (combined_mask > depth_threshold).float()
    
    return new_region_mask







def update_extrinsics(batch, updated_pose):

    extrinsics = batch.clone()
    
    
    
    

    updated_extrinsics = extrinsics.clone()
    updated_extrinsics[:, 1:, :, :] = updated_pose.unsqueeze(1)
    
    return updated_extrinsics
