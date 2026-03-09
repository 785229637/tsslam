import time

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils

from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth

import os
import sys
from hydra import compose, initialize
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

module_path = os.path.abspath("/home/ben/G-3DGS/wb-2")
if module_path not in sys.path:
    sys.path.append(module_path)
from transplat.src.config import load_typed_root_config

sys.path.append(os.path.join(os.path.dirname(__file__), "transplat"))


from encoder_wb import get_encoder
from transplat.src.global_cfg import set_cfg, get_cfg
from omegaconf import DictConfig
from transplat.src.model.decoder import get_decoder



import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def compute_visible_ratio(gaussians_means, extrinsic_matrix, projection_matrix, image_width, image_height):

    device = gaussians_means.device
    extrinsic_matrix = extrinsic_matrix.to(device).float()
    projection_matrix = projection_matrix.to(device).float()
    

    N = gaussians_means.shape[1]
    

    points_3d = gaussians_means.squeeze(0)
    


    R = extrinsic_matrix[:3, :3]
    T = extrinsic_matrix[:3, 3]
    


    points_relative = points_3d - T.unsqueeze(0)  # [N, 3]
    points_camera = torch.matmul(points_relative, R.t())  # [N, 3]
    

    points_camera_homogeneous = torch.cat([points_camera, torch.ones(N, 1, device=device)], dim=1)  # [N, 4]
    points_proj_homogeneous = torch.matmul(points_camera_homogeneous, projection_matrix.t())  # [N, 4]
    

    w = points_proj_homogeneous[:, 3:4]
    points_proj = points_proj_homogeneous[:, :3] / w  # [N, 3]
    


    in_front = points_proj[:, 2] < 0
    

    x_in_range = (points_proj[:, 0] <= 0) & (points_proj[:, 0] > -image_width)
    y_in_range = (points_proj[:, 1] <= 0) & (points_proj[:, 1] > -image_height)
    

    visible_mask = in_front & x_in_range & y_in_range
    

    visible_ratio = visible_mask.sum().item() / N
    
    return visible_ratio, visible_mask, points_proj[:, :2]

def plot_3d_scene(gaussians_means, extrinsic_matrix, visible_mask=None, max_points=1000):

    R = extrinsic_matrix[:3, :3].cpu().numpy()
    T = extrinsic_matrix[:3, 3].cpu().numpy()
    

    points_3d = gaussians_means.squeeze(0).cpu().numpy()  # [N, 3]
    

    camera_dir = R @ np.array([0, 0, 1])
    camera_up = R @ np.array([1, 0, 0])
    camera_right = R @ np.array([0, 1, 0])
    

    N = points_3d.shape[0]
    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        points_3d = points_3d[indices]
        if visible_mask is not None:
            visible_mask = visible_mask[indices].cpu().numpy()
    

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    

    if visible_mask is not None:

        ax.scatter(points_3d[visible_mask, 0], points_3d[visible_mask, 1], points_3d[visible_mask, 2], 
                   c='green', s=2, alpha=0.6, label='可见点')
        ax.scatter(points_3d[~visible_mask, 0], points_3d[~visible_mask, 1], points_3d[~visible_mask, 2], 
                   c='red', s=2, alpha=0.3, label='不可见点')
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=2, alpha=0.5, label='点云')
    

    ax.scatter(T[0], T[1], T[2], c='yellow', s=100, marker='^', label='相机位置')
    

    all_points = np.vstack([points_3d, T])
    max_range = np.max(np.ptp(all_points, axis=0))
    line_length = max_range * 0.3
    

    line_end = T + camera_dir * line_length
    ax.plot([T[0], line_end[0]], 
            [T[1], line_end[1]], 
            [T[2], line_end[2]], 
            'b-', linewidth=2, alpha=0.8, label='视线方向')
    

    arrow_length = max_range * 0.1
    
    a_up = Arrow3D([T[0], T[0] + camera_up[0] * arrow_length * 0.5],
                   [T[1], T[1] + camera_up[1] * arrow_length * 0.5],
                   [T[2], T[2] + camera_up[2] * arrow_length * 0.5],
                   mutation_scale=10, lw=2, arrowstyle="-|>", color="green")
    ax.add_artist(a_up)
    
    a_right = Arrow3D([T[0], T[0] + camera_right[0] * arrow_length * 0.5],
                      [T[1], T[1] + camera_right[1] * arrow_length * 0.5],
                      [T[2], T[2] + camera_right[2] * arrow_length * 0.5],
                      mutation_scale=10, lw=2, arrowstyle="-|>", color="red")
    ax.add_artist(a_right)
    

    ax.text(line_end[0], line_end[1], line_end[2], '前', color='blue')
    ax.text(T[0] + camera_up[0] * arrow_length * 0.5,
            T[1] + camera_up[1] * arrow_length * 0.5,
            T[2] + camera_up[2] * arrow_length * 0.5,
            '上', color='green')
    ax.text(T[0] + camera_right[0] * arrow_length * 0.5,
            T[1] + camera_right[1] * arrow_length * 0.5,
            T[2] + camera_right[2] * arrow_length * 0.5,
            '右', color='red')
    

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('三维点云和相机位置与视线方向')
    ax.legend()
    

    max_range = np.max(np.ptp(all_points, axis=0)) / 2
    mid_x = np.mean(all_points[:, 0])
    mid_y = np.mean(all_points[:, 1])
    mid_z = np.mean(all_points[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def visualize_projection(projected_points, visible_mask, image_width, image_height, max_points=1000):

    projected_points = projected_points.cpu().numpy()
    visible_mask = visible_mask.cpu().numpy()
    

    N = projected_points.shape[0]
    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        projected_points = projected_points[indices]
        visible_mask = visible_mask[indices]
    

    plt.figure(figsize=(10, 10 * image_height / image_width))
    plt.scatter(projected_points[visible_mask, 0], image_height - projected_points[visible_mask, 1], 
                c='green', s=5, alpha=0.6, label='可见点投影')
    plt.scatter(projected_points[~visible_mask, 0], image_height - projected_points[~visible_mask, 1], 
                c='red', s=5, alpha=0.3, label='不可见点投影')
    

    plt.plot([0, 0, image_width, image_width, 0], 
             [0, image_height, image_height, 0, 0], 'k--', alpha=0.5)
    
    plt.xlim(-image_width * 0.1, image_width * 1.1)
    plt.ylim(-image_height * 0.1, image_height * 1.1)
    plt.xlabel('图像X坐标')
    plt.ylabel('图像Y坐标')
    plt.title('点在图像平面上的投影')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.show()


def load_decoder(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)

    # current_dir = os.path.dirname(os.path.abspath(__file__))

    # config_relative_path = os.path.join("transplat", "config")
    


    # cfg_dict = compose(config_name="main", overrides=[
    #     "+experiment=tum_rgbd",
    #     "dataset.view_sampler.num_context_views=2",

    # ])
    

    set_cfg(cfg_dict)
    

    if get_cfg() is None:
        raise ValueError("全局配置未正确初始化")
    if not hasattr(get_cfg(), 'mode'):
        raise ValueError("全局配置中缺少mode参数")
    

    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    return decoder

def load_encoder(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)

    # current_dir = os.path.dirname(os.path.abspath(__file__))

    # config_relative_path = os.path.join("transplat", "config")
    


    # cfg_dict = compose(config_name="main", overrides=[
    #     "+experiment=tum_rgbd",
    #     "dataset.view_sampler.num_context_views=2",

    # ])
    

    set_cfg(cfg_dict)
    

    if get_cfg() is None:
        raise ValueError("全局配置未正确初始化")
    if not hasattr(get_cfg(), 'mode'):
        raise ValueError("全局配置中缺少mode参数")
    

    encoder, _ = get_encoder(cfg.model.encoder)

    def fix_checkpoint_keys(ckpt_state_dict):
        new_state_dict = {}
        for k, v in ckpt_state_dict.items():

            k = k.replace("backbone.cam_param_reduce_conv", "backbone.cam_param_encoder.reduce_conv")
            k = k.replace("backbone.cam_param_context_conv", "backbone.cam_param_encoder.context_conv")
            k = k.replace("backbone.cam_param_bn", "backbone.cam_param_encoder.bn")
            k = k.replace("backbone.cam_param_layer", "backbone.cam_param_encoder.layer")
            k = k.replace("backbone.cam_param_context_mlp", "backbone.cam_param_encoder.context_mlp")
            k = k.replace("backbone.cam_param_context_se", "backbone.cam_param_encoder.context_se")
            

            k = k.replace("depth_predictor.cam_param_reduce_conv", "depth_predictor.cam_param_encoder.reduce_conv")
            k = k.replace("depth_predictor.cam_param_context_conv", "depth_predictor.cam_param_encoder.context_conv")
            k = k.replace("depth_predictor.cam_param_bn", "depth_predictor.cam_param_encoder.bn")
            k = k.replace("depth_predictor.cam_param_layer", "depth_predictor.cam_param_encoder.layer")
            k = k.replace("depth_predictor.cam_param_context_mlp", "depth_predictor.cam_param_encoder.context_mlp")
            k = k.replace("depth_predictor.cam_param_context_se", "depth_predictor.cam_param_encoder.context_se")
            

            k = k.replace("depth_predictor.coarse_transformer.layers", "depth_predictor.coarse_transformer.encoder.layers")
            k = k.replace("depth_predictor.fine_transformer.layers", "depth_predictor.fine_transformer.encoder.layers")
            
            new_state_dict[k] = v
        return new_state_dict


    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2026-02-09/21-38-21/checkpoints/epoch_24-step_10000.ckpt", map_location="cpu")
    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2026-01-14/17-32-42/checkpoints/epoch_68-step_6000.ckpt", map_location="cpu")#cables_2
    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2026-01-15/11-16-19/checkpoints/epoch_39-step_10000.ckpt", map_location="cpu")#sfm_lab_room_1
    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2026-01-15/16-34-31/checkpoints/epoch_99-step_6000.ckpt", map_location="cpu")#plant_1
    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2026-01-15/19-27-35/checkpoints/epoch_35-step_10000.ckpt", map_location="cpu")#einstein_1

    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2026-01-14/17-24-19/checkpoints/epoch_16-step_10000.ckpt", map_location="cpu")#planar_2
    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2025-12-24/14-29-56/checkpoints/epoch_4-step_8000.ckpt", map_location="cpu")#r0
    ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2025-08-21/11-02-12/checkpoints/epoch_5-step_20000.ckpt", map_location="cpu")#f3
    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/chackpoint/tum-f1.ckpt", map_location="cpu")#f1

    raw_encoder_weights = {
        k.replace("encoder.", ""): v 
        for k, v in ckpt["state_dict"].items() 
        if k.startswith("encoder.")
    }

    fixed_encoder_weights = fix_checkpoint_keys(raw_encoder_weights)

    encoder.load_state_dict(fixed_encoder_weights, strict=True)



    # ckpt = torch.load("/home/ben/G-3DGS/wb-2/outputs/2025-08-19/13-45-02/checkpoints/epoch_59-step_35500.ckpt", map_location="cpu")
    # # ckpt = torch.load("/home/ben/G-3DGS/wb-2/chackpoint/new_dp.ckpt", map_location="cpu")

    # encoder_weights = {
    #     k.replace("encoder.", ""): v 
    #     for k, v in ckpt["state_dict"].items() 
    #     if k.startswith("encoder.")
    # }

    # encoder.load_state_dict(encoder_weights)

    return encoder, cfg_dict


def encoder_gs(encoder,input_data,pose_ture = False,existing_gaussians=None):


    with torch.no_grad():
        # gaussians = encoder(
        #     context=input_data,

        #     deterministic=True
        # )
        gaussians, out_ex,T = encoder(
            context=input_data, 
            global_step=0, 
            existing_gaussians=existing_gaussians,
            deterministic=False,
            scene_names='0',
            pose_ture = False
        )
        # gaussians = transform_gaussians_to_cam(gaussians, input_data["extrinsics"][0][0])
    
    return gaussians, out_ex,T


def is_image_blurry(image_tensor, threshold):


    image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    image_np = image_np.astype(np.uint8)
    

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    

    variance = np.var(laplacian)
    
    return variance < threshold

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.gaussians_track = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")


        print("加载编码器...")

        with initialize(version_base=None, config_path="../../transplat/config"):
            cfg_dict = compose(
                config_name="main",
                overrides=["+experiment=tum_rgbd",
                # overrides=["+experiment=replica",
                           "checkpointing.load=/home/ben/G-3DGS/wb-2/outputs/2025-08-18/21-15-04/checkpoints/epoch_58-step_35000.ckpt",
                           "data_loader.train.batch_size=1", 
                           "dataset.view_sampler.num_context_views=2"]
            )
        encoder, encoder_cfg = load_encoder(cfg_dict)
        decoder = load_decoder(cfg_dict)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)


        print("encoder准备完毕")

    def encoder_input(self, cur_frame_idx,current_window):

        # valid_indices = []

        # for idx in range(cur_frame_idx, -1, -1):
        #     if idx in self.cameras and self.cameras[idx].original_image is not None:
        #         valid_indices.append(idx)

        #         if len(valid_indices) == 2:
        #             break


        valid_indices = [cur_frame_idx]


        found_element = None
        max_distance = -1
        max_element = None


        for idx in current_window:

            distance = abs(cur_frame_idx - idx)
            

            if found_element is None and distance > 3:
                found_element = idx
                break
            

            if distance > max_distance:
                max_distance = distance
                max_element = idx


        if found_element is not None:
            element_to_add = found_element
        else:
            element_to_add = max_element


        valid_indices.append(element_to_add)

        valid_indices = valid_indices[::-1]
        
        if len(valid_indices) < 2:

            inputer_encoder = None
        else:
            num_views = 2
            batch_size = 1
            device = self.device
            target_size = (256, 256)
            

            cams = [self.cameras[idx] for idx in valid_indices]
            

            images = []
            intrinsics = []
            for cam in cams:

                img = cam.original_image.clone()
                if img.dim() == 3 and img.shape[0] == 3:

                    original_height, original_width = img.shape[1], img.shape[2]
                    

                    scale_height = target_size[0] / original_height
                    scale_width = target_size[1] / original_width
                    


                    resized_img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    images.append(resized_img)
                    

                    K = torch.zeros(3, 3, device=device)

                    K[0, 0] = cam.fx * scale_width /target_size[1]
                    K[1, 1] = cam.fy * scale_height /target_size[0]

                    K[0, 2] = cam.cx * scale_width/target_size[1]
                    K[1, 2] = cam.cy * scale_height/target_size[0]
                    K[2, 2] = 1.0




                    # K[0, 2] = 127.5/256
                    # K[1, 2] = 127.5/256
                    # K[2, 2] = 1.0
                    intrinsics.append(K)
                else:

                    raise ValueError(f"Invalid image format for viewpoint {cam.uid}")
            

            images = torch.stack(images, dim=0).unsqueeze(0)
            

            intrinsics = torch.stack(intrinsics, dim=0).unsqueeze(0)
            

            extrinsics = []
            for cam in cams:

                w2c = torch.eye(4, device=device)
                w2c[:3, :3] = cam.R.T
                w2c[:3, 3] = -cam.R.T @ cam.T
                extrinsics.append(w2c)
            extrinsics = torch.stack(extrinsics, dim=0).unsqueeze(0)  # (1, 2, 4, 4)
            

            near = torch.tensor([[1, 1]], device=device)
            far = torch.tensor([[100.0, 100.0]], device=device)
            

            return {
                "image": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "near": near,
                "far": far,
                "index": torch.tensor([[0, 1]], device=device)
            }

    def encoder_input_init(self, viewpoint_1, viewpoint_2):

        valid_viewpoints = []
        for viewpoint in [viewpoint_1, viewpoint_2]:
            if viewpoint is not None and viewpoint.original_image is not None:
                valid_viewpoints.append(viewpoint)
        
        if len(valid_viewpoints) < 2:

            return None
        else:
            num_views = 2
            batch_size = 1
            device = self.device
            target_size = (256, 256)
            

            cams = valid_viewpoints
                    

            images = []
            intrinsics = []
            for cam in cams:

                img = cam.original_image.clone()
                if img.dim() == 3 and img.shape[0] == 3:

                    original_height, original_width = img.shape[1], img.shape[2]
                    

                    scale_height = target_size[0] / original_height
                    scale_width = target_size[1] / original_width
                    

                    resized_img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    images.append(resized_img)
                    

                    K = torch.zeros(3, 3, device=device)

                    K[0, 0] = cam.fx * scale_width /target_size[1]
                    K[1, 1] = cam.fy * scale_height /target_size[0]

                    K[0, 2] = cam.cx * scale_width/target_size[1]
                    K[1, 2] = cam.cy * scale_height/target_size[0]
                    K[2, 2] = 1.0
                    intrinsics.append(K)
                else:

                    raise ValueError(f"Invalid image format for viewpoint {cam.uid}")
            

            images = torch.stack(images, dim=0).unsqueeze(0)
            

            intrinsics = torch.stack(intrinsics, dim=0).unsqueeze(0)
            

            extrinsics = []
            for cam in cams:

                w2c = torch.eye(4, device=device)
                w2c[:3, :3] = cam.R_gt.T
                w2c[:3, 3] = -cam.R_gt.T @ cam.T_gt
                # w2c[:3, :3] = cam.R
                # w2c[:3, 3] = cam.T
                # w2c[:3, 3] = w2c[:3, 3] * 0.1
                extrinsics.append(w2c)
            extrinsics = torch.stack(extrinsics, dim=0).unsqueeze(0)  # (1, 2, 4, 4)
            

            near = torch.tensor([[1, 1]], device=device)
            far = torch.tensor([[100.0, 100.0]], device=device)
            

            return {
                "image": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "near": near,
                "far": far,
                "index": torch.tensor([[0, 5]], device=device)
            }



    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]

        self.kf_indices.append(cur_frame_idx)

        viewpoint = self.cameras[cur_frame_idx]

        gt_img = viewpoint.original_image.cuda()

        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:

                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:

                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]

        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0
        return initial_depth[0].numpy()


    def initialize(self, cur_frame_idx, viewpoint,projection_matrix ):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []


        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose

        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)


        if cur_frame_idx < 5:
            cur_frame_idx_encoder = cur_frame_idx+5
        else:
            cur_frame_idx_encoder = cur_frame_idx-5

        viewpoint_encoder = Camera.init_from_dataset(
            self.dataset, cur_frame_idx_encoder, projection_matrix
        )

        input_data = self.encoder_input_init(viewpoint,viewpoint_encoder)



        gaussians_encoder, _ ,_= encoder_gs(self.encoder,input_data,pose_ture = False,existing_gaussians=None)

        # output = self.decoder.forward(
        #     gaussians_encoder,
        #     # updated_extrinsics,
        #     input_data["extrinsics"],
        #     input_data["intrinsics"],
        #     input_data["near"],
        #     input_data["far"],
        #     (256, 256),
        #     depth_mode=None,
        # )
        if 0:

            w2c = torch.eye(4)
            w2c[:3, :3] = viewpoint.R
            w2c[:3, 3] = viewpoint.T
            gaussians_means = gaussians_encoder.means
            extrinsic_matrix = w2c
            image_width, image_height = viewpoint.image_width, viewpoint.image_height


            visible_ratio, visible_mask, projected_points = compute_visible_ratio(
                gaussians_means,
                extrinsic_matrix,
                projection_matrix,
                image_width,
                image_height
            )

            print(f"点云总点数: {gaussians_means.shape[1]}")
            print(f"可见点数: {visible_mask.sum().item()}")
            print(f"可见比例: {visible_ratio:.2%}")


            plot_3d_scene(gaussians_means, extrinsic_matrix, visible_mask)


            visualize_projection(projected_points, visible_mask, image_width, image_height)


        self.kf_indices = []

        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map,gaussians_encoder)
        self.reset = False


    def tracking(self, cur_frame_idx, viewpoint, out_ex_encoder):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]

        
        R_c2w = out_ex_encoder[0][1][..., :3, :3]
        T_c2w = out_ex_encoder[0][1][..., :3, 3]
        

        R_w2c = R_c2w.transpose(0,1)
        

        T_c2w = -torch.matmul(R_w2c, T_c2w)

        distance = torch.norm(prev.T - T_c2w)
        if cur_frame_idx < 10:
            prev = self.cameras[cur_frame_idx]
            viewpoint.update_RT(prev.R_gt, prev.T_gt)
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.update_RT(prev.R, prev.T)
        # else:
        #     viewpoint.update_RT(R_w2c, T_c2w)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )


        pose_optimizer = torch.optim.Adam(opt_params)

        for tracking_itr in range(self.tracking_itr_num):

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            


            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg


    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
        blur_check
    ):

        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]


        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]

        # def is_image_blurry(image_tensor, threshold):
        #     """






        #     """


        #     image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0

            

        #     gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            

        #     laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            

        #     variance = np.var(laplacian)
            
        #     return variance < threshold


        # blur_check = is_image_blurry(curr_frame.original_image,100)
        if blur_check:
            return False


        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        

        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth


        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap,gaussians_encoder):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap,gaussians_encoder]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map,gaussians_encoder):
        msg = ["init", cur_frame_idx, viewpoint, depth_map,gaussians_encoder]
        self.backend_queue.put(msg)
        self.requested_init = True


    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()


    def run(self):
        cur_frame_idx = 0

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)



        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break


                if self.requested_init: 
                    time.sleep(0.01)
                    continue
                

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)


                self.cameras[cur_frame_idx] = viewpoint


                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint,projection_matrix )
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1

                    continue
                


                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )


                # Tracking
                input_data = self.encoder_input(cur_frame_idx,self.current_window)
                # if cur_frame_idx < 5:
                #     cur_frame_idx_encoder = cur_frame_idx+2
                # else:
                #     cur_frame_idx_encoder = cur_frame_idx-2

                # viewpoint_encoder = Camera.init_from_dataset(
                #     self.dataset, cur_frame_idx_encoder, projection_matrix
                # )

                # input_data = self.encoder_input_init(viewpoint,viewpoint_encoder)
                gaussians_encoder, out_ex_encoder,T = encoder_gs(self.encoder, input_data,pose_ture = False , existing_gaussians = False)

                # w2c = torch.eye(4)

                # # w2c[:3, :3] = viewpoint.R
                # # w2c[:3, 3] = viewpoint.T





                # visible_ratio, visible_mask, projected_points = compute_visible_ratio(
                #     gaussians_means,
                #     extrinsic_matrix,
                #     projection_matrix,
                #     image_width,
                #     image_height
                # )






                # plot_3d_scene(gaussians_means, extrinsic_matrix, visible_mask)


                # visualize_projection(projected_points, visible_mask, image_width, image_height)

                
                render_pkg = self.tracking(cur_frame_idx, viewpoint,out_ex_encoder)

                # input_data = self.encoder_input(cur_frame_idx,self.current_window)
                # gaussians_encoder,_,_ = encoder_gs(self.encoder, input_data,pose_ture = True , existing_gaussians = None)
                # self.gaussians_track = gaussians_encoder

                current_window_dict = {}

                current_window_dict[self.current_window[0]] = self.current_window[1:]

                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]


                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                


                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]

                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval

                curr_visibility = (render_pkg["n_touched"] > 0).long()


                curr_frame = self.cameras[cur_frame_idx]
                # blur_check = is_image_blurry(curr_frame.original_image,100)
                blur_check = False

                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                    blur_check
                )



                if len(self.current_window) < self.window_size:


                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero() 
                    
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    
                    point_ratio = intersection / union
                    

                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                        and (not blur_check)
                    )


                if self.single_thread:
                    create_kf = check_time and create_kf

                if (cur_frame_idx - last_keyframe_idx) >= self.kf_interval and (not blur_check):
                    create_kf = True

                # create_kf = True
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )

                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map,gaussians_encoder
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()


                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
