import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import cv2
import torch.nn.functional as F
import random
import trimesh
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

@dataclass
class DatasetTUMRGBDCfg(DatasetCfgCommon):
    name: Literal["tum_rgbd"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    roots: list[Path]

    


class DatasetTUMRGBD(IterableDataset):
    cfg: DatasetTUMRGBDCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    near: float = 1
    far: float = 1000.0
    data: List[Tuple[Path, np.ndarray]]

    def __init__(
        self,
        cfg: DatasetTUMRGBDCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # cfg.name= "re10k"
        cfg.root = Path('/home/ben/data/ETH3D')
        cfg.image_size  = (458, 739)
        cfg.sequence ="mannequin_1"
        cfg.max_time_diff = 0.02
        self.scene_idx_offset = 0
        self.scene_suffix = ""

        

        

        self.near = 1.0
        self.far = 100.0
        

        # if cfg.near != -1:
        #     self.near = cfg.near
        # if cfg.far != -1:

            

        self.sequence_path = cfg.root / cfg.sequence
        self.rgb_path = self.sequence_path / "rgb"
        self.depth_path = self.sequence_path / "depth"
        self.pose_file = self.sequence_path / "groundtruth.txt"
        self.rgb_timestamps_file = self.sequence_path / "rgb.txt"
        self.depth_timestamps_file = self.sequence_path / "depth.txt"
        

        if not self.rgb_path.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_path}")
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")
        if not self.rgb_timestamps_file.exists():
            raise FileNotFoundError(f"RGB timestamps file not found: {self.rgb_timestamps_file}")
        

        self.data = self.load_and_align_data()
        
        if not self.data:
            raise ValueError(f"No aligned data found for sequence: {cfg.sequence}")

    def load_and_align_data(self) -> List[Tuple[Path, np.ndarray]]:

        rgb_timestamps, rgb_ts_to_filename = self.load_timestamps(self.rgb_timestamps_file)
        

        pose_data = []
        with open(self.pose_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                timestamp = float(parts[0])
                pose_data.append((timestamp, parts[1:]))
        
        pose_data.sort(key=lambda x: x[0])
        pose_timestamps = [p[0] for p in pose_data]
        

        aligned_data = []
        for rgb_ts in rgb_timestamps:

            pose_idx = np.argmin(np.abs(np.array(pose_timestamps) - rgb_ts))
            pose_ts = pose_timestamps[pose_idx]
            pose_diff = abs(pose_ts - rgb_ts)
            

            if pose_diff < self.cfg.max_time_diff:


                filename = rgb_ts_to_filename[rgb_ts]

                if Path(filename).is_absolute():
                    img_path = Path(filename)
                else:
                    img_path = self.sequence_path / filename
                

                if not img_path.exists():
                    print(f"Warning: Image file not found - {img_path}, skipping")
                    continue
                

                pose_parts = pose_data[pose_idx][1]
                trans = [float(x) for x in pose_parts[0:3]]
                quat = [float(x) for x in pose_parts[3:7]]
                

                T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
                T[:3, 3] = trans
                
                aligned_data.append((img_path, T))
        
        print(f"Aligned {len(aligned_data)} RGB images with poses for sequence: {self.cfg.sequence}")
        return aligned_data
    
    def load_timestamps(self, file_path: Path) -> Tuple[List[float], Dict[float, str]]:
        timestamps = []
        ts_to_filename = {}
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    ts = float(parts[0])
                    filename = parts[1]
                    timestamps.append(ts)
                    ts_to_filename[ts] = filename
        return timestamps, ts_to_filename
    
    def c2w_to_w2c(self,extrinsics):

        assert extrinsics.shape[-2:] == (4, 4), "外参矩阵必须是4x4的"
        

        B = extrinsics.shape[0]
        

        w2c = torch.zeros_like(extrinsics)
        

        R_c2w = extrinsics[:, :3, :3]
        

        R_w2c = R_c2w.transpose(1, 2)
        w2c[:, :3, :3] = R_w2c
        

        t_c2w = extrinsics[:, :3, 3:4]
        

        t_w2c = -torch.bmm(R_w2c, t_c2w)
        w2c[:, :3, 3:4] = t_w2c
        

        w2c[:, 3, 3] = 1.0
        
        return w2c

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,   2*x*z + 2*y*w],
            [2*x*y + 2*z*w,   1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)
    
    def is_image_blurry(self,image_tensor, threshold):


        image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
        image_np = image_np.astype(np.uint8)
        

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        

        variance = np.var(laplacian)
        
        return variance < threshold

    def __iter__(self):

        image_paths = [item[0] for item in self.data]
        poses = [item[1] for item in self.data]
        
        scene = self.cfg.sequence + self.scene_suffix
        

        extrinsics = torch.tensor(np.array(poses), dtype=torch.float32)
        # extrinsics = self.c2w_to_w2c(extrinsics)

        # intrinsics = torch.tensor([
        #     [525.0, 0, 319.5],
        #     [0, 525.0, 239.5], 
        #     [0, 0, 1]
        # ], dtype=torch.float32)


        # scale_x = self.cfg.image_shape[1] / self.cfg.image_size[1]
        # scale_y = self.cfg.image_shape[0] / self.cfg.image_size[0]
        # intrinsics[0] *= scale_x
        # intrinsics[1] *= scale_y




        # norm_intrinsics = torch.tensor([
        #     [intrinsics[0,0]/width, 0, intrinsics[0,2]/width],
        #     [0, intrinsics[1,1]/height, intrinsics[1,2]/height],
        #     [0, 0, 1]
        # ], dtype=torch.float32)
        # intrinsics = norm_intrinsics.repeat(len(image_paths), 1, 1)

        # norm_intrinsics = torch.tensor([


        #     [0, 0, 1]
        # ], dtype=torch.float32)(520.908620, 521.007327, 325.141442, 249.701764, 640, 480)

        norm_intrinsics = torch.tensor([
            [726.28741455078/739, 0, 354.6496887207/739],
            [0, 726.28741455078/458, 186.46566772461/458],
            [0, 0, 1]
        ], dtype=torch.float32)

        # new_intrinsics = torch.tensor([
        #     [210.0/256, 0, 127.5/256],
        #     [0, 280.0/256, 127.5/256],
        #     [0, 0, 1]
        # ], dtype=torch.float32)

        intrinsics_256 = norm_intrinsics.repeat(len(image_paths), 1, 1)
        intrinsics = norm_intrinsics.repeat(len(image_paths), 1, 1)





        runs = range(
            self.cfg.test_times_per_scene 
            if self.stage == "test" 
            else 1
        )

        

        blur_threshold = 10

        max_extend_steps = 20

        for run_idx in runs:
            for scene_idx in range(0,extrinsics.size()[0]-10,1):
                ###########################################

                ###########################################
                try:

                    start_image = self.load_images([scene_idx], image_paths)[0]  # [3, H, W]

                    if self.is_image_blurry(start_image, blur_threshold):
                        print(f"[Blur Check] Skipping scene_idx {scene_idx}: start image is blurry")
                        continue
                except Exception as e:
                    print(f"[Blur Check] Error checking start image (scene_idx {scene_idx}): {str(e)}")
                    continue
                
                ###########################################

                ###########################################
                # n = torch.randint(low=3, high=10, size=())
                n = 5

                if scene_idx + n >= extrinsics.size()[0]:
                    finally_n = extrinsics.size()[0] - 1
                else:
                    finally_n = scene_idx + n
                

                current_end = finally_n
                extend_count = 0
                valid_end = False
                
                while current_end < extrinsics.size()[0] - 1:
                    try:

                        end_image = self.load_images([current_end], image_paths)[0]

                        if not self.is_image_blurry(end_image, blur_threshold):
                            valid_end = True
                            break

                        current_end += 1
                        extend_count += 1
                        

                        if extend_count >= max_extend_steps:
                            print(f"[Blur Check] Skipping scene_idx {scene_idx}: no clear end image found after {max_extend_steps} extensions")
                            break
                    except Exception as e:
                        print(f"[Blur Check] Error checking end image (scene_idx {scene_idx}, current_end {current_end}): {str(e)}")
                        current_end += 1
                        extend_count += 1
                        if extend_count >= max_extend_steps:
                            break
                

                if not valid_end:
                    continue
                

                finally_n = current_end
                context_indices = torch.tensor([scene_idx, finally_n])
                
                ###########################################

                ###########################################

                original_start = scene_idx
                original_end = finally_n
                

                extended_start = max(0, original_start-3)
                extended_end = min(extrinsics.size()[0] - 1, original_end+3 )
                

                target_indices = torch.arange(extended_start, extended_end + 1,1)
                

                clear_target_indices = []
                for idx in target_indices:
                    try:

                        target_img = self.load_images_target([idx], image_paths)[0]

                        if not self.is_image_blurry(target_img, blur_threshold):
                            clear_target_indices.append(idx)
                    except Exception as e:
                        print(f"[Blur Check] Error checking target image (idx {idx}): {str(e)}")
                        continue
                

                if not clear_target_indices:
                    print(f"[Blur Check] Skipping scene_idx {scene_idx}: no clear target images")
                    continue
                

                target_indices = torch.asarray(clear_target_indices)
                
                ###########################################

                ###########################################

                context_images = self.load_images(context_indices.tolist(), image_paths)
                # context_images_depth = self.load_images(context_indices.tolist(), depth_paths)/5000
                # context_images_depth = 1
                

                target_images = self.load_images_target(target_indices.tolist(), image_paths)
                # target_images_depth = self.load_images_target(target_indices.tolist(), depth_paths)/5000
                # target_images_depth = 1
                

                # if self.cfg.skip_bad_shape:
                #     expected_shape = (3, *self.cfg.image_size)
                #     if context_images.shape[1:] != expected_shape:
                #         print(f"Skipped bad context image shape: {context_images.shape}")
                #         continue
                #     if target_images.shape[1:] != expected_shape:
                #         print(f"Skipped bad target image shape: {target_images.shape}")
                #         continue


                scale = 1
                # context_extrinsics = extrinsics[context_indices]
                # if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                #     a, b = context_extrinsics[:, :3, 3]
                #     scale = (a - b).norm()
                #     if scale < self.cfg.baseline_epsilon:
                #         print(f"Skipped {scene} due to small baseline: {scale:.6f}")
                #         continue
                #     extrinsics[:, :3, 3] /= scale


                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics_256[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }


                # if self.stage == "train" and self.cfg.augment:
                #     example = apply_augmentation_shim(example)
                
                    
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))
            self.scene_idx_offset=self.scene_idx_offset+1

            self.scene_suffix = f"_{self.scene_idx_offset}"

    def load_images(self, indices: list[int], image_paths: list[Path]) -> Float[Tensor, "batch 3 h w"]:
        images = []
        for idx in indices:
            img_path = image_paths[idx]
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize(self.cfg.image_shape[::-1], Image.BILINEAR)
            images.append(self.to_tensor(img))
        return torch.stack(images)
    
    def load_images_target(self, indices: list[int], image_paths: list[Path]) -> Float[Tensor, "batch 3 h w"]:
        images = []
        for idx in indices:
            img_path = image_paths[idx]
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize(self.cfg.image_shape[::-1], Image.BILINEAR)
            images.append(self.to_tensor(img))
        return torch.stack(images)


    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.data)
    

