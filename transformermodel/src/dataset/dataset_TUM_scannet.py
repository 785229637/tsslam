import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Tuple
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

# from ..geometry.projection import get_fov
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
        

        cfg.root = Path('/home/ben/data/ScanNet')
        cfg.image_size = (968, 1296)
        cfg.sequence = "scene0169_00"
        self.scene_idx_offset = 0
        self.scene_suffix = ""
        

        self.near = 1.0
        self.far = 100.0
        

        self.sequence_path = cfg.root / cfg.sequence
        self.color_path = self.sequence_path / "color"
        self.depth_path = self.sequence_path / "depth"
        self.pose_path = self.sequence_path / "pose"
        
        if not self.color_path.exists():
            raise FileNotFoundError(f"Color directory not found: {self.color_path}")
        if not self.pose_path.exists():
            raise FileNotFoundError(f"Pose directory not found: {self.pose_path}")
        

        self.data = self.load_data()
        
        if not self.data:
            raise ValueError(f"No data found for sequence: {cfg.sequence}")

    def load_data(self) -> List[Tuple[Path, np.ndarray]]:

        color_files = sorted(self.color_path.glob("*.jpg"), key=lambda x: int(x.stem))

        pose_files = sorted(self.pose_path.glob("*.txt"), key=lambda x: int(x.stem))
        

        if len(color_files) != len(pose_files):
            raise ValueError(f"Mismatch: {len(color_files)} color images vs {len(pose_files)} poses")
        

        data = []
        skipped_count = 0
        for color_file, pose_file in zip(color_files, pose_files):

            pose = np.loadtxt(pose_file).astype(np.float32)
            if pose.shape != (4, 4):
                print(f"Warning: Invalid pose shape in {pose_file}: {pose.shape}, skipping...")
                skipped_count += 1
                continue
            

            if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
                print(f"Warning: Pose contains nan/inf in {pose_file}, skipping...")
                skipped_count += 1
                continue
            

            R = pose[:3, :3]

            if not self.is_valid_rotation_matrix(R):
                print(f"Warning: Invalid rotation matrix in {pose_file}, attempting to fix...")

                pose[:3, :3] = self.fix_rotation_matrix(R)
                if not self.is_valid_rotation_matrix(pose[:3, :3]):
                    print(f"Warning: Cannot fix rotation matrix in {pose_file}, skipping...")
                    skipped_count += 1
                    continue
            
            data.append((color_file, pose))
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} frames with invalid poses")
        print(f"Loaded {len(data)} valid frames for sequence: {self.cfg.sequence}")
        return data
    
    def is_valid_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-2) -> bool:

        if R.shape != (3, 3):
            return False
        

        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            return False
        

        should_be_identity = R.T @ R
        identity = np.eye(3, dtype=np.float32)
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False
        

        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=tolerance):
            return False
        
        return True
    
    def fix_rotation_matrix(self, R: np.ndarray) -> np.ndarray:
        try:

            U, _, Vt = np.linalg.svd(R)
            R_fixed = U @ Vt
            

            if np.linalg.det(R_fixed) < 0:
                U[:, -1] *= -1
                R_fixed = U @ Vt
            
            return R_fixed.astype(np.float32)
        except:
            return R

    def c2w_to_w2c(self, extrinsics):
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

    def is_image_blurry(self, image_tensor, threshold):
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
        


        norm_intrinsics = torch.tensor([
            [1169.621094/1296, 0, 646.295044/1296],
            [0, 1167.105103/968, 489.927032/968],
            [0, 0, 1]
        ], dtype=torch.float32)

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
            for scene_idx in range(0, extrinsics.size()[0]-10, 5):
                ###########################################

                ###########################################
                try:
                    start_image = self.load_images([scene_idx], image_paths)[0]
                    if self.is_image_blurry(start_image, blur_threshold):
                        print(f"[Blur Check] Skipping scene_idx {scene_idx}: start image is blurry")
                        continue
                except Exception as e:
                    print(f"[Blur Check] Error checking start image (scene_idx {scene_idx}): {str(e)}")
                    continue
                
                ###########################################

                ###########################################
                # n = random.randint(10, 25)
                n=10
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
                            print(f"[Blur Check] Skipping scene_idx {scene_idx}: no clear end image")
                            break
                    except Exception as e:
                        print(f"[Blur Check] Error checking end image: {str(e)}")
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
                
                extended_start = max(0, original_start-5)
                extended_end = min(extrinsics.size()[0] - 1, original_end+5 )
                

                FIXED_NUM = 10


                target_indices = torch.linspace(
                    start=extended_start,
                    end=extended_end,
                    steps=FIXED_NUM,
                    dtype=torch.int64
                )
                                
    
                
                ###########################################

                ###########################################
                context_images = self.load_images(context_indices.tolist(), image_paths)
                target_images = self.load_images_target(target_indices.tolist(), image_paths)

                scale = 1
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

                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                    
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))
            
            self.scene_idx_offset += 1
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
