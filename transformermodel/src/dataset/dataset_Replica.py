import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Tuple
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
class DatasetReplicaCfg(DatasetCfgCommon):
    name: Literal["replica"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = 1
    far: float = 100
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    sequence: Optional[str] = 'office0'


class DatasetReplica(IterableDataset):
    cfg: DatasetReplicaCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    near: float = 1.0
    far: float = 100.0
    data: List[Tuple[Path, np.ndarray]]
    

    fx: float
    fy: float
    cx: float
    cy: float
    replica_original_size: Tuple[int, int]


    def __init__(
        self,
        cfg: DatasetReplicaCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()

        cfg.root = Path('/home/ben/3dgsslam/splat-depth/SplaTam_v1/data/Replica')
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        


        if not self.cfg.roots:
            raise ValueError("DatasetReplica requires non-empty 'roots' in config")
        self.cfg.root = self.cfg.roots[0]


        self.cfg.sequence = self.cfg.sequence if hasattr(self.cfg, "sequence") else "room0"
        self.cfg.image_size = self.cfg.image_size if hasattr(self.cfg, "image_size") else (480, 640)
        self.cfg.max_time_diff = 0.02
        self.scene_idx_offset = 0
        self.scene_suffix = "_0"


        if self.cfg.near != -1:
            self.near = self.cfg.near
        if self.cfg.far != -1:
            self.far = self.cfg.far


        self.sequence_path = self.cfg.root / self.cfg.sequence
        self.results_path = self.sequence_path / "results"
        self.pose_file = self.sequence_path / "traj.txt"
        self.camera_info_file = self.sequence_path / "cam_params.json"


        self._validate_paths()


        self._load_camera_intrinsics()


        self.data = self.load_and_align_data()
        if not self.data:
            raise ValueError(f"No aligned data found for sequence: {self.cfg.sequence}")


    def _validate_paths(self) -> None:
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_path}")
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file (traj.txt) not found: {self.pose_file}")
        if not self.camera_info_file.exists():
            raise FileNotFoundError(f"Camera info file not found: {self.camera_info_file}")


    def _load_camera_intrinsics(self) -> None:
        with open(self.camera_info_file, "r") as f:
            camera_info = json.load(f)
        

        if "K" in camera_info:
            K = np.array(camera_info["K"], dtype=np.float32).reshape(3, 3)
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
        else:
            self.fx = camera_info["camera"]["fx"]
            self.fy = camera_info["camera"]["fx"]
            self.cx = camera_info["camera"]["fx"]
            self.cy = camera_info["camera"]["fx"]
        

        self.replica_original_width = camera_info.get("width", 640)
        self.replica_original_height = camera_info.get("height", 480)
        self.replica_original_size = (self.replica_original_height, self.replica_original_width)


    def load_and_align_data(self) -> List[Tuple[Path, np.ndarray]]:

        rgb_extensions = (".jpg", ".png")

        rgb_files = [
            f for f in self.results_path.iterdir() 
            if f.suffix.lower() in rgb_extensions and "jpg" in f.name.lower()
        ]
        
        if not rgb_files:
            raise FileNotFoundError(f"No RGB images found in results directory: {self.results_path}")
        

        def _get_image_index(file_path: Path) -> int:

            import re
            match = re.search(r'\d+', file_path.stem)
            return int(match.group()) if match else 0
        
        sorted_rgb_paths = sorted(rgb_files, key=_get_image_index)
        num_rgb = len(sorted_rgb_paths)
        print(f"Found {num_rgb} RGB images in results directory: {self.results_path}")


        pose_data = []
        with open(self.pose_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    print(f"Skipping invalid pose line: {line}")
                    continue

                pose_matrix = np.array(parts).reshape(4, 4)
                quat = pose_matrix[:3, :3]
                trans = pose_matrix[:3, 3]
                pose_data.append((trans, quat))
        num_poses = len(pose_data)
        print(f"Found {num_poses} valid poses in traj.txt")


        if num_rgb != num_poses:
            raise ValueError(
                f"RGB count ({num_rgb}) != Pose count ({num_poses}) "
                f"for sequence: {self.cfg.sequence}"
            )


        aligned_data = []
        for img_path, (trans, quat) in zip(sorted_rgb_paths, pose_data):


            T_w2c = torch.eye(4).numpy()
            T_w2c[:3,:3] =  quat
            
            T_w2c[:3, 3] = trans


            T = np.linalg.inv(T_w2c)


            # T[:3, 1] *= -1

            aligned_data.append((img_path, T.astype(np.float32)))

        print(f"Aligned {len(aligned_data)} images-poses for sequence: {self.cfg.sequence}")
        return aligned_data


    def c2w_to_w2c(self, extrinsics: Tensor) -> Tensor:
        assert extrinsics.shape[-2:] == (4, 4), "Extrinsics must be 4x4"
        B = extrinsics.shape[0]
        w2c = torch.zeros_like(extrinsics)

        R_w2c = extrinsics[:, :3, :3].transpose(1, 2)
        w2c[:, :3, :3] = R_w2c

        t_w2c = -torch.bmm(R_w2c, extrinsics[:, :3, 3:4])
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


    def load_images(self, indices: list[int], image_paths: list[Path]) -> Float[Tensor, "batch 3 h w"]:
        images = []
        for idx in indices:
            img_path = image_paths[idx]
            img = Image.open(img_path).convert("RGB")

            img = img.resize((256,256), Image.BILINEAR)
            images.append(self.to_tensor(img))
        return torch.stack(images)


    def load_images_target(self, indices: list[int], image_paths: list[Path]) -> Float[Tensor, "batch 3 h w"]:
        images = []
        for idx in indices:
            img_path = image_paths[idx]
            img = Image.open(img_path).convert("RGB")

            images.append(self.to_tensor(img))
        return torch.stack(images)


    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, "view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)


    def __iter__(self):

        image_paths = [item[0] for item in self.data]
        poses = [item[1] for item in self.data]
        scene = self.cfg.sequence + self.scene_suffix


        extrinsics = torch.tensor(np.array(poses), dtype=torch.float32)


        original_w = self.replica_original_width
        original_h = self.replica_original_height
        target_h, target_w = self.cfg.image_size
        scale_x = target_w / original_w
        scale_y = target_h / original_h

        scaled_fx = self.fx * scale_x
        scaled_fy = self.fy * scale_y
        scaled_cx = self.cx * scale_x
        scaled_cy = self.cy * scale_y


        norm_intrinsics = torch.tensor([
            [600 / 1200, 0.0, 599.5 / 1200],
            [0.0, 600 / 680, 339.5 / 680],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)

        intrinsics_256 = norm_intrinsics.repeat(len(image_paths), 1, 1)
        intrinsics = norm_intrinsics.repeat(len(image_paths), 1, 1)


        runs = range(
            self.cfg.test_times_per_scene 
            if self.stage == "test" 
            else 1
        )


        for run_idx in runs:
            max_idx = extrinsics.size(0) - 10
            for scene_idx in range(self.scene_idx_offset, max_idx, 1):
                n = 5
                finally_n = scene_idx + n if (scene_idx + n) < extrinsics.size(0) else max_idx

                context_indices = torch.tensor([scene_idx, finally_n])

                extended_start = max(0, scene_idx - 10)
                extended_end = min(extrinsics.size(0) - 1, finally_n + 10)
                target_indices = torch.arange(extended_start, extended_end + 1, 3)


                context_images = self.load_images(context_indices.tolist(), image_paths)
                target_images = self.load_images_target(target_indices.tolist(), image_paths)


                scale = 1.0
                # if context_indices.shape[0] == 2 and self.cfg.make_baseline_1:
                #     a, b = extrinsics[context_indices][:, :3, 3]
                #     scale = (a - b).norm()
                #     if scale < self.cfg.baseline_epsilon:
                #         print(f"Skipped {scene} (small baseline: {scale:.6f})")
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


                # yield apply_crop_shim(example, tuple(self.cfg.image_size))
                yield example


            self.scene_idx_offset = (self.scene_idx_offset + 1) % n
            self.scene_suffix = f"_{self.scene_idx_offset}"


    def __len__(self) -> int:
        return len(self.data)