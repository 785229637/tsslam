import os
import queue
import torch
import numpy as np
from dataclasses import dataclass
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gui.gl_render.util_gau import GaussianData
from gui.gui_utils import ParamsGUI, GaussianPacket
from gui.slam_gui import SLAM_GUI
from utils.camera_utils import Camera
import yaml
from pathlib import Path
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering



@dataclass
class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.model_params = config["model_params"]
        self.dataset_params = config["Dataset"]
        self.training_params = config["Training"]


def init_config(config_path):
    return Config(config_path)


def load_gaussian_model(config, ply_path):

    sh_degree = 3 if config.training_params["spherical_harmonics"] else 0
    gaussian_model = GaussianModel(sh_degree=sh_degree, config=config.__dict__)
    

    gaussian_model.load_ply(ply_path)
    print(f"成功加载高斯点云：{ply_path}，包含{len(gaussian_model.get_xyz)}个点")
    return gaussian_model


def convert_to_gaussian_data(gaussian_model):

    xyz = gaussian_model.get_xyz.detach().cpu().numpy()
    rot = gaussian_model.get_rotation.detach().cpu().numpy()
    scale = gaussian_model.get_scaling.detach().cpu().numpy()
    opacity = gaussian_model.get_opacity.detach().cpu().numpy()
    sh = gaussian_model.get_features.detach().cpu().numpy()[:, 0, :]
    
    return GaussianData(
        xyz=xyz,
        rot=rot,
        scale=scale,
        opacity=opacity,
        sh=sh
    )


def main(config_path, ply_path):

    config = init_config(config_path)
    

    gaussian_model = load_gaussian_model(config, ply_path)
    

    gaussian_data = convert_to_gaussian_data(gaussian_model)
    

    q_main2vis = queue.Queue()
    q_vis2main = queue.Queue()
    
    params_gui = ParamsGUI(
        pipe=config.training_params,
        background=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        gaussians=gaussian_model,
        q_main2vis=q_main2vis,
        q_vis2main=q_vis2main
    )
    

    print("启动可视化GUI...")
    gui_app = SLAM_GUI(params_gui)
    

    packet = GaussianPacket(gaussians=gaussian_model)
    q_main2vis.put(packet)
    

    gui_app.window.show()
    gui.Application.instance.run()

if __name__ == "__main__":

    CONFIG_PATH = "/home/ben/G-3DGS/wb-2/results/TUM_RGBD_rgbd_dataset_freiburg1_desk/2025-08-19-21-06-49/config.yml"

    PLY_PATH = "/home/ben/G-3DGS/wb-2/results/TUM_RGBD_rgbd_dataset_freiburg1_desk/2025-08-19-21-06-49/point_cloud/final/point_cloud.ply"
    

    if not Path(CONFIG_PATH).exists():
        raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")
    if not Path(PLY_PATH).exists():
        raise FileNotFoundError(f"PLY文件不存在：{PLY_PATH}")
    
    main(CONFIG_PATH, PLY_PATH)
