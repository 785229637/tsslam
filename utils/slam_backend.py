import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.utils.sh_utils import RGB2SH

from gaussian_splatting.utils.general_utils import inverse_sigmoid, build_rotation

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
# from src.misc.sh_rotation import rotate_sh
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
def rotate_sh(sh: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    sh_rotated = sh.clone()
    n_points, d_sh = sh.shape[0], sh.shape[1]
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]



    if d_sh > 1:
        sh1 = sh[:, 1:4, :]  # [N, 3, 3]
        sh_rotated[:, 1, :] = r20 * sh1[:, 0, :] + r21 * sh1[:, 1, :] + r22 * sh1[:, 2, :]
        sh_rotated[:, 2, :] = r00 * sh1[:, 0, :] + r01 * sh1[:, 1, :] + r02 * sh1[:, 2, :]
        sh_rotated[:, 3, :] = r10 * sh1[:, 0, :] + r11 * sh1[:, 1, :] + r12 * sh1[:, 2, :]


    if d_sh > 4:
        sh2 = sh[:, 4:9, :]  # [N, 5, 3]
        sh_rotated[:, 4, :] = (r20**2)*sh2[:,0] + (r21**2)*sh2[:,1] + (r22**2)*sh2[:,2] + 2*r20*r21*sh2[:,3] + 2*r20*r22*sh2[:,4]
        sh_rotated[:, 5, :] = r00*r20*sh2[:,0] + r01*r21*sh2[:,1] + r02*r22*sh2[:,2] + (r01*r20 + r00*r21)*sh2[:,3] + (r02*r20 + r00*r22)*sh2[:,4]
        sh_rotated[:, 6, :] = (r00**2)*sh2[:,0] + (r01**2)*sh2[:,1] + (r02**2)*sh2[:,2] + 2*r00*r01*sh2[:,3] + 2*r00*r02*sh2[:,4]
        sh_rotated[:, 7, :] = r10*r20*sh2[:,0] + r11*r21*sh2[:,1] + r12*r22*sh2[:,2] + (r11*r20 + r10*r21)*sh2[:,3] + (r12*r20 + r10*r22)*sh2[:,4]
        sh_rotated[:, 8, :] = r00*r10*sh2[:,0] + r01*r11*sh2[:,1] + r02*r12*sh2[:,2] + (r01*r10 + r00*r11)*sh2[:,3] + (r02*r10 + r00*r12)*sh2[:,4]


    if d_sh > 9:
        sh3 = sh[:, 9:16, :]  # [N, 7, 3]

        sh_rotated[:, 9, :] = r20**3 * sh3[:,0] + r21**3 * sh3[:,1] + r22**3 * sh3[:,2] + 3*r20**2*r21*sh3[:,3] + 3*r20**2*r22*sh3[:,4] + 3*r20*r21**2*sh3[:,5] + 3*r20*r22**2*sh3[:,6]
        sh_rotated[:, 10, :] = r00*r20**2 * sh3[:,0] + r01*r21**2 * sh3[:,1] + r02*r22**2 * sh3[:,2] + r01*r20**2*sh3[:,3] + r00*r20*r21*sh3[:,3] + r02*r20**2*sh3[:,4] + r00*r20*r22*sh3[:,4] + r01*r20*r21*sh3[:,5] + r00*r21**2*sh3[:,5] + r02*r20*r22*sh3[:,6] + r00*r22**2*sh3[:,6]

        sh_rotated[:, 15, :] = r00*r10*r20 * sh3[:,0] + r01*r11*r21 * sh3[:,1] + r02*r12*r22 * sh3[:,2] + (r01*r10*r20 + r00*r11*r20 + r00*r10*r21)*sh3[:,3] + (r02*r10*r20 + r00*r12*r20 + r00*r10*r22)*sh3[:,4]


    if d_sh > 16:
        sh4 = sh[:, 16:25, :]  # [N, 9, 3]

        sh_rotated[:, 16, :] = r20**4 * sh4[:,0] + r21**4 * sh4[:,1] + r22**4 * sh4[:,2] + 4*r20**3*r21*sh4[:,3] + 4*r20**3*r22*sh4[:,4] + 6*r20**2*r21**2*sh4[:,5] + 6*r20**2*r22**2*sh4[:,6] + 4*r20*r21**3*sh4[:,7] + 4*r20*r22**3*sh4[:,8]

        sh_rotated[:, 24, :] = r00*r10*r20**2 * sh4[:,0] + r01*r11*r21**2 * sh4[:,1] + r02*r12*r22**2 * sh4[:,2]

    return sh_rotated

class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None


        self.max_scaling_threshold = self.config["Training"].get("max_scaling_threshold", 0.5)
        self.min_opacity_threshold = self.config["Training"].get("min_opacity_threshold", 0.1)

    def prune_large_or_transparent_gaussians(self, volume_threshold=0.1, opacity_threshold=0.01):
        if self.gaussians.get_xyz.shape[0] == 0:
            return



        scalings = self.gaussians.get_scaling
        

        if scalings.shape[1] == 1:

            volumes = scalings ** 3
        else:

            volumes = scalings.prod(dim=1, keepdim=True)
        

        opacities = self.gaussians.get_opacity



        too_large = (volumes > volume_threshold).squeeze()
        too_transparent = (opacities < opacity_threshold).squeeze()
        to_prune = too_large | too_transparent


        num_pruned = to_prune.sum().item()
        if num_pruned > 0:
            self.gaussians.prune_points(to_prune)
            print(f"Pruned {num_pruned} gaussians (large: {too_large.sum()}, transparent: {too_transparent.sum()})")



    def add_encoder_gaussians(self, gaussians_encoder, viewpoint, kf_id=-1):
        
        # gaussians_encoder.means = gaussians_encoder.means
        # --------------------------

        # --------------------------


        xyz = gaussians_encoder.means.squeeze(0)
        

        # # frez = torch.diag(torch.tensor([1.0, 1.0, 1.0], device='cuda:0', dtype=torch.float64)) # [3, 3]
        # R = viewpoint.R 
        # T = viewpoint.T  # [3]
        # T = -T
        
        

        # xyz_cam = xyz_cam.float().to(self.device)
        # R = R.float().to(self.device)
        # T = T.float().to(self.device)
        



        # xyz_world = torch.matmul(xyz_cam+ T, R) 


        # --------------------------

        # --------------------------
        opacities = gaussians_encoder.opacities.squeeze(0).unsqueeze(1).to(self.device)
        opacities = inverse_sigmoid(opacities).to(torch.float32)

        # --------------------------

        # --------------------------
        sh = gaussians_encoder.harmonics.squeeze(0).to(self.device)  # [N, 3, 25]
        sh = sh.to(torch.float32)
        # sh = rotate_sh(sh.permute(0, 2, 1), R).permute(0, 2, 1)  





        
        # target_max_sh_degree = 3
        # target_dim = (target_max_sh_degree + 1) **2
        # if sh.shape[-1] > target_dim:
        #     sh = sh[:, :, :target_dim]  # [N, 3, 16]
        
        features_dc = sh[:, :, 0:1].transpose(1, 2).contiguous()  # [N, 1, 3]
        features_rest = sh[:, :, 1:].transpose(1, 2).contiguous()   # [N, 15, 3]

        # --------------------------

        # --------------------------
        cov = gaussians_encoder.covariances.squeeze(0).to(self.device)  # [N, 3, 3]
        # cov = R.T@ cov @ R
        cov = cov.to(torch.float32)
        

        L = torch.linalg.cholesky(cov)  # [N, 3, 3]
        

        scaling = torch.diagonal(L, dim1=1, dim2=2).abs()  # [N, 3]


        scaling = scaling * 1.0

        

        S_inv = torch.diag_embed(1.0 / scaling.clamp_min(1e-8))  # [N, 3, 3]
        R_mat = L @ S_inv

        # --------------------------

        # --------------------------
        def rotmat_to_quat(R):

            r00 = R[:, 0, 0]
            r01 = R[:, 0, 1]
            r02 = R[:, 0, 2]
            r10 = R[:, 1, 0]
            r11 = R[:, 1, 1]
            r12 = R[:, 1, 2]
            r20 = R[:, 2, 0]
            r21 = R[:, 2, 1]
            r22 = R[:, 2, 2]


            tr = r00 + r11 + r22
            qw = 0.5 * torch.sqrt(torch.clamp(tr + 1.0, min=1e-8))
            qx = 0.5 * torch.sign(r21 - r12) * torch.sqrt(torch.clamp(r00 - r11 - r22 + 1.0, min=1e-8))
            qy = 0.5 * torch.sign(r02 - r20) * torch.sqrt(torch.clamp(r11 - r00 - r22 + 1.0, min=1e-8))
            qz = 0.5 * torch.sign(r10 - r01) * torch.sqrt(torch.clamp(r22 - r00 - r11 + 1.0, min=1e-8))


            quat = torch.stack([qw, qx, qy, qz], dim=1)
            return torch.nn.functional.normalize(quat, dim=1)
        

        rotation = rotmat_to_quat(R_mat)  # [N, 4]


        scales = torch.log(scaling)  # [N, 3]


        # --------------------------

        # --------------------------
        new_xyz = torch.nn.Parameter(xyz.requires_grad_(True).to(torch.float32))
        new_features_dc = torch.nn.Parameter(features_dc.requires_grad_(True).to(torch.float32))
        new_features_rest = torch.nn.Parameter(features_rest.requires_grad_(True).to(torch.float32))
        new_scaling = torch.nn.Parameter(scales.requires_grad_(True).to(torch.float32))
        new_rotation = torch.nn.Parameter(rotation.requires_grad_(True).to(torch.float32))
        new_opacity = torch.nn.Parameter(opacities.requires_grad_(True).to(torch.float32))
        
        # new_unique_kfIDs = torch.full((xyz.shape[0],), kf_id, dtype=torch.int, device=self.device)
        # new_n_obs = torch.ones(xyz.shape[0], dtype=torch.int, device=self.device)

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id

        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        

        self.gaussians.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs.cpu(),
            new_n_obs=new_n_obs.cpu()
        )
        
        print(f"Added {xyz.shape[0]} encoder gaussians for keyframe {kf_id}")


        


    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )


        # self.max_scaling_threshold = self.config["Training"].get("max_scaling_threshold", 1.0)
        # self.min_opacity_threshold = self.config["Training"].get("min_opacity_threshold", 0.01)


    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):


        if len(current_window) == 0:
            return


        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]


        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)


        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []



            num_keyframes = len(current_window)


            weights = torch.linspace(1.0, 0.1, num_keyframes)  # [0.1, 0.2, ..., 1.0]


            # import math
            # weights = torch.tensor([math.exp(-0.5*(num_keyframes-1 - i)) for i in range(num_keyframes)])




            for cam_idx in range(num_keyframes):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )

                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )


                current_weight = weights[cam_idx]
                loss_mapping += current_weight * get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)


            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)


            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False



            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 1
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            # mask = mask = (self.gaussians.unique_kfIDs >= sorted_window[2]) & (self.gaussians.unique_kfIDs < sorted_window[0])
                            mask = self.gaussians.unique_kfIDs >= sorted_window[5]
                            
                            if not self.initialized:

                                mask = self.gaussians.unique_kfIDs >= 0

                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():

                if self.pause:
                    time.sleep(0.01)
                    continue 


                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                



                self.map(self.current_window)

                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)

                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    gaussians_encoder = data[4]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint

                    #     cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    # )
                    self.add_encoder_gaussians(gaussians_encoder, viewpoint, cur_frame_idx)
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":

                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    gaussians_encoder = data[5]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    

                    # self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    # self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

                    self.add_encoder_gaussians(gaussians_encoder, viewpoint, cur_frame_idx)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]

                    iter_per_kf = self.mapping_itr_num if self.single_thread else 500

                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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

                    self.keyframe_optimizers = torch.optim.Adam(opt_params)


                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    
                    self.push_to_frontend("keyframe")
                    
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
