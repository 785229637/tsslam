from dataclasses import dataclass
from pathlib import Path
import random
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 
from transplat.src.dataset.data_module import get_data_shim
from transplat.src.dataset.types import BatchedExample
from transplat.src.dataset import DatasetCfg
from transplat.src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from transplat.src.global_cfg import get_cfg
from transplat.src.loss import Loss
from transplat.src.misc.benchmarker import Benchmarker
from transplat.src.misc.image_io import prep_image, save_image, save_video
from transplat.src.misc.LocalLogger import LOG_PATH, LocalLogger
from transplat.src.misc.step_tracker import StepTracker
from transplat.src.model.types import Gaussians
from transplat.src.visualization.annotation import add_label
from transplat.src.visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from transplat.src.visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)  
from transplat.src.visualization.color_map import apply_color_map_to_image
from transplat.src.visualization.layout import add_border, hcat, vcat
from transplat.src.visualization import layout
from transplat.src.visualization.validation_in_3d import render_cameras, render_projections
from transplat.src.model.decoder.decoder import Decoder, DepthRenderingMode
from encoder_wb import Encoder
from encoder_wb.visualization.encoder_visualizer import EncoderVisualizer
import torch.nn.functional as F


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0
        self.existing_gaussians = None
        
        self.existing_scene = None
        self.num_save = 0
        self.pose_save = 0
        self.pose_all = []
        self.pose_gt = []
        self.image_all = []
        self.near = 0
        self.far = 0
        self.intrinsics = None
        

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx ):
        batch: BatchedExample = self.data_shim(batch)
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key].to('cuda')
            
        _, _, _, h, w = batch["target"]["image"].shape
        
        # Run the model.

        #     if self.existing_scene != batch["scene"]:
        #         self.pose_all = []
        #         self.existing_gaussians = None
        #         T0 = batch["context"]["extrinsics"][0, 0]
        #         self.existing_scene = batch["scene"]
        #     else:

        #         batch["context"]["extrinsics"][0][0] = self.pose_all[-1]

            # existing = self.existing_gaussians.detach() if self.existing_gaussians else None
        gaussians, out_ex,delta_T_encoder = self.encoder(
            batch["context"], 
            self.global_step, 
            existing_gaussians=None,
            deterministic=False,
            scene_names=batch["scene"],
            pose_ture = False

        )
        # if self.existing_gaussians != None:
            # self.existing_gaussians = merge_gaussians(self.existing_gaussians,gaussians) 
        # self.existing_gaussians = merge_gaussians(self.existing_gaussians,gaussians) 
        # else:
            # self.existing_gaussians = gaussians


        # rotation_pred = rotation_pred.detach().requires_grad_(True)
        # # translation_pred = translation_pred.detach().requires_grad_(True)
        # poseloss = self.losses["pose"](batch, out_ex, delta_T_encoder)
        # updated_extrinsics = out_ex
        # self.pose_all.append(updated_extrinsics[0][1])
        # # updated_extrinsics = updated_extrinsics[:, 1:2, :, :]
        # # batch["target"]["extrinsics"] = torch.cat((batch["target"]["extrinsics"], updated_extrinsics), dim=1)
        # # batch["target"]["intrinsics"] = torch.cat((batch["target"]["intrinsics"], batch["context"]["intrinsics"][:, 1:2:, :, :]), dim=1) 
        # # batch["target"]["near"] = torch.cat((batch["target"]["near"], batch["context"]["near"][:, 1:2]), dim=1) 
        # # batch["target"]["far"] = torch.cat((batch["target"]["far"], batch["context"]["far"][:, 1:2]), dim=1) 

        # # output = self.decoder.forward(
        # #     gaussians,
        # #     batch["target"]["extrinsics"],
        # #     batch["target"]["intrinsics"],
        # #     batch["target"]["near"],
        # #     batch["target"]["far"],
        # #     (h, w),
        # #     depth_mode=self.train_cfg.depth_mode,
        # # )
        # # batch["target"]["image"] = torch.cat((batch["target"]["image"],batch["context"]["image"][:, 1:2, :, :]), dim=1)
        # # target_gt = batch["target"]["image"]







        # src_pose_inv = torch.inverse(re_pose)



        # delta_T = torch.matmul(src_pose_inv, ne_pose)



        # poseloss = pose_loss(pred_poses, target_poses)
        # t_pred = pred_poses[..., :3, 3]
        # t_target = delta_T[..., :3, 3]
        

        # R_pred = pred_poses[..., :3, :3]
        # R_target = delta_T[..., :3, :3]
        # poseloss = pose_loss_zeng(delta_T_encoder, R_target,t_target)


        

        # poseloss = pose_loss(pred_poses, target_poses)
        

        # translation_loss = torch.sum((t_pred - t_target)**2, dim=-1)
        


        # I = torch.eye(3, device=pred_poses.device).expand_as(R_pred)
        # R_product = torch.matmul(R_pred, R_target.transpose(-1, -2))
        # rotation_error = I - R_product
        

        # rotation_loss = torch.sum(rotation_error**2, dim=(-2, -1))
        

        # poseloss = 10*translation_loss + 20 * rotation_loss


        # wandb.log({
        #     "pose/rot_error_deg": rot_deg_error,
        #     "pose/trans_error_cm": trans_error*100,
        #     "pose/combined_loss": poseloss.item()
        # })


        # wandb.log({"poseloss":poseloss})



        output = self.decoder.forward(
            gaussians,
            # updated_extrinsics,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        # output_yuce = self.decoder.forward(
        #     gaussians,
        #     batch["context"]["extrinsics"],
        #     batch["context"]["intrinsics"],
        #     batch["context"]["near"],
        #     batch["context"]["far"],
        #     (256,256),
        #     depth_mode=None,
        # )
        target_gt = batch["target"]["image"]

        

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        # if self.existing_scene != batch["scene"] and self.num_save!= 0:
        if 0:
            # out_gt = np.transpose(target_gt[0][-1].cpu().numpy(),(1, 2, 0))
            # wandb.log({"out/gtimage1": wandb.Image(out_gt)})
            # # out_gt = np.transpose(target_gt[0][1].cpu().numpy(),(1, 2, 0))
            # # wandb.log({"gtimage2": wandb.Image(out_gt)})
            # out_put = np.transpose(output.color[0][-1].cpu().detach().numpy(),(1, 2, 0))
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            gt_all   = target_gt[0].cpu().numpy()          # (T, C, H, W)
            pred_all = output.color[0].cpu().detach().numpy()


            gt_all   = np.transpose(gt_all,   (0, 2, 3, 1))
            pred_all = np.transpose(pred_all, (0, 2, 3, 1))

            T = gt_all.shape[0]
            H, W = gt_all.shape[1:3]


            fig, ax = plt.subplots(2, T, figsize=(T*3, 6))
            if T == 1:
                ax = ax.reshape(2, 1)

            for t in range(T):

                ax[0, t].imshow(np.clip(gt_all[t], 0, 1))
                ax[0, t].set_title(f'GT {t}')
                ax[0, t].axis('off')


                ax[1, t].imshow(np.clip(pred_all[t], 0, 1))
                ax[1, t].set_title(f'Pred {t}')
                ax[1, t].axis('off')

            plt.tight_layout()
            plt.show()
            # # wandb.log({"out/putimage1": wandb.Image(out_put/ out_put.max())})
            # # out_put_yuce = np.transpose(output_yuce.color[0][-1].cpu().detach().numpy(),(1, 2, 0))
            # # wandb.log({"out_put_yuce/putimage1": wandb.Image(out_put_yuce/ out_put_yuce.max())})
            
        # if self.existing_scene != batch["scene"] and self.num_save!= 0:
        #     pose_loss_out = self.pose_save/self.num_save
        #     self.existing_scene = batch["scene"]
        #     self.pose_save=0
        #     self.num_save = 0
        #     # # wandb.log({"rot_loss":rot_loss})
        #     # # wandb.log({"trans_loss":trans_loss})
        #     self.log("pose_sum/loss",pose_loss_out)
        # self.num_save += 1
        # self.pose_save += poseloss
        
        # self.log("gtimage",target_gt[0][0])
        # self.log("gtimage",output.color[0])
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step,out_ex, delta_T_encoder)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss 
        total_loss = total_loss
        self.log("loss/total", total_loss)
        with torch.no_grad():
            if self.existing_scene != batch["scene"] and self.num_save!= 0:
                pose_loss_out = self.pose_save/self.num_save
                self.existing_scene = batch["scene"]
                self.pose_save=0
                self.num_save = 0
                # # wandb.log({"rot_loss":rot_loss})
                # # wandb.log({"trans_loss":trans_loss})
                self.log("total_loss_epico/epico",pose_loss_out)
            self.num_save += 1
            self.pose_save += total_loss

        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}; "
                # f"poss_loss = {poseloss:.6f}; "
                # f"prot_deg_error = {rot_deg_error:.6f};"
                # f"trans_error = {trans_error:.6f};"
                f"psnr = {psnr_probabilistic.mean():.6f};"
                f"lr = {lr}"

            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor
        self.log("lr", lr) 

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        return total_loss
        # return poseloss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if self.pose_all == []:
            self.pose_all.append(batch["context"]["extrinsics"][0][0])
            self.pose_gt.append(batch["context"]["extrinsics"][0][0])
            self.image_all.append(batch["context"]["image"][0][0])
            self.near = batch["target"]["near"][0][0]
            self.far = batch["target"]["far"][0][0]
            self.intrinsics = batch["target"]["intrinsics"][0][0]
        else:
            batch["context"]["extrinsics"][0][0] = self.pose_all[-1]
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, out_ex = self.encoder(
                batch["context"],
                self.global_step,
                existing_gaussians=self.existing_gaussians,
                deterministic=False,
            )
        updated_extrinsics = out_ex
        if self.existing_gaussians == None:
            self.existing_gaussians = gaussians
        else:
            self.existing_gaussians = merge_gaussians(self.existing_gaussians,gaussians)
        self.pose_gt.append(batch["context"]["extrinsics"][0][1])
        self.pose_all.append(updated_extrinsics[0][1])
        self.image_all.append(batch["context"]["image"][0][1])
        

        # if len(self.pose_all) % 5 == 0 and len(self.pose_all) >= 5:

        #     with torch.enable_grad():

        #         gaussian_params = [
        #             self.existing_gaussians.means.clone().detach().requires_grad_(True),
        #             self.existing_gaussians.covariances.clone().detach().requires_grad_(True),
        #             self.existing_gaussians.harmonics.clone().detach().requires_grad_(True),
        #             self.existing_gaussians.opacities.clone().detach().requires_grad_(True)
        #         ]
        #         pose_params = [p.clone().detach().requires_grad_(True) for p in self.pose_all[-5:]]
                

        #         optimizer = torch.optim.Adam(gaussian_params , lr=0.001)
                

        #         for _ in range(10):
        #             optimizer.zero_grad()
        #             total_loss = 0
                    

        #             for i in range(len(self.pose_all)-5, len(self.pose_all)):

        #                 current_pose = pose_params[i-(len(self.pose_all)-5)].unsqueeze(0).unsqueeze(0)
        #                 current_intrinsics = self.intrinsics.unsqueeze(0).unsqueeze(0)
        #                 current_near = self.near.unsqueeze(0).unsqueeze(0)
        #                 current_far = self.far.unsqueeze(0).unsqueeze(0)
                        
        #                 output = self.decoder.forward(
        #                     Gaussians(
        #                         means=gaussian_params[0],
        #                         covariances=gaussian_params[1],
        #                         harmonics=gaussian_params[2],
        #                         opacities=gaussian_params[3]
        #                     ),
        #                     current_pose,
        #                     current_intrinsics,
        #                     current_near,
        #                     current_far,
        #                     (h, w),
        #                     depth_mode=None,
        #                 )
        #                 loss = torch.nn.functional.l1_loss(
        #                     output.color[0], 
        #                     self.image_all[i].unsqueeze(0)
        #                 )
        #                 total_loss += loss
                    
        #             total_loss.backward()
        #             optimizer.step()
            

        #     self.existing_gaussians = Gaussians(
        #         means=gaussian_params[0].detach(),
        #         covariances=gaussian_params[1].detach(),
        #         harmonics=gaussian_params[2].detach(),
        #         opacities=gaussian_params[3].detach()
        #     )
        #     self.pose_all[-5:] = [p.detach() for p in pose_params]
            

            # scales = torch.exp(self.existing_gaussians.covariances[:, :3])
            # mean_scales = torch.mean(scales, dim=1)
            # alphas = torch.sigmoid(self.existing_gaussians.opacities)
            

            # keep_mask = (mean_scales <= 0.1) & (alphas >= 0.01)
            

            # self.existing_gaussians = Gaussians(
            #     means=self.existing_gaussians.means[keep_mask],
            #     covariances=self.existing_gaussians.covariances[keep_mask],
            #     harmonics=self.existing_gaussians.harmonics[keep_mask],
            #     opacities=self.existing_gaussians.opacities[keep_mask]
            # )




        # (scene,) = batch["scene"]
        # name = get_cfg()["wandb"]["name"]
        # path = self.test_cfg.output_path / name
        # images_prob = output.color
        # rgb_gt = batch["target"]["image"]

        # # Save images.
        # if self.test_cfg.save_image:
        #     for index, color in zip(batch["target"]["index"][0], images_prob):
        #         save_image(color, path / scene / f"color/{index:0>6}.png")

        # # save video
        # if self.test_cfg.save_video:
        #     frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
        #     save_video(
        #         [a for a in images_prob],
        #         path / "video" / f"{scene}_frame_{frame_str}.mp4",
        #     )

        # # compute scores
        # if self.test_cfg.compute_scores:
        #     if batch_idx < self.test_cfg.eval_time_skip_steps:
        #         self.time_skip_steps_dict["encoder"] += 1
        #         self.time_skip_steps_dict["decoder"] += v
        #     rgb = images_prob

        #     if f"psnr" not in self.test_step_outputs:
        #         self.test_step_outputs[f"psnr"] = []
        #     if f"ssim" not in self.test_step_outputs:
        #         self.test_step_outputs[f"ssim"] = []
        #     if f"lpips" not in self.test_step_outputs:
        #         self.test_step_outputs[f"lpips"] = []
        #     if f"scene" not in self.test_step_outputs:
        #         self.test_step_outputs[f"scene"] = []
            
        #     self.test_step_outputs[f"psnr"].append(
        #         compute_psnr(rgb_gt, rgb).mean().item()
        #     )
        #     self.test_step_outputs[f"ssim"].append(
        #         compute_ssim(rgb_gt, rgb).mean().item()
        #     )
        #     self.test_step_outputs[f"lpips"].append(
        #         compute_lpips(rgb_gt, rgb).mean().item()
        #     )
        #     self.test_step_outputs[f"scene"].append(
        #         batch['scene'][0]
        #     )

    def on_test_end(self) -> None:
        _,h, w = self.image_all[0].shape
        psnr_all = 0

        for i in range(len(self.pose_all)):
            output = self.decoder.forward(
                self.existing_gaussians,
                self.pose_gt[i].unsqueeze(0).unsqueeze(0),
                self.intrinsics.unsqueeze(0).unsqueeze(0),
                self.near.unsqueeze(0).unsqueeze(0),
                self.far.unsqueeze(0).unsqueeze(0),
                (h, w),
                depth_mode=None,
            )
            images_prob = output.color[0]
            rgb_gt = self.image_all[i].unsqueeze(0)
            psnr_all += compute_psnr(rgb_gt, images_prob).mean().item()

        psnr = psnr_all/len(self.pose_all)


        # name = get_cfg()["wandb"]["name"]
        # out_dir = self.test_cfg.output_path / name
        # saved_scores = {}
        # if self.test_cfg.compute_scores:
        #     self.benchmarker.dump_memory(out_dir / "peak_memory.json")
        #     self.benchmarker.dump(out_dir / "benchmark.json")

        #     for metric_name, metric_scores in self.test_step_outputs.items():
        #         if metric_name == 'scene':
        #             with (out_dir / f"{metric_name}_all.json").open("w") as f:
        #                 json.dump(metric_scores, f)
        #         else:
        #             avg_scores = sum(metric_scores) / len(metric_scores)
        #             saved_scores[metric_name] = avg_scores
        #             print(metric_name, avg_scores)
        #             with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
        #                 json.dump(metric_scores, f)
        #             metric_scores.clear()

        #     for tag, times in self.benchmarker.execution_times.items():
        #         times = times[int(self.time_skip_steps_dict[tag]) :]
        #         saved_scores[tag] = [len(times), np.mean(times)]
        #         print(
        #             f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
        #         )
        #         self.time_skip_steps_dict[tag] = 0

        #     with (out_dir / f"scores_all_avg.json").open("w") as f:
        #         json.dump(saved_scores, f)
        #     self.benchmarker.clear_history()
        # else:
        #     self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        #     self.benchmarker.dump_memory(
        #         self.test_cfg.output_path / name / "peak_memory.json"
        #     )
        #     self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        print("1                   ")
        # batch: BatchedExample = self.data_shim(batch)

        # if self.global_rank == 0:
        #     print(
        #         f"validation step {self.global_step}; "
        #         f"scene = {[a[:20] for a in batch['scene']]}; "
        #         f"context = {batch['context']['index'].tolist()}"
        #     )

        # # Render Gaussians.
        # b, _, _, h, w = batch["target"]["image"].shape
        # assert b == 1
        # gaussians_softmax = self.encoder(
        #     batch["context"],
        #     self.global_step,
        #     deterministic=False,
        # )
        # output_softmax = self.decoder.forward(
        #     gaussians_softmax,
        #     batch["target"]["extrinsics"],
        #     batch["target"]["intrinsics"],
        #     batch["target"]["near"],
        #     batch["target"]["far"],
        #     (h, w),
        # )
        # rgb_softmax = output_softmax.color[0]

        # # Compute validation metrics.
        # rgb_gt = batch["target"]["image"][0]
        # for tag, rgb in zip(
        #     ("val",), (rgb_softmax,)
        # ):
        #     psnr = compute_psnr(rgb_gt, rgb).mean()
        #     self.log(f"val/psnr_{tag}", psnr)
        #     lpips = compute_lpips(rgb_gt, rgb).mean()
        #     self.log(f"val/lpips_{tag}", lpips)
        #     ssim = compute_ssim(rgb_gt, rgb).mean()
        #     self.log(f"val/ssim_{tag}", ssim)

        # # Construct comparison image.
        # comparison = hcat(
        #     add_label(vcat(*batch["context"]["image"][0]), "Context"),
        #     add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
        #     add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        # )
        # self.logger.log_image(
        #     "comparison",
        #     [prep_image(add_border(comparison))],
        #     step=self.global_step,
        #     caption=batch["scene"],
        # )

        # # Render projections and construct projection image.
        # projections = hcat(*render_projections(
        #                         gaussians_softmax,
        #                         256,
        #                         extra_label="(Softmax)",
        #                     )[0])
        # self.logger.log_image(
        #     "projection",
        #     [prep_image(add_border(projections))],
        #     step=self.global_step,
        # )

        # # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image(
        #     "cameras", [prep_image(add_border(cameras))], step=self.global_step
        # )

        # if self.encoder_visualizer is not None:
        #     for k, image in self.encoder_visualizer.visualize(
        #         batch["context"], self.global_step
        #     ).items():
        #         self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        # if self.train_cfg.extended_visualization:
        #     self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step,  existing_gaussians=None,deterministic=False,)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            if result.sum() == 0:
                result += 1e-7
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), 30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
    #     if self.optimizer_cfg.cosine_lr:
    #         warm_up = torch.optim.lr_scheduler.OneCycleLR(
    #                         optimizer, self.optimizer_cfg.lr,
    #                         self.trainer.max_steps + 10,
    #                         pct_start=0.01,
    #                         cycle_momentum=False,
    #                         anneal_strategy='cos',
    #                     )
    #     else:
    #         warm_up_steps = self.optimizer_cfg.warm_up_steps
    #         warm_up = torch.optim.lr_scheduler.LinearLR(
    #             optimizer,
    #             1 / warm_up_steps,
    #             1,
    #             total_iters=warm_up_steps,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": warm_up,
    #             "interval": "step",
    #             "frequency": 1,
    #         },
    #     }
    def configure_optimizers(self):

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        for name, param in self.named_parameters():

            if "existing_gaussians" in name:
                param.requires_grad = False
            if param.requires_grad:
                trainable_params.append(param)
        

        if not trainable_params:
            print("警告：没有可训练参数！添加一个虚拟参数以避免错误")
            dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            trainable_params = [dummy_param]
        

        optimizer = optim.Adam(
            trainable_params, 
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.lr
        )
        

        if self.optimizer_cfg.cosine_lr:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy='cos',
            )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
def update_extrinsics(batch, rotation, translation):

    extrinsics = batch.clone()
    

    q = torch.nn.functional.normalize(rotation, dim=-1)
    # q = rotation
    

    w, x, y, z = q.unbind(-1)
    

    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    
    R[..., 0, 0] = 1 - 2*y**2 - 2*z**2
    R[..., 0, 1] = 2*x*y - 2*w*z
    R[..., 0, 2] = 2*x*z + 2*w*y
    
    R[..., 1, 0] = 2*x*y + 2*w*z
    R[..., 1, 1] = 1 - 2*x**2 - 2*z**2
    R[..., 1, 2] = 2*y*z - 2*w*x
    
    R[..., 2, 0] = 2*x*z - 2*w*y
    R[..., 2, 1] = 2*y*z + 2*w*x
    R[..., 2, 2] = 1 - 2*x**2 - 2*y**2
    

    t = translation.unsqueeze(-1)  # B*(V-1)*3 -> B*(V-1)*3*1
    top = torch.cat([R, t], dim=-1)  # B*(V-1)*3*4
    

    bottom = torch.tensor(
        [0, 0, 0, 1], 
        device=top.device, 
        dtype=top.dtype
    ).reshape(1, 1, 1, 4).repeat(top.size(0), top.size(1), 1, 1)
    
    delta_poses = torch.cat([top, bottom], dim=2)  # B*(V-1)*4*4
    

    base_pose = extrinsics[:, 0, :, :].unsqueeze(1)  # B*1*4*4
    

    new_poses = torch.matmul(delta_poses, base_pose)  # B*(V-1)*4*4
    

    updated_extrinsics = extrinsics.clone()
    updated_extrinsics[:, 1:, :, :] = new_poses
    
    return updated_extrinsics

def merge_gaussians(existing_gaussians: Gaussians, new_gaussians: Gaussians) -> Gaussians:

    if existing_gaussians != None:
        try:
            existing_means = existing_gaussians.means.detach().requires_grad_(False)
            existing_cov = existing_gaussians.covariances.detach().requires_grad_(False)
            existing_harmonics = existing_gaussians.harmonics.detach().requires_grad_(False)
            existing_opacities = existing_gaussians.opacities.detach().requires_grad_(False)
        except:
            existing_means = existing_gaussians.means().detach().requires_grad_(False)
            existing_cov = existing_gaussians.covariances().detach().requires_grad_(False)
            existing_harmonics = existing_gaussians.harmonics().detach().requires_grad_(False)
            existing_opacities = existing_gaussians.opacities().detach().requires_grad_(False)
        

        return Gaussians(
            means=torch.cat([existing_means, new_gaussians.means.detach()], dim=1),
            covariances=torch.cat([existing_cov, new_gaussians.covariances.detach()], dim=1),
            harmonics=torch.cat([existing_harmonics, new_gaussians.harmonics.detach()], dim=1),
            opacities=torch.cat([existing_opacities, new_gaussians.opacities.detach()], dim=1)
        )
    else:
        return Gaussians(
            means= new_gaussians.means.clone,
            covariances= new_gaussians.covariances.clone,
            harmonics= new_gaussians.harmonics.clone,
            opacities= new_gaussians.opacities.clone
        )


