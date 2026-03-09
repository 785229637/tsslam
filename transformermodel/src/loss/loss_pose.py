from dataclasses import dataclass
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


# ------------------------------

# ------------------------------
@dataclass
class LossPoseCfg:
    weight: float
    alpha: float = 1.0
    beta: float = 0


@dataclass
class LossPoseCfgWrapper:
    pose: LossPoseCfg


# ------------------------------

# ------------------------------

class LossPose(Loss[LossPoseCfg, LossPoseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        out_ex: Tensor, 
        delta_T_encoder: Tensor,
    ) -> Float[Tensor, ""]:

        updated_extrinsics = out_ex
        pred_poses = updated_extrinsics[:, 1:2, :, :]  # [B, 1, 4, 4]
        ne_pose = batch["context"]["extrinsics"][:, 1:2, :, :].detach()
        re_pose = batch["context"]["extrinsics"][:, 0:1, :, :].detach()
        

        # src_pose_inv = torch.inverse(re_pose)

        delta_T = batch["context"]["extrinsics"][:, 1:2, :, :].detach()
        

        t_target = delta_T[..., :3, 3]
        R_target = delta_T[..., :3, :3]
        

        loss_val = pose_loss_zeng(
            # delta_T=delta_T_encoder,
            delta_T=pred_poses,
            R_target=R_target,
            t_target=t_target,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            src_pose=re_pose,
            tgt_pose=ne_pose
        )

        return self.cfg.weight * loss_val


def pose_loss_zeng(delta_T, R_target, t_target, alpha=1.0, beta=0.2, src_pose=None, tgt_pose=None):

    pred_rot = delta_T[..., :3, :3]
    pred_trans = delta_T[..., :3, 3]
    


    rot_loss = enhanced_rotation_loss(pred_rot, R_target)

    trans_loss = small_translation_loss(pred_trans, t_target)
    base_loss = rot_loss + trans_loss
    

    consistency_loss = single_pair_consistency(
        pred_delta=delta_T,
        src_pose=src_pose,
        tgt_pose=tgt_pose
    )
    

    return alpha * base_loss + beta * consistency_loss


def enhanced_rotation_loss(pred_R, target_R):

    R_diff = torch.matmul(pred_R.transpose(-1, -2), target_R)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]

    angle_rad = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))
    angle_loss = F.smooth_l1_loss(angle_rad, torch.zeros_like(angle_rad))
    

    identity = torch.eye(3, device=pred_R.device).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
    ortho_loss = torch.norm(torch.matmul(pred_R.transpose(-1, -2), pred_R) - identity, p=2)
    

    det = torch.det(pred_R)
    det_loss = F.smooth_l1_loss(det, torch.ones_like(det))
    
    return 0.5 * angle_loss + 0.3 * ortho_loss + 0.2 * det_loss


def small_translation_loss(pred_t, target_t):

    l1_loss = F.l1_loss(pred_t, target_t)
    

    eps = 1e-8
    pred_norm = torch.norm(pred_t, p=2, dim=-1, keepdim=True) + eps
    target_norm = torch.norm(target_t, p=2, dim=-1, keepdim=True) + eps

    dir_pred = pred_t / pred_norm
    dir_target = target_t / target_norm
    direction_loss = 1 - (dir_pred * dir_target).sum(dim=-1).mean()
    

    scale_ratio = pred_norm / target_norm

    scale_weight = torch.where(target_norm < 0.1, 5.0, 1.0)
    scale_loss = (torch.abs(scale_ratio - 1.0) * scale_weight).mean()
    
    return 0.4 * l1_loss + 0.4 * direction_loss + 0.2 * scale_loss


def single_pair_consistency(pred_delta, src_pose, tgt_pose):
    if src_pose is None or tgt_pose is None:
        return torch.tensor(0.0, device=pred_delta.device)
    

    pred_tgt = torch.matmul(src_pose, pred_delta)  # [B,1,4,4]


    rot_error = torch.norm(pred_tgt[..., :3, :3] - tgt_pose[..., :3, :3], p=2)

    trans_error = F.l1_loss(pred_tgt[..., :3, 3], tgt_pose[..., :3, 3])
    
    return 0.5 * rot_error + 0.5 * trans_error



def matrix_to_quaternion(rot_mat):
    batch_dim = rot_mat.shape[:-2]
    r00, r01, r02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
    r10, r11, r12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
    r20, r21, r22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]
    
    w = torch.sqrt(torch.clamp(1.0 + r00 + r11 + r22, min=1e-8)) / 2.0
    x = torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=1e-8)) / 2.0
    y = torch.sqrt(torch.clamp(1.0 - r00 + r11 - r22, min=1e-8)) / 2.0
    z = torch.sqrt(torch.clamp(1.0 - r00 - r11 + r22, min=1e-8)) / 2.0
    
    x = x * torch.sign(r21 - r12)
    y = y * torch.sign(r02 - r20)
    z = z * torch.sign(r10 - r01)
    
    return torch.stack([w, x, y, z], dim=-1).view(*batch_dim, 4)


def rotation_loss(pred_R, target_R):
    R_diff = torch.matmul(pred_R.transpose(-1, -2), target_R)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle_rad = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    return F.smooth_l1_loss(angle_rad, torch.zeros_like(angle_rad))

def log_translation_loss(pred_t, target_t):
    pred_norm = torch.norm(pred_t, p=2, dim=-1, keepdim=True)
    target_norm = torch.norm(target_t, p=2, dim=-1, keepdim=True)
    direction_loss = 1 - F.cosine_similarity(pred_t, target_t, dim=-1)
    scale_loss = torch.abs(pred_norm - target_norm) / (target_norm + 1e-6)
    return 0.7 * direction_loss.mean() + 0.3 * scale_loss.mean()

def quaternion_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w,
        2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w,
        2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y
    ], dim=-1).view(*(q.shape[:-1]), 3, 3)
