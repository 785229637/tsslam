import torch
from einops import rearrange

from .unimatch.backbone import CNNEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

from transplat.src.geometry.epipolar_lines import get_depth
from encoder_wb.matching.conversions import depth_to_relative_disparity

from transplat.src.model.utils import cam_param_encoder
from encoder_wb.backbone.PoseEstimation_dp_2 import PoseEstimationHead

def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list

class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""
    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
        downscale_factor=8,
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_channels = feature_channels

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )
        
        self.cam_param_encoder = cam_param_encoder(in_channels=128, mid_channels=128, embed_dims=128)

        self.pose_head = PoseEstimationHead().to("cuda:0")
        self.last_coord=None
        self.last_uncertainty=None
        
        

    def normalize_images(self, images):
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)
        return features

    

    def quaternion_to_rotation_matrix(self, quat):

        # quat_norm = torch.norm(quat, dim=1, keepdim=True)

        
        w , x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        batch_size = quat.size(0)
        

        rot_mat = torch.empty((batch_size, 3, 3), device=quat.device)
        rot_mat[:, 0, 0] = 1 - 2*y*y - 2*z*z
        rot_mat[:, 0, 1] = 2*x*y - 2*z*w
        rot_mat[:, 0, 2] = 2*x*z + 2*y*w
        
        rot_mat[:, 1, 0] = 2*x*y + 2*z*w
        rot_mat[:, 1, 1] = 1 - 2*x*x - 2*z*z
        rot_mat[:, 1, 2] = 2*y*z - 2*x*w
        
        rot_mat[:, 2, 0] = 2*x*z - 2*y*w
        rot_mat[:, 2, 1] = 2*y*z + 2*x*w
        rot_mat[:, 2, 2] = 1 - 2*x*x - 2*y*y
        


        # rot_mat += torch.eye(rot_mat.shape[-1]).to("cuda:0") * eps
        # u, s, vh = torch.svd(rot_mat)

        

        # det = torch.det(rot_mat)

        # correction = torch.eye(3, device=rot_mat.device).unsqueeze(0).repeat(batch_size, 1, 1)

        

        # rot_mat = rot_mat @ correction
        


        # rot_mat += torch.eye(rot_mat.shape[-1]).to("cuda:0")  * eps
        # u, s, vh = torch.svd(rot_mat)
        # rot_mat = u @ vh
        

        # final_det = torch.det(rot_mat)
        # if not torch.allclose(final_det, torch.ones_like(final_det), atol=1e-5):

        
        return rot_mat
    
    def construct_transformation_matrix(self, rotation, translation):
        batch_size = rotation.size(0)
        device = rotation.device
        

        rot_mat = self.quaternion_to_rotation_matrix(rotation)  # [B, 3, 3]
        

        T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 4, 4]
        T[:, :3, :3] = rot_mat
        T[:, :3, 3] = translation
        
        return T

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        camk=None,
        c2w=None,
        dino_feature =None,
        pose_ture = False

    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''

        features = self.extract_feature(
            self.normalize_images(images))  # list of features
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features_next = features[::-1]
        # rotations, translations, features_next = self.pose_head(images,camk)

        features_list = [[] for _ in range(2)]
        for feature in features_next:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=1, v=2)
            for idx in range(2):
                features_list[idx].append(feature[:, idx])

        cur_features_list = [x[0] for x in features_list]

        img2world = torch.matmul(c2w, camk)


        if img2world is not None and len(cur_features_list) >= 2:
            if pose_ture == True:
                delta_T = torch.eye(4, device=img2world.device).unsqueeze(0).repeat(c2w.shape[0], 1, 1)  # [B, 4, 4]
            else:


                rotations, translations = self.pose_head(features[0],dino_feature)
                rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
                rotations = [rotations[0]]
                # translations = [translations[0].squeeze(dim=2)]
                translations = [translations[0]]

                

                delta_rotation = rotations[-1]
                delta_translation = translations[-1]

                

                delta_T = self.construct_transformation_matrix(
                    delta_rotation.squeeze(1),  # [B, 4]
                    delta_translation.squeeze(1)  # [B, 3]
                )  # [B, 4, 4]

                # output = self.pose_head(images,camk,self.last_coord,self.last_uncertainty)
                # self.last_coord=output['kf_coord']
                # self.last_uncertainty=output['kf_uncertainty']
                # delta_T = output['relative_pose'][0]
                


                
            # with torch.no_grad():

            ref_pose = c2w[:, -2]  # [B, 4, 4]
            

            updated_pose = torch.matmul(ref_pose, delta_T)

                

            # updated_pose = check_and_fix_SE3(updated_pose)


            



            camk_inv = torch.inverse(camk)
            img2world_1 = torch.matmul(updated_pose, camk_inv[:,1].squeeze(1))

        if return_cnn_features:
            cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        

        # add cam param to features
        feature_list_with_cam = [] 
        # for v_id, cur_features in enumerate(cur_features_list):
        feature_list_with_cam.append(self.cam_param_encoder(cur_features_list[0], img2world[:,0]))
        feature_list_with_cam.append(self.cam_param_encoder(cur_features_list[1], img2world_1))
        cur_features_list = feature_list_with_cam

        # add position to features
        cur_features_list = feature_add_position_list(
            cur_features_list, attn_splits, self.feature_channels)

        # Transformer
        cur_features_list = self.transformer(
            cur_features_list, attn_num_splits=attn_splits)

        features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if return_cnn_features:
            out_lists = [features, cnn_features, updated_pose,delta_T]
        else:
            out_lists = [features, None]

        return out_lists

def to_SE3(T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    T[0, 3, :] = torch.tensor([0, 0, 0, 1], dtype=T.dtype, device=T.device)


    R = T[0, :3, :3]                       # [3,3]
    eps = 1e-8
    R  += torch.eye(R.shape[-1], device=R.device) * eps
    U, _, Vh = torch.linalg.svd(R, full_matrices=False)
    R_new = U @ Vh

    if torch.det(R_new) < 0:
        Vh[2, :] *= -1
        R_new = U @ Vh
    T[0, :3, :3] = R_new
    return T

def check_and_fix_SE3(updated_pose: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    R = updated_pose[0, :3, :3]
    t = updated_pose[0, :3, 3]
    last_row = updated_pose[0, 3]


    ortho = torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=eps)
    det_ok = torch.allclose(torch.det(R), torch.tensor(1.0, device=R.device), atol=eps)
    last_ok = torch.allclose(last_row, torch.tensor([0.0, 0.0, 0.0, 1.0], device=R.device, dtype=R.dtype), atol=eps)

    if ortho and det_ok and last_ok:
        return updated_pose
    else:
        return to_SE3(updated_pose)