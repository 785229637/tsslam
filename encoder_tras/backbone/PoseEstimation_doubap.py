import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class PoseEstimationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_branch = nn.Sequential(

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [2, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # [2, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # [2, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        

        self.depth_branch = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [2, 128, 72, 72]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [2, 256, 36, 36]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # [2, 128, 18, 18]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),  # [2, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,  # 128(RGB) + 128(Depth)
            num_heads=8,
            batch_first=True
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # [2, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # [2, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        


        self.rotation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # [2, 128, 8, 8]
            nn.Flatten(),  # [2, 128*8*8=8192]
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
        

        self.translation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # [2, 128, 8, 8]
            nn.Flatten(),  # [2, 8192]
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, rgb_feats, depth_feats):

        rgb1, rgb2 = rgb_feats[0:1], rgb_feats[1:2]
        depth1, depth2 = depth_feats[0:1], depth_feats[1:2]
        

        rgb_feat1 = self.rgb_branch(rgb1)  # [1, 128, 16, 16]
        rgb_feat2 = self.rgb_branch(rgb2)
        depth_feat1 = self.depth_branch(depth1)  # [1, 128, 16, 16]
        depth_feat2 = self.depth_branch(depth2)
        

        feat1 = torch.cat([rgb_feat1, depth_feat1], dim=1)  # [1, 256, 16, 16]
        feat2 = torch.cat([rgb_feat2, depth_feat2], dim=1)  # [1, 256, 16, 16]
        

        motion_feat = torch.abs(feat1 - feat2)  # [1, 256, 16, 16]
        

        b, c, h, w = motion_feat.shape
        motion_flat = motion_feat.flatten(2, 3).transpose(1, 2)  # [1, 256, 256]
        attn_out, _ = self.cross_attention(motion_flat, motion_flat, motion_flat)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)  # [1, 256, 16, 16]
        

        fusion_out = self.fusion_conv(attn_out)  # [1, 128, 16, 16]
        

        rotation = self.rotation_head(fusion_out)  # [1, 4]
        translation = self.translation_head(fusion_out)  # [1, 3]
        

        rotation = rotation.unsqueeze(1)
        translation = translation.unsqueeze(1)*0.1

        rotation = F.normalize(rotation, p=2, dim=-1)


        
        return rotation, translation