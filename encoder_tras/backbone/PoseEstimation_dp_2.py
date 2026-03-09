import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientFeatureMatching(nn.Module):
    def __init__(self, cnn_channels=128, dino_channels=64, fused_channels=256):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2.25, mode='bilinear', align_corners=False),
            nn.Conv2d(cnn_channels, cnn_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels//2, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        

        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_channels, fused_channels//2, kernel_size=1),
            nn.BatchNorm2d(fused_channels//2),
            nn.ReLU(inplace=True)
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_channels, fused_channels//2, kernel_size=1),
            nn.BatchNorm2d(fused_channels//2),
            nn.ReLU(inplace=True)
        )
        

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        

        self.attention = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels//4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features, dino_feature):

        up_features = self.upsample(features)
        

        cnn_feat = self.cnn_proj(up_features)
        dino_feat = self.dino_proj(dino_feature)
        

        fused = torch.cat([cnn_feat, dino_feat], dim=1)
        fused = self.fuse_conv(fused)
        

        attention = self.attention(fused)
        fused = fused * attention
        

        img1_feat, img2_feat = torch.chunk(fused, 2, dim=0)
        

        diff_feat = torch.abs(img1_feat - img2_feat)
        dot_feat = img1_feat * img2_feat
        
        combined = torch.cat([diff_feat, dot_feat], dim=1)
        return combined

class DualPathPoseNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        

        self.rotation_path = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.rotation_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        

        self.translation_path = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SpatialAttentionModule(64)
        )
        self.translation_fc = nn.Sequential(
            nn.Linear(64*9*9, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, x):

        x = self.shared_conv(x)
        

        rot_feat = self.rotation_path(x)
        rot_feat = rot_feat.view(rot_feat.size(0), -1)
        rotation = self.rotation_fc(rot_feat)
        rotation = F.normalize(rotation, p=2, dim=-1)
        

        trans_feat = self.translation_path(x)
        trans_feat = trans_feat.view(trans_feat.size(0), -1)
        translation = self.translation_fc(trans_feat)
        
        return rotation.unsqueeze(1), translation.unsqueeze(1)

class SpatialAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class PoseEstimationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_matching = EfficientFeatureMatching()
        self.pose_head = DualPathPoseNetwork()
        
    def forward(self, features, dino_feature):

        matched_features = self.feature_matching(features, dino_feature)
        

        rotation, translation = self.pose_head(matched_features)
        

        translation = translation * 0.1
        
        return rotation, translation