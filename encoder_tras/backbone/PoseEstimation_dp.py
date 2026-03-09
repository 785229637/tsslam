import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMatchingNetwork(nn.Module):
    def __init__(self, cnn_channels=128, dino_channels=64, fused_channels=256):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(size=(144, 144), mode='bilinear', align_corners=False),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        

        self.cnn_proj = nn.Conv2d(cnn_channels, fused_channels, kernel_size=1)
        self.dino_proj = nn.Conv2d(dino_channels, fused_channels, kernel_size=1)
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        

        self.correlation = nn.Sequential(

            nn.Conv2d(fused_channels * 2, fused_channels * 2, kernel_size=1),
            nn.BatchNorm2d(fused_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        

        self.attention = nn.Sequential(
            nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features, dino_feature):

        up_features = self.upsample(features)
        up_features = self.cnn_proj(up_features)
        

        dino_feature = self.dino_proj(dino_feature)
        

        fused = torch.cat([up_features, dino_feature], dim=1)
        fused = self.fuse_conv(fused)
        

        img1_feat, img2_feat = torch.chunk(fused, 2, dim=0)
        


        diff_feat = torch.abs(img1_feat - img2_feat)
        

        b, c, h, w = img1_feat.shape
        img1_flat = img1_feat.view(b, c, h*w).permute(0, 2, 1)  # [1, h*w, c]
        img2_flat = img2_feat.view(b, c, h*w)  # [1, c, h*w]
        corr = torch.matmul(img1_flat, img2_flat)  # [1, h*w, h*w]
        corr = corr.view(b, h, w, h*w).permute(0, 3, 1, 2)  # [1, h*w, h, w]
        corr = F.upsample(corr, size=(h, w), mode='bilinear', align_corners=False)
        corr = self.correlation(torch.cat([img1_feat, img2_feat], dim=1))
        

        attention_map = self.attention(torch.cat([diff_feat, corr], dim=1))
        diff_feat = diff_feat * attention_map
        corr = corr * attention_map
        

        combined = torch.cat([diff_feat, corr], dim=1)
        return combined

class FullPoseNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),  # 144->72
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),         # 72->36
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),         # 36->18
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),          # 18->9
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        

        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        

        self.rotation_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )
        

        self.translation_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )
        

        self.skip_conv1 = nn.Conv2d(256, 64, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x):

        x1 = self.conv_block1(x)     # [1, 256, 72, 72]
        x2 = self.conv_block2(x1)    # [1, 128, 36, 36]
        x3 = self.conv_block3(x2)    # [1, 128, 18, 18]
        x4 = self.conv_block4(x3)    # [1, 64, 9, 9]
        

        skip1 = self.skip_conv1(x1)
        skip1 = F.adaptive_avg_pool2d(skip1, (9, 9))
        
        skip2 = self.skip_conv2(x2)
        skip2 = F.adaptive_avg_pool2d(skip2, (9, 9))
        
        x4 = x4 + skip1 + skip2
        

        avg_pool = self.global_avg_pool(x4).flatten(1)  # [1, 64]
        max_pool = self.global_max_pool(x4).flatten(1)  # [1, 64]
        x = torch.cat([avg_pool, max_pool], dim=1)      # [1, 128]
        

        x = self.shared_fc(x)  # [1, 64]
        

        rotation = self.rotation_fc(x)     # [1, 4]
        translation = self.translation_fc(x)  # [1, 3]
        

        rotation = rotation.unsqueeze(1)    # [1, 1, 4]
        translation = translation.unsqueeze(1)  # [1, 1, 3]
        return rotation, translation

class PoseEstimationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_matching = FeatureMatchingNetwork()
        self.pose_head_out = FullPoseNetwork()

    def forward(self, features, dino_feature):

        matched_features = self.feature_matching(features, dino_feature)
        

        rotation, translation = self.pose_head_out(matched_features)
        rotation = F.normalize(rotation, p=2, dim=-1)
        translation = translation*0.1
        
        return rotation, translation