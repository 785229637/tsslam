import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientFeatureMatching(nn.Module):
    def __init__(self, cnn_channels=128, dino_channels=64, fused_channels=192):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(size=(144, 144)),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
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
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        

        self.local_corr = nn.Sequential(
            nn.Conv2d((2*4+1)**2, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        

        self.attention = nn.Sequential(
            nn.Conv2d(fused_channels + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def build_cost_volume(self, feat1, feat2, max_displacement=4):
        b, c, h, w = feat1.shape
        feat2_padded = F.pad(feat2, (max_displacement, max_displacement, 
                                    max_displacement, max_displacement))
        
        cost_volumes = []
        for i in range(2*max_displacement+1):
            for j in range(2*max_displacement+1):
                shifted = feat2_padded[:, :, i:i+h, j:j+w]
                cost = torch.sum(feat1 * shifted, dim=1, keepdim=True)
                cost_volumes.append(cost)
        
        return torch.cat(cost_volumes, dim=1)

    def forward(self, features, dino_feature):

        up_features = self.upsample(features)
        up_features = self.cnn_proj(up_features)
        dino_feature = self.dino_proj(dino_feature)
        

        if up_features.shape[2:] != dino_feature.shape[2:]:
            dino_feature = F.interpolate(dino_feature, size=up_features.shape[2:], mode='bilinear')
        

        fused = torch.cat([up_features, dino_feature], dim=1)
        fused = self.fuse_conv(fused)
        

        img1_feat, img2_feat = torch.chunk(fused, 2, dim=0)
        

        diff_feat = torch.abs(img1_feat - img2_feat)
        

        corr_volume = self.build_cost_volume(img1_feat, img2_feat, max_displacement=4)
        corr_feat = self.local_corr(corr_volume)
        

        attention_input = torch.cat([fused[:1], corr_feat], dim=1)
        attention_map = self.attention(attention_input)
        

        combined = torch.cat([diff_feat * attention_map, corr_feat * attention_map], dim=1)
        return combined

class EnhancedPoseNetwork(nn.Module):
    def __init__(self, in_channels=192):
        super().__init__()

        self.conv_block1 = self._make_residual_block(in_channels, 128, stride=2)  # 144->72
        self.conv_block2 = self._make_residual_block(128, 128, stride=2)         # 72->36
        self.conv_block3 = self._make_residual_block(128, 128, stride=2)         # 36->18
        self.conv_block4 = self._make_residual_block(128, 128, stride=2)         # 18->9
        

        self.skip_fusion = nn.Sequential(
            nn.Conv2d(128*4, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        

        self.translation_branch = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )
        

        self.rotation_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

    def _make_residual_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv_block1(x)     # [1, 128, 72, 72]
        x2 = self.conv_block2(x1)    # [1, 128, 36, 36]
        x3 = self.conv_block3(x2)    # [1, 128, 18, 18]
        x4 = self.conv_block4(x3)    # [1, 128, 9, 9]
        

        x1_resized = F.interpolate(x1, size=(9,9), mode='bilinear', align_corners=False)
        x2_resized = F.interpolate(x2, size=(9,9), mode='bilinear', align_corners=False)
        x3_resized = F.interpolate(x3, size=(9,9), mode='bilinear', align_corners=False)
        fused = torch.cat([x4, x3_resized, x2_resized, x1_resized], dim=1)
        fused = self.skip_fusion(fused)
        

        translation = self.translation_branch(fused)
        rotation = self.rotation_branch(fused)
        
        return rotation.unsqueeze(1), translation.unsqueeze(1)

class PoseEstimationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_matching = EfficientFeatureMatching()


        self.pose_head_out = EnhancedPoseNetwork(in_channels=192)
        

        self.channel_adapter = nn.Conv2d(256, 192, kernel_size=1)

    def forward(self, features, dino_feature):
        matched_features = self.feature_matching(features, dino_feature)
        

        matched_features = self.channel_adapter(matched_features)
        
        rotation, translation = self.pose_head_out(matched_features)
        rotation = F.normalize(rotation, p=2, dim=-1)
        translation = translation * 0.1
        
        return rotation, translation