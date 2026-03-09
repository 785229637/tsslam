from typing import Tuple
from torch import Tensor, nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class PoseEstimationHead(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        combined_dim = 2 * feature_dim
        

        self.linear_proj = nn.Linear(combined_dim, 256)
        

        self.shared_feature_extractor = nn.Sequential(
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
        )
        

        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 4)
        )
        
        self.translation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 3)
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.001)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, num_views = features.shape[:2]
        

        prev_features = features[:, :-1]
        next_features = features[:, 1:]
        

        combined = torch.cat([prev_features, next_features], dim=-1)
        combined = combined.view(-1, 2 *  self.feature_dim)
        projected = self.linear_proj(combined)  # [batch_size*(num_views-1), 256]
        

        shared_features = self.shared_feature_extractor(projected)
        shared_features += projected
        

        rotation = self.rotation_head(shared_features)
        translation = self.translation_head(shared_features)
        

        rotation = rotation.view(batch_size, num_views-1, 4)
        translation = translation.view(batch_size, num_views-1, 3)

        rotation = rotation *0.1
        translation = translation *0.1
        
        return rotation, translation
