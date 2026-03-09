import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

class PoseEstimationHead(nn.Module):
    def __init__(self, hidden_dim=256, lstm_hidden=512, lstm_layers=2, dropout_rate=0.3):
        super(PoseEstimationHead, self).__init__()
        

        self.cnn_feature_processor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        

        self.dino_feature_processor = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        


        self.lstm_input_size = hidden_dim * 8 * 8
        

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        

        self.lstm_output_processor = nn.Sequential(
            nn.Linear(lstm_hidden, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        

        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        

        self.translation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )
        

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):

                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        kaiming_normal_(param.data)
                    elif 'bias' in name:
                        param.data.zero_()

                        n = param.size(0)
                        param.data[n//4 : n//2].fill_(1.)
    
    def forward(self, features, dino_feature):

        cnn_feat = self.cnn_feature_processor(features)
        

        dino_feat = self.dino_feature_processor(dino_feature)
        dino_feat = F.interpolate(
            dino_feat, 
            size=(64, 64), 
            mode='bilinear', 
            align_corners=False
        )
        

        fused_feat = torch.cat([cnn_feat, dino_feat], dim=1)
        

        fused_feat = self.fusion_conv(fused_feat)
        

        seq_len = fused_feat.size(0)
        lstm_input = fused_feat.view(1, seq_len, -1)  # [1, 2, lstm_input_size]
        

        lstm_out, _ = self.lstm(lstm_input)  # [1, 2, lstm_hidden]
        

        last_step_out = lstm_out[:, -1, :]  # [1, lstm_hidden]
        

        processed_feat = self.lstm_output_processor(last_step_out)  # [1, 256]
        

        rotations = self.rotation_head(processed_feat)  # [1, 4]
        translations = self.translation_head(processed_feat)  # [1, 3]
        

        rotations = F.normalize(rotations, p=2, dim=1)
        

        rotations = rotations.unsqueeze(0)  # [1, 1, 4]
        translations = translations.unsqueeze(0)  # [1, 1, 3]*
        
        return rotations, translations