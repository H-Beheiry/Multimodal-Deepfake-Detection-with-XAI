import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class VideoResNet(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64, lstm_layers=1, freeze_backbone=True):
        super(VideoResNet, self).__init__()
        weights= EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model= efficientnet_b0(weights=weights)
        self.backbone= nn.Sequential(base_model.features, base_model.avgpool)
        self.feature_dim= 1280
        
        for param in self.backbone.parameters():
            param.requires_grad= False
            
        self.bottleneck= nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        
        self.temporal_cnn= nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        

        self.classifier= nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_embedding=False):
        B, C, T, H, W= x.shape
        
        x= x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features= self.backbone(x) # (B*T, 1280, 1, 1)
        features= torch.flatten(features, 1) # (B*T, 1280)
        
        features= self.bottleneck(features) # (B*T, 64)
        
        features= features.view(B, T, -1).permute(0, 2, 1)

        temporal_features= self.temporal_cnn(features) # Shape: (B, 64, T)
        embedding= torch.max(temporal_features, dim=2)[0] # Shape: (B, 64)
        
        if return_embedding:
            return embedding
            
        logits= self.classifier(embedding)
        return logits