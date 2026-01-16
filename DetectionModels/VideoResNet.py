import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class VideoResNet(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64, lstm_layers=1, freeze_backbone=True):
        super(VideoResNet, self).__init__()
# 1. Strong Eyes (EfficientNet Backbone) - Frozen
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = efficientnet_b0(weights=weights)
        self.backbone = nn.Sequential(base_model.features, base_model.avgpool)
        self.feature_dim = 1280
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. The "Bottleneck" (Squeeze 1280 -> 64)
        # This forces the model to keep only the most vital info per frame
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        
        # 3. Temporal Detector (1D Convolution)
        # Kernel Size 3: Looks at (Previous, Current, Next) frames together
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6) # High dropout to prevent memorization
        )
        
        # 4. Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_embedding=False):
        B, C, T, H, W = x.shape
        
        # --- Spatial Pass ---
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.backbone(x) # (B*T, 1280, 1, 1)
        features = torch.flatten(features, 1) # (B*T, 1280)
        
        # Squeeze dimensions
        features = self.bottleneck(features) # (B*T, 64)
        
        # --- Temporal Pass ---
        # Reshape for Conv1D: (Batch, Channels, Time) -> (B, 64, T)
        features = features.view(B, T, -1).permute(0, 2, 1)
        
        # Apply 1D Convolution across time
        temporal_features = self.temporal_cnn(features) # Shape: (B, 64, T)
        
        # Global Max Pooling (Find the moment with the *strongest* fake artifact)
        # We take the max across time because if *any* frame is fake, the video is fake.
        embedding = torch.max(temporal_features, dim=2)[0] # Shape: (B, 64)
        
        if return_embedding:
            return embedding
            
        logits = self.classifier(embedding)
        return logits