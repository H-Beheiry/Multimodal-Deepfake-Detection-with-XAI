import torch
import torch.nn as nn

class AVModel(nn.Module):
    def __init__(self, video_model, audio_model, num_classes=2):
        super(AVModel, self).__init__()
        
        self.video_model= video_model
        self.audio_model= audio_model
        self.fusion_input_dim= 64 + 128 
        for param in self.video_model.parameters():
            param.requires_grad= False
        for param in self.audio_model.parameters():
            param.requires_grad= False
        self.gate_layer = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_input_dim),
            nn.Sigmoid()
        )
        self.fusion_mlp= nn.Sequential(
            nn.Linear(self.fusion_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, video_input, audio_input):
        vid_embed= self.video_model(video_input, return_embedding=True) # (B, 64)
        aud_embed= self.audio_model.get_embedding(audio_input) # (B, 128)
        combined_features= torch.cat((vid_embed, aud_embed), dim=1)
        logits= self.fusion_mlp(combined_features)
        return logits