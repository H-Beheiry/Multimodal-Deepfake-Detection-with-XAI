import torch
import torch.nn as nn
from torchvision import models

class AudioResNet18(nn.Module):
    def __init__(self):
        super(AudioResNet18, self).__init__()
        self.model= models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        original_weights= self.model.conv1.weight.data
        self.model.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight.data= original_weights.sum(1, keepdim=True) / 3.0
        num_features= self.model.fc.in_features
        self.model.fc= nn.Linear(num_features, 2)
        
    def forward(self, x):
        if x.size(2) < 64:
             x= torch.nn.functional.interpolate(x, size=(64, x.size(3)), mode='bilinear', align_corners=False)
        return self.model(x)