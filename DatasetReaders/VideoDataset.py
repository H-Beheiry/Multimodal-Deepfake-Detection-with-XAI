from torch.utils import Dataset
import torchvision
import torch
import os


class VideoDataset(Dataset):
    def __init__(self,dataset_path,VideoPreprocessor):
        self.all_files= None
        pass
    
    
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self,index):
        pass
        