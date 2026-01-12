from torch.utils.data import Dataset
import torchvision
import torch
import os

class VideoDataset(Dataset):
    def __init__(self,dataset_df,VideoPreprocessor,device):
        self.dataset_df= dataset_df
        self.device= device
        self.VideoPreprocessor= VideoPreprocessor
    def __len__(self):
        return len(self.dataset_df)
    def __getitem__(self,index):
        vid= self.VideoPreprocessor.preprocess(self.dataset_df["path"][index])
        label= self.dataset_df["video_class"][index]
        return vid, label
        