from torch.utils.data import Dataset
import torchaudio
import torch
import os
 
class AudioDataset(Dataset):
    def __init__(self, dataset_df, AudioPreprocesser):
        super().__init__()
        self.dataset_df= dataset_df.reset_index(drop=True)
        self.AudioPreprocessor= AudioPreprocesser
    def __len__(self):
        return len(self.dataset_df)
    def __getitem__(self,index):
        signal= self.AudioPreprocessor.preprocess((self.dataset_df["path"][index]))
        label= self.dataset_df["audio_class"][index]
        return signal, label