from torch.utils.data import Dataset
import torchaudio
import torch
import os

# TODO: Add more dynamic data loading
# TODO: Add skipping corrupted files [done]
# TODO: Understand resampling, torchaudio.transforms (MELSpectrogram, MFCC)
 
class AudioDataset(Dataset):
    def __init__(self, fake_audio_path, real_audio_path, transformation, target_sample_rate,num_samples, device, corrupted_files, AudioPreprocesser):
        super().__init__()
        self.real_audio_files= [os.path.join(real_audio_path, file) for file in os.listdir(real_audio_path) if file.endswith('.wav')]
        self.real_audio_files=  [f for f in self.real_audio_files if os.path.basename(f) not in corrupted_files]
        self.fake_audio_files= [os.path.join(fake_audio_path, file) for file in os.listdir(fake_audio_path) if file.endswith('.wav')]
        self.fake_audio_files=  [f for f in self.fake_audio_files if os.path.basename(f) not in corrupted_files]
        self.device= device
        self.all_files= self.real_audio_files + self.fake_audio_files
        self.labels = [0]*len(self.real_audio_files) + [1]*len(self.fake_audio_files)
        self.transformation= transformation.to(self.device)
        self.target_sample_rate= target_sample_rate
        self.num_samples= num_samples
        self.AudioPreprocesser= AudioPreprocesser(transformation,target_sample_rate,num_samples,device)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self,index):
        signal= self.audio_preprocesser.Preprocess(self.all_files[index])
        label= torch.tensor(self.labels[index])
        return signal, label