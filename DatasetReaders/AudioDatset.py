from torch.utils.data import Dataset
import torchaudio
import torch
import os

# TODO: Add more dynamic data loading
# TODO: Add skipping corrupted files [done]
# TODO: Understand resampling, torchaudio.transforms (MELSpectrogram, MFCC)
 
class AudioDataset(Dataset):
    def __init__(self, fake_audio_path, real_audio_path, transformation, target_sample_rate,num_samples, device, corrupted_files):
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
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self,index):
        signal, sr= torchaudio.load(self.all_files[index])
        signal= signal.to(self.device) 
        signal= self.resample_if_needed(signal,sr=sr)
        signal= self.mix_down_if_needed(signal)
        signal= self.right_pad_if_needed(signal)
        signal= self.cut_if_needed(signal)
        signal= self.transformation(signal)
        label= torch.tensor(self.labels[index])
        return signal, label
    
    def resample_if_needed(self,signal,sr):
        if sr!= self.target_sample_rate:
            resampler= torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler= resampler
            signal= resampler(signal)
        return signal

    def mix_down_if_needed(self, signal):
        if signal.shape[0]>1:
            signal= torch.mean(signal, dim=0, keepdim=True)
        return signal

    def right_pad_if_needed(self,signal):
        ## [1,1,1,1,1] --> [1,1,1,1,1,0,0,0]
        length_of_signal= signal.shape[1]
        if length_of_signal < self.num_samples:
            num_missing_samples= self.num_samples - length_of_signal
            last_dim_padding= (0,num_missing_samples)
            signal= torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def cut_if_needed(self,signal):
        if signal.shape[1]> self.num_samples:
            signal= signal[:, :self.num_samples]
        return signal