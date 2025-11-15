import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np

# TODO: Check how the audio input will be taken

class AudioPreprocesser():
    def __init__(self, transformation, target_sample_rate, num_samples, device):
        self.device= device
        self.transformation= transformation.to(device)
        self.target_sample_rate= target_sample_rate
        self.num_samples= num_samples
    
    def Preprocess(self,audio_file_path):
        self.orignal_signal, self.orignal_sr= torchaudio.load(audio_file_path)
        signal= self.orignal_signal.to(self.device) 
        signal= self.resample_if_needed(signal, self.orignal_sr)
        signal= self.mix_down_if_needed(signal)
        signal= self.right_pad_if_needed(signal)
        signal= self.cut_if_needed(signal)
        signal= self.transformation(signal)
        return signal

    def resample_if_needed(self,signal,sr):
        if sr!= self.target_sample_rate:
            resampler= torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler= resampler.to(self.device)
            signal= resampler(signal)
        return signal

    def mix_down_if_needed(self, signal):
        if signal.shape[0]>1:
            signal= torch.mean(signal, dim=0, keepdim=True)
        return signal

    def right_pad_if_needed(self,signal):
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

    def plot_amp_time(self,signal=None, sr=None):
        if signal is None:
            signal = self.orignal_signal
        if sr is None:
            sr = self.orignal_sr
        self.duration= torch.arange(signal.shape[-1]) / sr
        fig, ax= plt.subplots(figsize=(5,5))
        ax.plot(self.duration, signal.squeeze().numpy())
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        return fig, ax
        
        