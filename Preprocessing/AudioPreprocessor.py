import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Explanations.audioXAI import AudioExplainer
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch

# TODO: Check how the audio input will be taken

class AudioHandler():
    def __init__(self, transformation, target_sample_rate, num_samples, device):
        self.device= device
        self.transformation= transformation.to(device)
        self.target_sample_rate= target_sample_rate
        self.num_samples= num_samples
    
    def preprocess(self,audio_file_path):
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
        if signal.shape[0]>1:
            signal= signal.mean(dim=0)
        self.duration= torch.arange(signal.shape[-1]) / sr
        fig, ax= plt.subplots(figsize=(10,5))
        ax.plot(self.duration, signal.squeeze().numpy())
        ax.set_title("Original Audio Signal")
        ax.set_xlabel("Time /S")
        ax.set_ylabel("Amplitude /Hz")
        return fig, ax

    def plot_processed_explination(self, preprocessed_input,model,flag):
        ae= AudioExplainer(preprocessed_input, model)
        processed_explinations= ae.process_explination()
        figures= []
        explination_methods= list(ae.attributes.keys())
        for i, time_window in enumerate(processed_explinations):
            fig, ax= self.plot_amp_time()
            ax.set_title(f"Explanation from {explination_methods[i]}")
            plt.close(fig)
            for start_time, end_time in time_window:
                ax.axvline(x=start_time, color=flag, linestyle='-', linewidth=2)
                ax.axvline(x=end_time, color=flag, linestyle='-', linewidth=2)
                ax.axvspan(start_time, end_time, color=flag, alpha=0.2)
            figures.append((fig, ax))
        return figures
            
        
        