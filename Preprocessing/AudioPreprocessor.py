import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import av
import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Explanations.audioXAI import AudioExplainer

class AudioHandler:
    def __init__(self, transformation, target_sample_rate, tau, device):
        self.device= device
        self.transformation= transformation.to(device)
        self.tau= tau
        self.target_sample_rate= target_sample_rate
        self.num_samples= int(tau * target_sample_rate)
        self.start_time= 0.0
        self.inference_waveform= None 
        self.orignal_signal= None
        self.orignal_sr= None

    def _get_signal(self, path):
        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            with av.open(path) as container:
                stream= container.streams.audio[0]
                resampler= av.AudioResampler(format='fltp', layout='mono', rate=self.target_sample_rate)
                frames= []
                for frame in container.decode(stream):
                    frames.extend(resampler.resample(frame))
                if not frames:
                    raise ValueError(f"No audio frames found in {path}")
                signal_np= np.concatenate([f.to_ndarray() for f in frames], axis=1)
                signal= torch.from_numpy(signal_np).float()
                return signal, self.target_sample_rate
        else:
            return torchaudio.load(path)

    def resample_if_needed(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler= torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal= resampler(signal)
        return signal

    def mix_down_if_needed(self, signal):
        if signal.shape[0] > 1:
            signal= torch.mean(signal, dim=0, keepdim=True)
        return signal

    def sample_audio(self, signal, start_time=None):
        total_samples= signal.shape[1]
        max_start= total_samples - self.num_samples
        if max_start <= 0:
            return signal[:, :self.num_samples], 0.0
        if start_time is None:
            start_sample= random.randint(0, max_start)
            selected_time= start_sample / self.target_sample_rate
        else:
            start_sample= int(start_time * self.target_sample_rate)
            selected_time= start_time
            
        end_sample= start_sample + self.num_samples
        return signal[:, start_sample:end_sample], selected_time

    def preprocess(self, audio_file_path, start_time=None):
        self.orignal_signal, self.orignal_sr= self._get_signal(audio_file_path)
        signal_for_sampling= self.orignal_signal.to(self.device)
        signal_for_sampling= self.resample_if_needed(signal_for_sampling, self.orignal_sr)
        cropped_signal, selected_start_time= self.sample_audio(signal_for_sampling, start_time)
        self.start_time= selected_start_time
        self.inference_waveform= cropped_signal.clone()
        final_signal= self.mix_down_if_needed(cropped_signal)
        final_signal= self.transformation(final_signal)
        final_signal= (final_signal - final_signal.min()) / (final_signal.max() - final_signal.min() + 1e-6)
        
        return final_signal

    def plot_amp_time(self, signal=None, sr=None):
        if signal is None:
            signal= self.orignal_signal
            sr= self.orignal_sr
        if sr is None:
            sr= self.target_sample_rate
        if isinstance(signal, torch.Tensor):
            signal= signal.cpu()
        if signal.shape[0] > 1:
            signal= torch.mean(signal, dim=0, keepdim=True)
        duration= torch.arange(signal.shape[-1]) / sr
        fig, ax= plt.subplots(figsize=(10, 5))
        ax.plot(duration, signal.squeeze().numpy())
        ax.set_title("Full Audio Signal")
        ax.set_xlabel("Time /s")
        ax.set_ylabel("Amplitude")
        return fig, ax

    def plot_processed_explination(self, preprocessed_input, model, flag, pred):
        ae= AudioExplainer(preprocessed_input, model, pred)
        processed_explinations= ae.process_explination()
        explination_methods= list(ae.attributes.keys())
        figures= []
        for i, time_window in enumerate(processed_explinations):
            fig, ax= self.plot_amp_time(signal=self.orignal_signal, sr=self.orignal_sr)
            ax.set_title(f"Explanation from {explination_methods[i]} (Region: {self.start_time:.1f}s - {self.start_time + self.tau:.1f}s)")
            roi_end= self.start_time + self.tau
            ax.axvspan(self.start_time, roi_end, color='gray', alpha=0.1, label="Analyzed Region")
            for start_T, end_time in time_window:
                abs_start= start_T + self.start_time
                abs_end= end_time + self.start_time
                ax.axvline(x=abs_start, color=flag, linestyle='-', linewidth=2)
                ax.axvline(x=abs_end, color=flag, linestyle='-', linewidth=2)
                ax.axvspan(abs_start, abs_end, color=flag, alpha=0.2)
            
            figures.append((fig, ax))
            
        return figures