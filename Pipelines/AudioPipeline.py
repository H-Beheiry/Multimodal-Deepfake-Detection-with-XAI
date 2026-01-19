import sys
import os
import io
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Add parent dir to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Preprocessing.AudioPreprocessor import AudioHandler

class AudioPipeline(AudioHandler):
    def __init__(self, model, transformation, target_sample_rate, num_samples, device="cpu"):
        super().__init__(transformation, target_sample_rate, num_samples, device)
        self.device= device
        self.model= model
        self.model.to(device)
        self.model.eval()

    def predict(self, audio_filepath):
        preprocessed_signal= self.preprocess(audio_filepath)
        self.preprocessed_signal= preprocessed_signal.unsqueeze(1)
        with torch.no_grad():
            logits= self.model(self.preprocessed_signal)
            self.pred= torch.argmax(logits).item()
            if self.pred== 0:
                self.label= "REAL"
                self.flag= "green"
            else:
                self.label= "FAKE"
                self.flag= "red"
        return self.label

    def fig_to_img(self, fig):
        buf= io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img= Image.open(buf)
        return img
    
    def explain(self):
        figs= self.plot_processed_explination(
            self.preprocessed_signal, 
            self.model, 
            self.flag, 
            self.pred
        )
        
        explanation_images= []
        for fig, _ in figs:
             img= self.fig_to_img(fig)
             plt.close(fig) 
             explanation_images.append(img)
        return explanation_images

    def run(self, audio_filepath):
        label= self.predict(audio_filepath)
        raw_fig, _= self.plot_amp_time()
        original_fig= self.fig_to_img(raw_fig)
        plt.close(raw_fig)
        explination_figs= self.explain()
        return {
            "prediction": label,
            "signal": original_fig,
            "explination": explination_figs
        }