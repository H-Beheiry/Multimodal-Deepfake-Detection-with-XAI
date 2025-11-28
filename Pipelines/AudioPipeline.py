import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Preprocessing.AudioPreprocessor import AudioHandler
import matplotlib.pyplot as plt
from PIL import Image
import torch
import io

class AudioPipeline(AudioHandler):
    def __init__(self, model, transformation, target_sample_rate, num_samples, device="cpu"):
        super().__init__(transformation, target_sample_rate, num_samples, device)
        self.device= device
        self.model= model

    def predict(self, audio_filepath):
        preprocessed_signal= self.preprocess(audio_filepath)
        self.preprocessed_signal= preprocessed_signal.unsqueeze(1)
        with torch.no_grad():
            logits= self.model(self.preprocessed_signal)
            pred= torch.argmax(logits).item()
            if pred==0:
                label= "REAL"
                self.flag= "green"
            else:
                label= "FAKE"
                self.flag= "red"
        return label

    def fig_to_img(self, fig):
            buf= io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img= Image.open(buf)
            return img
    
    def explain(self):
        figs= self.plot_processed_explination(self.preprocessed_signal, self.model,self.flag)
        explanation_images = []
        for fig,_ in figs:
             img = self.fig_to_img(fig)
             plt.close(fig) 
             explanation_images.append(img)
        return explanation_images

    def run(self,audio_filepath):
        pred= self.predict(audio_filepath)
        raw_fig, _= self.plot_amp_time()
        original_fig= self.fig_to_img(raw_fig)
        plt.close(raw_fig)
        explination_figs= self.explain()

        return {
            "prediction": pred,
            "signal": original_fig,
            "explination": explination_figs
        }
        
