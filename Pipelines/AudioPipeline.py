import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Preprocessing.AudioPreprocessor import AudioHandler
import matplotlib.pyplot as plt
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
            logits= self.model(preprocessed_signal)
            pred= torch.argmax(logits)
        return pred

    def fig_to_bytes(self, fig):
        buf= io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    def explain(self):
        figs= self.plot_processed_explination(self.preprocessed_signal, self.model)
        buffers= []
        for fig, _ in figs:
            buf= self.fig_to_bytes(fig)
            plt.close(fig)
            buffers.append(buf)
        return buffers

    def run(self,audio_filepath):
        pred= self.predict(audio_filepath)
        orignal_fig, _= self.plot_amp_time()
        plt.close(orignal_fig)
        orignal_fig= self.fig_to_bytes(orignal_fig)
        explination_fig= self.explain()

        return {
            "prediction": int(pred.item()),
            "orignal_signal": orignal_fig,
            "explination": explination_fig
        }
        
