import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Preprocessing.VideoPreprocessor import VideoHandler
import matplotlib.pyplot as plt
from PIL import Image
import torch
import io

class VideoPipeline(VideoHandler):
    def __init__(self,transformation,tau,fps,device,model):
        super().__init__(transformation,tau,fps,device)
        self.device= device
        self.model= model.to(device)
        self.model.eval()
    
    def predict(self,file_path):
        self.vid= self.preprocess(file_path)
        input_tensor= self.vid.unsqueeze(0)
        with torch.no_grad():
            logits= self.model(input_tensor)
            self.pred= torch.argmax(logits).item()
        if self.pred==0:
            label= "REAL"
        else:
            label= "FAKE"
        return label

    def fig_to_img(self, fig):
        buf= io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img= Image.open(buf)
        return img

    def explain(self):
        input_tensor= self.vid.unsqueeze(0)
        figs= self.plot_processed_explination(input_tensor,self.model,self.pred)
        explanation_images= []
        for fig,_ in figs:
             img= self.fig_to_img(fig)
             plt.close(fig) 
             explanation_images.append(img)
        return explanation_images

    def run(self,video_path):
        label= self.predict(video_path)
        raw_fig,_= self.plot_mid_frame()
        original_fig= self.fig_to_img(raw_fig)
        plt.close(raw_fig)
        explination_figs= self.explain()
        return {
            "prediction": label,
            "mid_frame": original_fig,
            "explination": explination_figs
        }