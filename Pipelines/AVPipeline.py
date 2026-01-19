import sys
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io

from Pipelines.VideoPipeline import VideoPipeline
from Pipelines.AudioPipeline import AudioPipeline

class AVPipeline:
    def __init__(self, av_model, vid_transform, aud_transform, vid_params, aud_params, device="cpu"):
        self.device= device
        self.model= av_model.to(device)
        self.model.eval()
        self.video_pipe= VideoPipeline(
            transformation=vid_transform,
            tau=vid_params['tau'],
            fps=vid_params['fps'],
            device=device,
            model=self.model.video_model
        )
        self.audio_pipe= AudioPipeline(
            model=self.model.audio_model,
            transformation=aud_transform,
            target_sample_rate=aud_params['sample_rate'],
            num_samples=vid_params['tau'],
            device=device
        )

    def predict(self, file_path):
        vid_tensor_raw= self.video_pipe.preprocess(file_path)
        aud_tensor_raw= self.audio_pipe.preprocess(file_path)
        vid_input= vid_tensor_raw.unsqueeze(0).to(self.device)
        aud_input= aud_tensor_raw.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits= self.model(vid_input, aud_input)
            self.pred= torch.argmax(logits).item()
        if self.pred == 0:
            self.label= "REAL"
            flag= "green"
        else:
            self.label= "FAKE"
            flag= "red"
        self.video_pipe.vid= vid_tensor_raw
        self.video_pipe.pred= self.pred
        self.audio_pipe.preprocessed_signal= aud_input
        self.audio_pipe.pred= self.pred
        self.audio_pipe.flag= flag
        return self.label

    def explain(self):
        vid_explanations= self.video_pipe.explain()
        aud_explanations= self.audio_pipe.explain()
        return {
            "video": vid_explanations,
            "audio": aud_explanations
        }

    def fig_to_img(self, fig):
        buf= io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img= Image.open(buf)
        return img

    def run(self, file_path):
        label= self.predict(file_path)
        raw_fig_vid, _= self.video_pipe.plot_mid_frame()
        mid_frame_img= self.fig_to_img(raw_fig_vid)
        plt.close(raw_fig_vid)
        raw_fig_aud, _= self.audio_pipe.plot_amp_time()
        signal_img= self.fig_to_img(raw_fig_aud)
        plt.close(raw_fig_aud)
        explanations= self.explain()

        return {
            "prediction": label,
            "mid_frame": mid_frame_img,
            "audio_signal": signal_img,
            "video_explanation": explanations["video"],
            "audio_explanation": explanations["audio"]
        }

    def visualize_result(self, result):
        fig= plt.figure(figsize=(18, 12))
        pred= result['prediction']
        color= "red" if pred == "FAKE" else "green"
        fig.suptitle(f"Model Prediction: {pred}", fontsize=20, fontweight='bold', color=color, y=0.95)
        ax_vid= plt.subplot2grid((3, 6), (0, 1), colspan=2)
        ax_vid.imshow(result['mid_frame'])
        ax_vid.set_title("Original Video Frame", fontsize=14)
        ax_vid.axis('off')
        ax_aud= plt.subplot2grid((3, 6), (0, 3), colspan=2)
        ax_aud.imshow(result['audio_signal'])
        ax_aud.set_title("Original Audio Signal", fontsize=14)
        ax_aud.axis('off')
        xai_titles= ["Saliency Map", "Integrated Gradients", "Layer GradCAM"]
        if 'video_explanation' in result:
            for i, img in enumerate(result['video_explanation']):
                if i < 3: 
                    ax= plt.subplot2grid((3, 6), (1, i*2), colspan=2)
                    ax.imshow(img)
                    if i == 0:
                        ax.set_ylabel("Video Model\nFocus", fontsize=16, fontweight='bold')
                    ax.set_title(f"Video: {xai_titles[i]}", fontsize=12)
                    ax.axis('off')
        if 'audio_explanation' in result:
            for i, img in enumerate(result['audio_explanation']):
                if i < 3:
                    ax= plt.subplot2grid((3, 6), (2, i*2), colspan=2)
                    ax.imshow(img)
                    if i == 0:
                        ax.set_ylabel("Audio Model\nFocus", fontsize=16, fontweight='bold')
                    ax.set_title(f"Audio: {xai_titles[i]}", fontsize=12)
                    ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        final_img= self.fig_to_img(fig)
        plt.close(fig)
        
        return final_img