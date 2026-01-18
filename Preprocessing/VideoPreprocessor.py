## TODO: ADD FACE DETECTION FOR REAL WORLD INPUT AS DATASET IS ALREADY FACE CENTERED
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Explanations.XAI import ExplanationPipeline
import torch.nn.functional as F
import torch
from torchvision.io import read_video
import matplotlib.pyplot as plt
import random

class VideoHandler():
    def __init__(self,transformation,tau,fps,device):
        self.device= device
        self.transformation= transformation
        self.tau= tau
        self.fps= fps
        self.num_frames= int(tau*fps)

        self.inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def preprocess(self,file_path):
        self.original_vid,_,_= read_video(file_path, pts_unit="sec")
        self.original_vid, self.start_fram_sec= self.sample_frames(self.original_vid)
        vid= self.original_vid.float() /255.0
        vid= vid.permute(0, 3, 1, 2)
        vid= self.transformation(vid)
        vid= vid.to(self.device)
        vid= vid.permute(1, 0, 2, 3)     # (T,H,W,C)
        return vid
    
    def sample_frames(self,vid):
        total_frames= vid.shape[0]
        max_start= total_frames - self.num_frames
        start_frame= random.randint(0, max_start)
        end_frame= start_frame + self.num_frames
        vid= vid[start_frame:end_frame]
        return vid, start_frame/self.fps

    def denormalize_frame(self, frame_tensor):
        frame= frame_tensor.clone().detach().cpu()
        frame= frame * self.inv_std + self.inv_mean
        frame= frame.clamp(0, 1)
        return frame.permute(1, 2, 0).numpy()

    def plot_mid_frame(self, vid_tensor=None):
        if vid_tensor is not None:
            mid_idx= vid_tensor.shape[1] // 2
            frame_tensor= vid_tensor[:, mid_idx, :, :]
            frame_img= self.denormalize_frame(frame_tensor)
        else:
            mid_idx= self.original_vid.shape[0] // 2
            frame_img= self.original_vid[mid_idx].cpu().numpy() / 255.0
        fig, ax= plt.subplots(figsize=(6, 6))
        ax.imshow(frame_img)
        ax.set_title("Input Frame")
        ax.axis("off")
        return fig, ax

    def plot_processed_explination(self, preprocessed_vid, model, pred):
        from Explanations.XAI import ExplanationPipeline
        ep= ExplanationPipeline(model)
        if preprocessed_vid.dim() == 4:
            input_tensor= preprocessed_vid.unsqueeze(0)
        else:
            input_tensor= preprocessed_vid
        explained= ep.explain(input_tensor, pred)
        figures= []
        if preprocessed_vid.dim() == 5:
             vid_data= preprocessed_vid.squeeze(0)
        else:
             vid_data= preprocessed_vid
        mid_idx= vid_data.shape[1] // 2
        background_tensor = vid_data[:, mid_idx, :, :]
        background_img= self.denormalize_frame(background_tensor)
       
        for method_name, attributions in explained.items():
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(background_img)
            if attributions.dim() == 5: 
                dims_to_sum = (0, 1, 2)
            elif attributions.dim() == 4:
                dims_to_sum = (0, 1)
                
            heatmap= torch.sum(torch.abs(attributions), dim=dims_to_sum)
            heatmap= heatmap.unsqueeze(0).unsqueeze(0) 
            target_h, target_w= background_img.shape[0], background_img.shape[1]
            heatmap= F.interpolate(
                    heatmap, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False)
            heatmap= heatmap.squeeze().detach().cpu()
            heatmap= (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap= heatmap.numpy()
            ax.imshow(heatmap, cmap='jet', alpha=0.6)
            
            ax.set_title(f"Explanation: {method_name}")
            ax.axis('off')
            figures.append((fig, ax))
            
        return figures
        