## TODO: ADD FACE DETECTION FOR REAL WORLD INPUT AS DATASET IS ALREADY FACE CENTERED
import torch
from torchvision.io import read_video
import random

# video_transform= T.Compose([
#     T.Resize((224, 224)),
#     T.RandomHorizontalFlip(p=0.5)
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
# ])

class VideoHandler():
    def __init__(self,transformation,tau,fps,device):
        self.device= device
        self.transformation= transformation
        self.tau= tau
        self.fps= fps
        self.num_frames= int(tau*fps)

    def preprocess(self,file_path):
        vid,_,_= read_video(file_path, pts_unit="sec")
        vid= self.sample_frames(vid)
        vid= vid.float() /255.0
        vid= vid.permute(0, 3, 1, 2)
        vid= vid.to(self.device)
        vid= self.transformation(vid)
        vid= vid.permute(1, 0, 2, 3)     # (T,H,W,C)
        return vid
    
    def sample_frames(self,vid):
        total_frames= vid.shape[0]
        max_start= total_frames - self.num_frames
        start_frame= random.randint(0, max_start)
        end_frame= start_frame + self.num_frames
        vid= vid[start_frame:end_frame]
        return vid