## TODO: ADD FACE DETECTION FOR REAL WORLD INPUT AS DATASET IS ALREADY FACE CENTERED
import torch
from torchvision.io import read_video

# video_transform= T.Compose([
#     T.Resize((224, 224)),
#     T.RandomHorizontalFlip(p=0.5)
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
# ])

class VideoHandler():
    def __init__(self,transformation,num_frames,device):
        self.device= device
        self.transformation= transformation
        self.num_frames= num_frames

    def preprocess(self,file_path):
        vid,_,_= read_video(file_path, pts_unit="sec")
        vid= self.ensure_frame(vid)
        vid= vid.float() /255.0
        vid= vid.permute(0, 3, 1, 2)
        vid= vid.to(self.device)
        vid= self.transformation(vid)
        vid= vid.permute(3, 0, 1, 2)        # (T,H,W,C)
        return vid
    
    def ensure_frame(self,vid):
        vid_frames= vid.shape[0]
        if vid_frames < self.num_frames:
            missing= self.num_frames-vid_frames
            padding= torch.zeros((missing, *video.shape[1:]), dtype=video.dtype)
            video= torch.cat((video, padding), dim=0)
        elif vid_frames > self.num_frames:
            vid= vid[:self.num_frames]
        return vid