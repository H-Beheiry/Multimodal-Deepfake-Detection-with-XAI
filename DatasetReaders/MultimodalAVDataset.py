from torch.utils.data import Dataset

class MultimodalAVDataset(Dataset):
    def __init__(self, df, video_handler, audio_handler):
        self.df= df.reset_index(drop=True)
        self.video_handler= video_handler
        self.audio_handler= audio_handler
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row= self.df.iloc[index]
        vid_tensor= self.video_handler.preprocess(row["path"])
        start_fps= self.video_handler.start_fram_sec
        aud_tensor= self.audio_handler.preprocess(row["path"],start_fps)
        label= row["overall_class"]
        return vid_tensor, aud_tensor, label