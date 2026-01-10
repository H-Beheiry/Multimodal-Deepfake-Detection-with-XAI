import os
import pandas as pd

def load_paths(csv_path,dataset_path):
    real_video_real_audio_paths= []
    fake_video_real_audio_paths= []
    real_video_fake_audio_paths= []
    fake_video_fake_audio_paths= []
    
    df= pd.read_csv(csv_path)
    df= df.rename(columns={"Unnamed: 9":"file_path"})
    for i in df["file_path"]:
        path= i.removeprefix("FakeAVCeleb/")
        full_path= os.path.join(dataset_path, path)
        full_path= os.path.normpath(os.path.join(dataset_path, path))
        
        if "RealVideo-RealAudio" in full_path:
            real_video_real_audio_paths.append(full_path)
        elif "FakeVideo-RealAudio" in full_path:
            fake_video_real_audio_paths.append(full_path)
        elif "RealVideo-FakeAudio" in full_path:
            real_video_fake_audio_paths.append(full_path)
        elif "FakeVideo-FakeAudio" in full_path:
            fake_video_fake_audio_paths.append(full_path)
    
    print(len(real_video_real_audio_paths)+len(fake_video_real_audio_paths)+len(real_video_fake_audio_paths)+len(fake_video_fake_audio_paths))

    dataset_file_paths= {
        "RealVideo-RealAudio":real_video_real_audio_paths,
        "FakeVideo-RealAudio":fake_video_real_audio_paths,
        "RealVideo-FakeAudio":real_video_fake_audio_paths,
        "FakeVideo-FakeAudio":fake_video_fake_audio_paths
    }
    return dataset_file_paths

def read_audio(file_paths):
    pass

def read_video(file_path):
    pass