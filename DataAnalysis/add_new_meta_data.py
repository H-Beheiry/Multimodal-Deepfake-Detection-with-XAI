import os
import av
import torch
import torchvision
import pandas as pd

csv_path= os.path.join("..","datasets","FakeAVCeleb_v1.2","FakeAVCeleb_v1.2","meta_data.csv")
dataset_path= os.path.join("..","datasets","FakeAVCeleb_v1.2","FakeAVCeleb_v1.2")

def load_videos(csv_path,dataset_path):
    dataset_properties= []
    df= pd.read_csv(csv_path)
    df= df.rename(columns={"Unnamed: 9":"file_path"})
    for i,row in df.iterrows():
        folder_path= row["file_path"].removeprefix("FakeAVCeleb/")
        full_path= os.path.join(dataset_path, row["file_path"])
        full_path= os.path.normpath(os.path.join(dataset_path, folder_path,row["path"]))
        
        if "RealVideo-RealAudio" in full_path:
            video_type= "RVRA"
            video_class= 0
        elif "FakeVideo-RealAudio" in full_path:
            video_type= "FVRA"
            video_class= 1
        elif "RealVideo-FakeAudio" in full_path:
            video_type= "RVFA"
            video_class= 1
        elif "FakeVideo-FakeAudio" in full_path:
            video_type= "FVFR"
            video_class= 1

        try:
            with av.open(full_path) as container:
                stream = container.streams.video[0]
                fps= float(stream.average_rate)
                frames= stream.frames
                width= stream.width
                height= stream.height
                duration= float(stream.duration * stream.time_base)

                if len(container.streams.audio) > 0:
                    has_audio= True
                    audio_stream= container.streams.audio[0]
                    audio_channels= audio_stream.channels
                    audio_rate= audio_stream.rate
                    if audio_stream.duration:
                        audio_duration= float(audio_stream.duration * audio_stream.time_base)
                        
        except Exception as e:
            continue        

        
        dataset_properties.append({
            "method":row["method"],
            "video_type":video_type,
            "video_class":video_class,
            "race": row["race"],
            "gender": row["gender"],
            "path": full_path,
            "fps": fps,
            "frame_count": frames,
            "width": width,
            "height": height,
            "duration": duration,
            "has_audio": has_audio,
            "audio_channels": audio_channels,
            "sample_rate": audio_rate,
            "audio_duration": audio_duration,
        })
        
    return pd.DataFrame(dataset_properties)
    
new_meta_data= load_videos(csv_path,dataset_path)
new_meta_data.to_csv("new_meta_data.csv",index=False)