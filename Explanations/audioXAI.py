import torchaudio
import captum
import torch
from Explanations.XAI import ExplanationPipeline

class AudioExplainer():
    def __init__(self, audio_input, model):
        self.attributes= ExplanationPipeline(model).explain(audio_input)
    
    def process_explination(self,HOP_DURATION=0.004):
        results=[]
        for exp in self.attributes.values():
            attr= exp.squeeze()
            max_col= torch.max(attr, dim=0)
            mean_val= max_col[0].mean()
            mean_max_cols_idx= [i for i, val in enumerate(max_col[0]) if val > mean_val]
            mean_max_cols_vals= [max_col[0][i] for i in mean_max_cols_idx]

            lengths = []
            count = 1
            for i in range(1, len(mean_max_cols_idx)):
                if mean_max_cols_idx[i] - mean_max_cols_idx[i-1]<= 3:
                    count+= 1
                else:
                    lengths.append(count)
                    count= 1  
            lengths.append(count)
            mean_length= sum(lengths) / len(lengths)

            result= []
            start_idx= 0  
            count= 1      
            for i in range(1, len(mean_max_cols_idx)):
                if  mean_max_cols_idx[i] - mean_max_cols_idx[i-1]<= 2 :
                    count+= 1
                else:
                    if count > mean_length:
                        result.append([start_idx, i-1])
                    start_idx = i
                    count= 1
            if count > mean_length:
                result.append([start_idx, len(mean_max_cols_idx)-1])

            result= [[x * HOP_DURATION, y * HOP_DURATION] for x, y in result]
            results.append(result)
        return results
            