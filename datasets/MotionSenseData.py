import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def correct_batch(batch):
    while len(batch) < 4000:
        batch.append(0)
    return batch[:4000]


class MotionSenseData(Dataset):

    def __init__(self, rest=False):
        if rest:
            act = ["sit", "std"]
        else:
            act = ["dws", "ups", "wlk", "jog"]
        self.result = []
        k = 0
        for dir in tqdm(os.listdir("/home/anasa2/deep_Mahalanobis_detector/MotionSense/")):
            if (dir[:3] in act):
                for filename in os.listdir("/home/anasa2/deep_Mahalanobis_detector/MotionSense/"+dir):
                        try:
                            df = pd.read_csv("/home/anasa2/deep_Mahalanobis_detector/MotionSense/"+dir+"/"+filename)
                            df = df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
                        except:
                            continue
                        
                        self.result.append([])
                        self.result[k].append([])
                        self.result[k].append([])
                        self.result[k].append([])
                        for data in df.iterrows():
                            data = data[1]
                            self.result[k][0].append(data["rotationRate.x"])
                            self.result[k][1].append(data["rotationRate.y"])
                            self.result[k][2].append(data["rotationRate.z"])

                        stdev = np.std(np.asarray(self.result[k]))
                        mean = np.mean(np.asarray(self.result[k]))
                        self.result[k] = ((np.asarray(self.result[k])-mean)/stdev).tolist()
                        
                        self.result[k][0] = correct_batch(self.result[k][0])
                        self.result[k][1] = correct_batch(self.result[k][1])
                        self.result[k][2] = correct_batch(self.result[k][2])
                        
                        k = k+1
              
        self.result = np.asarray(self.result)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        return [np.asarray(self.result[idx]), 0]
