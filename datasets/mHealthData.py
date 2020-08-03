import os
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
from tqdm import tqdm
import numpy as np

def correct_batch(batch):
    while len(batch)<4000:
        batch.append(0)
    return batch[:4000]

class mHealthData(Dataset):

    def __init__(self, transform=None):
        self.result = []
        self.labels=[]
        k = 0
        for filename in tqdm(os.listdir("/home/anasa2/deep_Mahalanobis_detector/mHealth Data/")):
        
            try:
                df = pd.read_csv("/home/anasa2/deep_Mahalanobis_detector/mHealth Data/"+filename, delimiter= '\s+', index_col=False)
            except:
                continue
            
            self.result.append([])
            self.result[k].append([])
            self.result[k].append([])
            self.result[k].append([])
            
            start = True
            eof = False
            for data in df.iterrows():
              data=data[1]
              if(data[23] not in [1,2,3,7,8]):
                 continue
    
              if(len(self.result[k][0])>=4000):
                eof = True
                stdev = np.std(np.asarray(self.result[k]))
                mean = np.mean(np.asarray(self.result[k]))
                self.result[k] = ((np.asarray(self.result[k])-mean)/stdev).tolist()

                self.result[k][0] = correct_batch(self.result[k][0])
                self.result[k][1] = correct_batch(self.result[k][1])
                self.result[k][2] = correct_batch(self.result[k][2])
                k=k+1
                self.result.append([])
                self.result[k].append([])
                self.result[k].append([])
                self.result[k].append([])
                self.labels.append(0)

              self.result[k][0].append(data[17])
              self.result[k][1].append(data[18])
              self.result[k][2].append(data[19])
              eof = False

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
        return (self.result[idx],0)