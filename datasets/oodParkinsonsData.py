import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def correct_batch(batch):
  while len(batch)<4000:
    batch.append(0)
  return batch[:4000]

class oodParkinsonsData(Dataset):

    # def __init__(self, transform=None, rest=True):
    #     self.result = []
    #     self.labels=[]
        
    #     if rest:
    #       act = [2,3,6,7]
    #     else:
    #       act = [1,4,5,7]
           
    #     k = 0
    #     for filename in tqdm(os.listdir("/home/anasa2/deep_Mahalanobis_detector/Other Parkinson_s Dataset/")):
    #         df = pd.read_csv("/home/anasa2/deep_Mahalanobis_detector/Other Parkinson_s Dataset/"+filename)
    #         df = df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()

    #         start = True
    #         for data in df.iterrows():
    #           data=data[1]
    #           # if(data[6] not in act):
    #           #   continue

    #           if(data[6]!=7):
    #             curr=1
    #           else:
    #             curr=0

    #           if start:
    #             self.result.append([])
    #             self.result[k].append([])
    #             self.result[k].append([])
    #             self.result[k].append([])
    #             self.labels.append(curr)
    #             prev = 1
    #             start=False 

    #           if(len(self.result[k][0])>=4000|prev!=curr):
    #             stdev = np.std(np.asarray(self.result[k]))
    #             mean = np.mean(np.asarray(self.result[k]))
    #             self.result[k] = ((np.asarray(self.result[k])-mean)/stdev).tolist()

    #             self.result[k][0] = correct_batch(self.result[k][0])
    #             self.result[k][1] = correct_batch(self.result[k][1])
    #             self.result[k][2] = correct_batch(self.result[k][2])
    #             k=k+1
    #             self.result.append([])
    #             self.result[k].append([])
    #             self.result[k].append([])
    #             self.result[k].append([])
    #             self.labels.append(curr)

    #           self.result[k][0].append(data[3])
    #           self.result[k][1].append(data[4])
    #           self.result[k][2].append(data[5])


    #         # stdev = np.std(np.asarray(self.result[k]))
    #         # mean = np.mean(np.asarray(self.result[k]))
    #         # self.result[k] = ((np.asarray(self.result[k])-mean)/stdev).tolist()
            
    #         # self.result[k][0] = correct_batch(self.result[k][0])
    #         # self.result[k][1] = correct_batch(self.result[k][1])
    #         # self.result[k][2] = correct_batch(self.result[k][2])
    #         # k = k+1




    # def __len__(self):
    #     return len(self.result)

    # def __getitem__(self, idx):
    #     return [np.asarray(self.result[idx]), self.labels[idx]]
    
    def __init__(self, transform=None):
          self.result = []
          self.labels=[]
          k = 0
          for filename in tqdm(os.listdir("/home/anasa2/deep_Mahalanobis_detector/Other Parkinson_s Dataset/")):
            if filename.endswith(".csv"):
              self.result.append([])
              self.result[k].append([])
              self.result[k].append([])
              self.result[k].append([])
              df = pd.read_csv("/home/anasa2/deep_Mahalanobis_detector/Other Parkinson_s Dataset/"+filename)
              df = df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
              start = True
              for data in df.iterrows():
                data=data[1]
                if(data[6] in [4,5]):
                  continue

                if(data[6]!=7):
                  curr=1
                else:
                  curr=0

                if start:
                  self.labels.append(curr)
                  prev = 1
                  start=False 

                if(len(self.result[k][0])>=4000|prev!=curr):
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
                  self.labels.append(curr)

                self.result[k][0].append(data[3])
                self.result[k][1].append(data[4])
                self.result[k][2].append(data[5])


              stdev = np.std(np.asarray(self.result[k]))
              mean = np.mean(np.asarray(self.result[k]))
              self.result[k] = ((np.asarray(self.result[k])-mean)/stdev).tolist()
              self.result[k][0] = correct_batch(self.result[k][0])
              self.result[k][1] = correct_batch(self.result[k][1])
              self.result[k][2] = correct_batch(self.result[k][2])
              k = k+1




    def __len__(self):
          return len(self.result)

    def __getitem__(self, idx):
          return [np.asarray(self.result[idx]), self.labels[idx]]