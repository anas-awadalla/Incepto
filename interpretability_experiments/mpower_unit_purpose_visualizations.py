# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import math
import json
import time
import itertools
import torch

# %%

class Attribution(object):
    def __init__(self):
        self.data = {}
    
    def update_unit(self, layer, unit, row):
        if layer not in self.data:
            self.data.update({layer:{}})
        
        if unit not in self.data[layer]:
            # df = {'peaks':[], 'amplitude':[]}
            self.data[layer].update({unit:{'peaks':[], 'amplitude':[]}})
            
        # print(self.data)
        self.data[layer][unit]['peaks'].append(row['peaks'])
        self.data[layer][unit]['amplitude'].append(row['amplitude'])
        
    def get_important_feature(self, layer, unit):
        max_value = float("-inf")
        max_outcome = ""
        
        df = pd.DataFrame(self.data[layer][unit])
        
        for col in df:
            # print(col)
            curr_count = df[col].max()[True]
            # print(curr_count)
            # print(df[col].value_counts())
            if curr_count > max_value:
                max_value = curr_count
                max_outcome = col
        
        return max_outcome

    
    def visualize_model(self, layers):
        total_result = collections.Counter()
        
        layer_result = collections.Counter()
        # detectors = {"perodic":"Shape","peaks":"Shape","spikes":"Shape","amplitude":"Intensity"}
        
        for i in range(1,layers):
            for unit in self.data[i]:
                # total_result[self.get_important_feature(i,unit)]+=1.0
                layer_result[self.get_important_feature(i,unit)]+=1.0
            
            self.plot_detectors(layer_result, f"Model Detectors Visualization for Layer {i}")

            total_result+= layer_result
            
            layer_result = collections.Counter()
                            
        self.plot_detectors(total_result, "Model Detectors Visualization for Entire Model")
                
        
           
                
        
    def plot_detectors(self, result, title):
         
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        data = list(result.values())
        detectors = list(result.keys())

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%".format(pct)


        wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                        textprops=dict(color="w"))

        ax.legend(wedges, detectors,
                title="Detector Type",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=8, weight="bold")

        ax.set_title(title)

        plt.show()        
                
# %%
class testMotionData(Dataset):

    def __init__(self, df, users, root_dir = '/home/jupyter/park/', transform=None):
      
        self.dataset = df
        self.root_dir = root_dir
        self.dataArray = []
        self.resultArray = []
        iterData = iter(self.dataset.iterrows())

        k = 0

        for j,z in zip(iterData,tqdm(range(int(len(self.dataset))))):
          j = j[1]
          healthcode = j[3]
        
          label = users.loc[healthcode][0]
          
          #print(j)
          for i in [8]:
            if(not math.isnan(j[i])):
                filedir = str(int(j[i]/10000))
                filename = str(j[i])
                length = len(filename)
                filename = filename[0:length-2]

                if(os.path.isfile(self.root_dir+filedir+"/"+filename+".json"))|(os.path.isfile(self.root_dir+"data"+"/"+filename+".json")):
                  if(os.path.isfile(self.root_dir+filedir+"/"+filename+".json")):
                    f = open(self.root_dir+filedir+"/"+filename+".json")
                  else:
                    f = open(self.root_dir+"data/"+filename+".json")
                try:
                    data = json.load(f)
                except:
                    continue
                self.dataArray.append([])
                self.dataArray[k].append([])
                self.dataArray[k].append([])
                self.dataArray[k].append([])
                for i in range(0,len(data),2):
                      x = data[i].get("rotationRate")
#                       print(i)
                      self.dataArray[k][0].append(x["x"])
                      self.dataArray[k][1].append(x["y"])
                      self.dataArray[k][2].append(x["z"])
            
                stdev = np.std(np.asarray(self.dataArray[k]))
                mean = np.mean(np.asarray(self.dataArray[k]))
                self.dataArray[k] = ((np.asarray(self.dataArray[k])-mean)/stdev).tolist()
                        
                self.dataArray[k][0] = correct_batch(self.dataArray[k][0])
                self.dataArray[k][1] = correct_batch(self.dataArray[k][1])
                self.dataArray[k][2] = correct_batch(self.dataArray[k][2])
                
                
                

                if(label):
                  self.resultArray.append(1)
                else:
                  self.resultArray.append(0)
                

                k = k + 1



        self.dataArray = np.asarray(self.dataArray)
        unique, counts = np.unique(np.array(self.resultArray), return_counts=True)
        print(dict(zip(unique, counts)))



    def __len__(self):
        return len(self.resultArray)

    def __getitem__(self, idx):
        sample = {'data': self.dataArray[idx], 'result': self.resultArray[idx]}

        return sample

def trans_equal(ten,length=1000):
    return torch.tensor([pad_zero(ten[0][0].cpu().numpy(),length),pad_zero(ten[0][1].cpu().numpy(),length),pad_zero(ten[0][2].cpu().numpy(),length)])

def pad_zero(arr,length):
    while(len(arr)<length):
        arr = np.append(arr,0)
    return arr[:length]


# %%
# ~~~~ Get a dataset ~~~~

dataloader = torch.load("/home/anasa2/originalParkinsonsDataloaders/val_loader.pth")
itr = iter(dataloader)
batch = itr.next()
X = batch["data"]
y = batch["result"]
while(True):
    try: 
        batch = next(itr) 
    except:
        break 

    X  = torch.cat((X, batch["data"]), dim=0)
    y  = torch.cat((y, batch["result"]), dim=0)

X = torch.FloatTensor(X.cpu().float())

print(X.shape)

# %% 
# ~~~~ Model for Feature Exraction ~~~~

from parkinsonsNet import Network
model = torch.load("/home/anasa2/pre_trained/parkinsonsNet-rest_mpower-rest.pth",map_location='cpu')

# %%
gpu = 1
device = torch.device('cuda:' + str(gpu))
model.to(device)

counter = 0
modules = {}
for top, module in model.named_children():
    if type(module) == torch.nn.Sequential:
            modules.update({"" + top: module})
            counter += 1

print(f"Total Interpretable layers: {counter}")

layer_map = {0: X}
index = 1
output = X.to(device)

for i in modules:
    layer = modules[i]
    layer.eval()#.to(device)
    output = layer(output)
    # print(output.shape)
    layer_map.update({index: output})
    index += 1

# %%
from mpower_layer_purpose import find_perodic, classify_frequency, get_peaks, classify_amplitude
import collections

# ~~ Generate Attribution Dataset ~~

df = collections.defaultdict(dict)

for index, signal in enumerate(X):
    # print(signal.shape)
    signal = torch.sum(signal, dim=1).numpy()
    # print(np.max(signal))
    df[index] = {'peaks':None, 'amplitude':None}
    # print(classify_amplitude(signal,100))
    df[index]['peaks'] = get_peaks(signal,2)
    df[index]['amplitude'] = classify_amplitude(signal,100)
    
# df = pd.DataFrame(df)

# %%
from tqdm.notebook import tqdm

attribution = Attribution()

for layer in tqdm(range(1,len(layer_map))):
    for index, signal in tqdm(enumerate(layer_map[layer])):
        for i, filter in enumerate(signal):
            for j, unit in enumerate(filter):
                if unit.item() >= 0.9:
                    # print(layer)
                    # print(i)
                    # print("here")
                    attribution.update_unit(layer,i*j+j, df[index])
                    
# %%
attribution.visualize_model(2)

# %%
