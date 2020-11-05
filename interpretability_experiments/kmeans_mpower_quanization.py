# %%
from scipy import cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# %%
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import math
import json
import time
from matplotlib.pylab import plt
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools

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

# %% 
# ~~~~ Model for Feature Exraction ~~~~

from parkinsonsNet import Network

model = torch.load("/home/anasa2/pre_trained/parkinsonsNet-rest_mpower-rest.pth")

# gpu = 0
# device = torch.device('cuda:' + str(gpu))

device = torch.device("cpu")

model.to(device)

counter = 0
modules = {}
for top, module in model.named_children():
    if type(module) == torch.nn.Sequential:
            modules.update({"" + top: module})
            counter += 1

print(f"Total Interpretable layers: {counter}")

layer_map = {0: X.flatten(start_dim=1)}
index = 1
output = X.to(device)

for i in modules:
    layer = modules[i]
    layer.eval().to(device)
    output = layer(output)
    print(output.shape)
    layer_map.update({index: output.flatten(start_dim=1)})
    index += 1
# %%
# ~~~~ Create Clustering Dataset ~~~~
from sklearn.decomposition import PCA

dataset = layer_map[len(layer_map)-1].cpu().detach().numpy()
pca = PCA(n_components=3)
pca_result = pca.fit_transform(dataset)
dataset = pca_result
#%%
from scipy import cluster

# ~~~~ K-Means Clustering ~~~~
codebook, _ = cluster.vq.kmeans(dataset, 10, iter=200)
print("Centroids:")
print(codebook)
# Use the codebook to assign each observation to a cluster via vector quantization
labels, __ = cluster.vq.vq(dataset, codebook)
# Use boolean indexing to extract points in a cluster from the dataset
cluster_one = dataset[labels == 0]
cluster_two = dataset[labels == 1]

# Check number of nodes assigned to a cluster
# print(np.shape(cluster_one)[0])
# %%
# ~~~~ Visualization ~~~~
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*codebook.T, c='r')

for label,c in zip(range(10),['g','b','c','m','y','b','orange','darkgreen','pink','teal']):
    cluster_c = dataset[labels==label]
    ax.scatter(*cluster_c.T, c=c, s=3)
    # ax.scatter(*cluster_two.T, c='g', s=3)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()

# %%

df = pd.DataFrame({'X':[],'Y':[],'Z':[],'label':[]}) 

for label in zip(range(10)):
    new_df = [[],[],[],[]]
    for point in dataset[labels==label]:
        new_df[0].append(point[0])
        new_df[1].append(point[1])
        new_df[2].append(point[2])
        new_df[3].append(label)
        
    df = pd.concat([df,pd.DataFrame({'X':new_df[0],'Y':new_df[1],'Z':new_df[2],'label':new_df[3]})] 
)
    
# %%
print(df)
    
# %%
import plotly
import plotly.graph_objs as go
import plotly.express as px
from ipywidgets import interact, widgets

plotly.offline.init_notebook_mode()
    
# Configure the trace.
trace = px.scatter_3d(df,
         x='X', 
         y='Y',  
         z='Z', 
         color= 'label',
        #  title="Graph for Layer "+str(layer)
     )
    
    
trace.show()

# %%

def correct_batch(batch):
    while len(batch)<4000:
        batch.append(0)
    return batch[:4000]

import os
from tqdm.notebook import tqdm
import numpy as np  
import pandas as pd

import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import math
import json
import time
from matplotlib.pylab import plt
%matplotlib inline 

class data(Dataset):

    def __init__(self, transform=None):
        self.result = []
        self.labels=[]
        k = 0
        for filename in tqdm(os.listdir("/content/drive/My Drive/mHealth Privacy/Evaluating Models - Motion Data/mHealth Data/")):
            self.result.append([])
            self.result[k].append([])
            self.result[k].append([])
            self.result[k].append([])
            
            df = pd.read_csv("/content/drive/My Drive/mHealth Privacy/Evaluating Models - Motion Data/mHealth Data/"+filename, delimiter= '\s+', index_col=False)
            start = True
            eof = False
            for data in df.iterrows():
              data=data[1]
              if(data[23] not in [1,2,3,7,8]):
                 continue
              # if(data[6]!=7):
              #   curr=1
              # else:
              #   curr=0

              # if start:
              #   self.labels.append(curr)
              #   prev = 1
              #   start=False 

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




    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        return self.result[idx]

    
    # %%
    testloader = torch.utils.data.DataLoader(data(), batch_size=1,shuffle=True)
    