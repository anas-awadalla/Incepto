# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __future__ import print_function
import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns


# %%
import torch
torch.cuda.set_device(0)
device = torch.device('cuda:0')


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
dataloader = torch.load("/home/anasa2/originalParkinsonsDataloaders/val_loader.pth")


# %%
itr = iter(dataloader)


# %%
batch = itr.next()
X = batch["data"]
y = batch["result"]


# %%
while(True):
    try: 
        batch = next(itr) 
    except:
        break 

    X  = torch.cat((X, batch["data"]), dim=0)
    y  = torch.cat((y, batch["result"]), dim=0)


# %%
X = torch.FloatTensor(X.cpu().float())


# %%
print(X.shape)
print(y.shape)


# %%
from parkinsonsNet import Network

model = torch.load("/home/anasa2/pre_trained/parkinsonsNet-rest_mpower-rest.pth")


# %%
counter = 0
modules = {}
for top, module in model.named_children():
    if type(module) == torch.nn.Sequential:
        modules.update({""+top:module})
        counter+=1
        
print(f"Total convolutional layers: {counter}")
# %%
# For reproducability of the results
np.random.seed(42)
X.cuda()
# %%
layer_map = {0:X.flatten(start_dim=1)}
index = 1
output = X

for i in modules:
    layer = modules[i]
    layer.eval().cpu()
    output = layer(output)
    layer_map.update({index:output.flatten(start_dim=1)})
    index += 1    

print("Intermediate Forward Pass Through "+str(len(layer_map)-1)+" Layers")

# %%
pca_layer_result = {}

for i in layer_map:
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(layer_map[i].detach().numpy())
    pca_layer_result.update({i:[]})
    pca_layer_result[i].append(pca_result[:,0])
    pca_layer_result[i].append(pca_result[:,1])
    pca_layer_result[i].append(pca_result[:,2])
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# %%
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df.loc[rndperm,:],
#     legend="full",
#     alpha=0.9
# )
print(pca_layer_result[0])
# %%
%matplotlib widget
from ipywidgets import interact, widgets

def pca_3d(layer=2, vertical_angle = 0, horizontal_angle = 30):
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=pca_layer_result[layer][0], 
        ys=pca_layer_result[layer][1], 
        zs=pca_layer_result[layer][2], 
        c=y, 
        cmap='rainbow'
    )
       
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    
    ax.view_init(vertical_angle,horizontal_angle)
    
    # # rotate the axes and update
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(10)
    plt.show()
    
interact(pca_3d, layer=widgets.IntSlider(min=0, max=8, step=1, value=0), vertical_angle= widgets.IntSlider(min=0, max=360, step=1, value=0), horizontal_angle = widgets.IntSlider(min=0, max=360, step=1, value=30));

# %%

print(X.shape)


# %%

import plotly
import plotly.graph_objs as go
import plotly.express as px



def pca_ploty_3d(layer):
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()
    
    df = pd.DataFrame({'x':np.asarray(pca_layer_result[layer][0]),'y':np.asarray(pca_layer_result[layer][1]),'z':np.asarray(pca_layer_result[layer][2]),'label':y}) 

    # Configure the trace.
    trace = px.scatter_3d(df,
        x='x', 
        y='y',  
        z='z', 
        color= 'label',
        
    )
    
    trace.show()

interact(pca_ploty_3d, layer=widgets.IntSlider(min=0, max=8, step=1, value=0));



# %%
