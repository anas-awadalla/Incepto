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
%matplotlib inline
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
# N = 10000
# df_subset = df.loc[rndperm[:N],:].copy()
# data_subset = df_subset[feat_cols].values
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(data_subset)
# df_subset['pca-one'] = pca_result[:,0]
# df_subset['pca-two'] = pca_result[:,1] 
# df_subset['pca-three'] = pca_result[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# # %%
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(data_subset)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# # %%
# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )


# # %%
# plt.figure(figsize=(16,7))
# ax1 = plt.subplot(1, 2, 1)
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax1
# )
# ax2 = plt.subplot(1, 2, 2)
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax2
# )


# # %%
# pca_50 = PCA(n_components=50)
# pca_result_50 = pca_50.fit_transform(data_subset)
# print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))


# # %%
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
# tsne_pca_results = tsne.fit_transform(pca_result_50)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# # %%
# df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
# df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
# plt.figure(figsize=(16,4))
# ax1 = plt.subplot(1, 3, 1)
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax1
# )
# ax2 = plt.subplot(1, 3, 2)
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax2
# )
# ax3 = plt.subplot(1, 3, 3)
# sns.scatterplot(
#     x="tsne-pca50-one", y="tsne-pca50-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax3
# )


# # %%
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# writer.add_figure("pca-one and pca-two",ax1)
# writer.add_figure("tsne-2d",ax2)
# writer.add_figure("tsne-pca50",ax3)


# # %%
# get_ipython().system('tensorboard --logdir=runs')


# # %%



