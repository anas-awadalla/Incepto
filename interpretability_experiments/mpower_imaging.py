
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

print(X.shape)

# %% 
# ~~~~ Model for Feature Exraction ~~~~

from parkinsonsNet import Network
model = torch.load("/home/anasa2/pre_trained/parkinsonsNet-rest_mpower-rest.pth",map_location='cpu')

# %%
# gpu = 0
# device = torch.device('cuda:' + str(gpu))
from captum.attr import LayerActivation,LayerConductance, LayerGradientXActivation, NeuronConductance

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
# attr_algo = LayerConductance(model, modules["conv8"])
# attributions = torch.flatten(attr_algo.attribute(X[:100],attribute_to_layer_input=False),start_dim=1)
# %%
# ~~~~ Create Clustering Dataset ~~~~
from sklearn.decomposition import PCA

dataset = layer_map[len(layer_map)-1].cpu().detach().numpy()
pca = PCA(n_components=3)
pca_result = pca.fit_transform(dataset)
dataset = pca_result
#%%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sil = []

kmeans = None
curr_score = float("inf")
num_clusters = 0
# ~~~~ K-Means Clustering ~~~~
for k in range(2,15):
    test_kmeans = KMeans(n_clusters=k,verbose=0,random_state=42).fit(dataset)
    labels = test_kmeans.labels_
    score = silhouette_score(dataset, labels, metric = 'euclidean')
    if score < curr_score:
        curr_score = score
        kmeans = test_kmeans
        num_clusters = k
    sil.append(score)
 

# Plot silhouette scores and determine best k

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.plot(sil)
   
print("Optimal K Value is ",num_clusters)
print("Silhouette Score ",curr_score)

# %%
from scipy import cluster, signal

codebook = kmeans.cluster_centers_

print("Centroids:")
print(codebook)

# Use the codebook to assign each observation to a cluster via vector quantization
labels, __ = cluster.vq.vq(dataset, codebook)

# %%
df = pd.DataFrame({'X':[],'Y':[],'Z':[],'label':[]}) 

for label in zip(range(num_clusters)):
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
################# Explore Codebook Summation and Centroids #################

i = 0
channel_labels = {'x','y','z'}
fig = plt.figure(figsize=(40, 40))

for c in enumerate(codebook):
        color_index = 0
        colors = list({'r','g','b'})
        ax = fig.add_subplot(8, 8, i + 1)
                
        points = np.asarray(pca_result)
        deltas = points - c[1]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        centroid = np.argmin(dist_2)
        
        code = c[0]
        
        # print(centroid)

        # print(y[centroid].item())
        ax.set_ylim((-15,15))

        ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))
        
        # print(torch.sum(X[centroid],dim=0).shape)

        for channel, label in zip(torch.sum(X[centroid],dim=0).unsqueeze(0), channel_labels):
            ax.plot(channel, color=colors[2], label="Motion")
            color_index += 1
                
        plt.legend()
        i += 1

# %%

### Plot plot_signal_attribution_timeseries_sum of Signals ###

from ipywidgets import interact_manual, widgets

def plot_signal_attribution_timeseries_sum(point, code, unit):
    point = dataset[labels==code][point]
    
    color_index = 0
    colors = list({'r','g','b'})
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(8, 8, i + 1)
                    
    points = np.asarray(pca_result)
    deltas = points - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    centroid = np.argmin(dist_2)
            
    print(y[centroid].item())

    ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))

    for channel, label in zip(torch.sum(X[centroid],dim=0).unsqueeze(0), channel_labels):
        ax.plot(channel, color=colors[2], label="Motion")
        color_index += 1
        
    plt.legend()

    input = X[centroid]
    
    attr_algo = LayerConductance(model, modules["conv8"])
    attributions = attr_algo.attribute(input.unsqueeze(0).cuda(),attribute_to_layer_input=False)
    
    # print(attributions.shape)

    m = torch.nn.functional.upsample(attributions,size=4000, mode='linear', align_corners=True)
    # print(m)

    attributions = (m[0][unit].cpu().detach().numpy()>0.001)
    
    # print(attributions)
        
    s, e = get_window(attributions)
    
     
    ax = fig.add_subplot(211)
 
    ax.plot(sum(input))
    
    # powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(torch.sum(X[centroid],dim=0))

    # ax.set_xlabel('Time')

    # ax.set_ylabel('Frequency')
    
    rect = Rectangle((s, -30), e-s, 60, color ='red') 
    
    ax.add_patch(rect)
       
    # plt.ylim((-15,15))
   
    plt.show()
                

interact_manual(plot_signal_attribution_timeseries_sum,point=widgets.IntSlider(min=0, max=500, step=1, value=0), code= widgets.IntSlider(min=0, max=10, step=1, value=0), unit= widgets.IntSlider(min=0, max=255, step=1, value=0));

# %%
### Plot plot_signal_attribution_spec_sum of Signals ###

from ipywidgets import interact_manual, widgets

def plot_signal_attribution_spec_sum(point, code, unit):
    point = dataset[labels==code][point]
    
    color_index = 0
    colors = list({'r','g','b'})
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(8, 8, i + 1)
                    
    points = np.asarray(pca_result)
    deltas = points - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    centroid = np.argmin(dist_2)
            
    print(y[centroid].item())

    ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))

    for channel, label in zip(torch.sum(X[centroid],dim=0).unsqueeze(0), channel_labels):
        ax.plot(channel, color=colors[2], label="Motion")
        color_index += 1
        
    plt.legend()

    input = X[centroid]
    
    attr_algo = LayerConductance(model, modules["conv8"])
    attributions = attr_algo.attribute(input.unsqueeze(0).cuda(),attribute_to_layer_input=False)
    
    # print(attributions.shape)

    m = torch.nn.functional.upsample(attributions,size=4000, mode='linear', align_corners=True)
    # print(m)

    attributions = (m[0][unit].cpu().detach().numpy()>0.001)
    
    # print(attributions)
        
    s, e = get_window(attributions)
    
     
    ax = fig.add_subplot(211)
 
    # ax.plot(sum(input))
    
    powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(torch.sum(X[centroid],dim=0))

    ax.set_xlabel('Time')

    ax.set_ylabel('Frequency')
    
    rect = Rectangle((s, -30), e-s, 60, color ='red',fc=(1,0,0,0.2), ec=(0,0,0,1)) 
    
    ax.add_patch(rect)
       
    # plt.ylim((-15,15))
   
    plt.show()
                

interact_manual(plot_signal_attribution_spec_sum,point=widgets.IntSlider(min=0, max=500, step=1, value=0), code= widgets.IntSlider(min=0, max=10, step=1, value=0), unit= widgets.IntSlider(min=0, max=255, step=1, value=0));

# %%
################# Explore Codebook Mean and Centroids #################

i = 0
channel_labels = {'x','y','z'}
fig = plt.figure(figsize=(40, 40))

for c in enumerate(codebook):
        color_index = 0
        colors = list({'r','g','b'})
        ax = fig.add_subplot(8, 8, i + 1) 
        points = np.asarray(pca_result)
        deltas = points - c[1]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        centroid = np.argmin(dist_2)
        
        code = c[0]
        
        # print(centroid)

        # print(y[centroid].item())

        ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))
        ax.set_ylim((-15,15))

        # print(torch.sum(X[centroid],dim=0).shape)

        for channel, label in zip(torch.mean(X[centroid],dim=0).unsqueeze(0), channel_labels):
            ax.plot(channel, color=colors[2], label="Motion")
            color_index += 1
                
        plt.legend()
        i += 1

# %%
### Plot Average of Signals ###

from ipywidgets import interact_manual, widgets

def plot_signal_mean(point, code):
    point = dataset[labels==code][point]
    
    color_index = 0
    colors = list({'r','g','b'})
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(8, 8, i + 1)
                    
    points = np.asarray(pca_result)
    deltas = points - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    centroid = np.argmin(dist_2)
            
    print(y[centroid].item())

    ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))
    ax.set_ylim((-15,15))

    for channel, label in zip(torch.mean(X[centroid],dim=0).unsqueeze(0), channel_labels):
        ax.plot(channel, color=colors[color_index], label=label)
        color_index += 1
                    
    plt.legend()


interact_manual(plot_signal_mean,point=widgets.IntSlider(min=0, max=100, step=1, value=0), code= widgets.IntSlider(min=0, max=num_clusters-1, step=1, value=0));





# %%
################# Explore Codebook Spectogram Sum and Centroids #################
from scipy import signal
from matplotlib import pyplot


i = 0
channel_labels = {'x','y','z'}
fig = plt.figure(figsize=(40, 40))

for c in enumerate(codebook):
        color_index = 0
        colors = list({'r','g','b'})
        ax = fig.add_subplot(8, 8, i + 1) 
        points = np.asarray(pca_result)
        deltas = points - c[1]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        centroid = np.argmin(dist_2)
        
        code = c[0]
  

        ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))
        

        powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(torch.sum(X[centroid],dim=0),250)
        ax.set_xlabel('Time')

        ax.set_ylabel('Frequency')
                
        i += 1

# %%
### Plot Sum Spectogram of Signals ###

from ipywidgets import interact_manual, widgets

def plot_signal_spectogram_sum(point, code):
    point = dataset[labels==code][point]
    
    color_index = 0
    colors = list({'r','g','b'})
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(8, 8, i + 1)
                    
    points = np.asarray(pca_result)
    deltas = points - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    centroid = np.argmin(dist_2)
            
    ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))
    
    powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(torch.sum(X[centroid],dim=0))

    ax.set_xlabel('Time')

    ax.set_ylabel('Frequency')


interact_manual(plot_signal_spectogram_sum,point=widgets.IntSlider(min=0, max=100, step=1, value=0), code= widgets.IntSlider(min=0, max=num_clusters-1, step=1, value=0));

# %%
print(modules.keys())

# %%
def get_window(arr):
    start_index = -1
    end_index = 0
    index = 0
    while index < len(arr):
        if arr[index] and start_index==-1:
            start_index = index
        
        if arr[index]:
            end_index = index
            
        index+=1
        
    return start_index, end_index
    
# %%

### Neuron Conductance if Layer Conductance is promising
### Run Clustering over vectors of neurons ?

from operator import itemgetter
from captum.attr import LayerActivation,LayerConductance, LayerGradientXActivation, NeuronConductance
from tqdm.notebook import tqdm
from matplotlib.patches import Rectangle 


model = model.cuda()

activations = {}
count = 0
attribution_algos = []

for layer in modules:
    attribution_algos.append(LayerConductance(model, modules[layer]))
  
# for idx in tqdm(range(len(X))):
#     count+=1
#     input = X[idx]
        
#     attributions = attr_algo.attribute(input.unsqueeze(0).cuda(),attribute_to_layer_input=True)
#     print(attributions[0].shape)

fig = plt.figure()

for attr_algo in attribution_algos:
    input = X[100]
    attributions = attr_algo.attribute(input.unsqueeze(0).cuda(),attribute_to_layer_input=True)
    attributions = (sum(attributions[0][0]).unsqueeze(0).cpu().detach().numpy()>0.75)
    
    
    # ax = sns.heatmap(attributions)
    
    s, e = get_window(attributions[0])
     
    ax = fig.add_subplot(111)
 
    ax.plot(torch.sum(input,dim=0))
    
    rect = Rectangle((s, -30), e-s, 60, color ='yellow') 
    
    ax.add_patch(rect)
       
    plt.ylim((-15,15))
   
    plt.show()
    break



 # %%
# import torch.nn.utils.prune as prune

# for name, module in model.named_modules():
#     # prune 20% of connections in all 1D-conv layers
#     if isinstance(module, torch.nn.Conv1d):
#             prune.l1_unstructured(module, name='weight', amount=0.5)
#     elif isinstance(module, torch.nn.Sequential):
#         for layer in module:
#             if isinstance(module, torch.nn.Conv1d):
#                 prune.l1_unstructured(module, name='weight', amount=0.5)

# # %%
# bit_mask = dict(model.named_buffers())['conv8.0.weight_mask']  # to verify that all masks exist

# print(bit_mask.shape)


# %%
# model = model.cuda()
class neuron:
    def __init__(self,signal,activations):
        self.signal = signal
        self.activations = activations
        
    def __lt__(self, other):
        return torch.sum(self.activations,dim=0) < torch.sum(other.activations,dim=0)

import heapq
from collections import defaultdict
### Maximum Activations ###
d = defaultdict(list)
attr_algo = LayerActivation(model, modules["conv8"])

for signal in tqdm(X):
    # Get neuron activation
    attributions = attr_algo.attribute(signal.unsqueeze(0).cuda(),attribute_to_layer_input=False)
    # print(attributions.shape)
    for i, act in enumerate(attributions[0]):
        heapq.heappush(d[i],neuron(signal,act))
        if len(d[i]) > 5:
            d[i].pop(-1)
    


# %%
from ipywidgets import interact_manual, widgets

def get_top_activations_spectogram_for_unit(channel):
    # for u in d[]:
    place = 1
        # plt.figure(place)
    # if len(d[u]) > 0:
    for i in d[channel]:
        ax = plt.subplot(5,5,place)
        powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(torch.mean(i.signal,dim=0))
                # title = "Map "+u+ " Unit "+i
                # ax.set_title(title)
                # ax.plot(torch.sum(i.signal,dim=0))
        place += 1
        plt.show()
        

interact_manual(get_top_activations_spectogram_for_unit,channel = widgets.IntSlider(min=1, max=256, step=1, value=1))
# %%
from ipywidgets import interact_manual, widgets

def get_top_activations_timeseries_for_unit(channel):
    # for u in d[]:
    place = 1
        # plt.figure(place)
    # if len(d[u]) > 0:
    for i in d[channel]:
        ax = plt.subplot(5,5,place)
        ax.plot(torch.mean(i.signal,dim=0))
                # title = "Map "+u+ " Unit "+i
                # ax.set_title(title)
                # ax.plot(torch.sum(i.signal,dim=0))
        place += 1
        plt.show()
        
interact_manual(get_top_activations_timeseries_for_unit,channel = widgets.IntSlider(min=1, max=256, step=1, value=1))

# %%
