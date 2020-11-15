
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


# Check number of nodes assigned to a cluster
# print(np.shape(cluster_one)[0])
# %%
# ~~~~ Visualization ~~~~
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*codebook.T, c='r')

# for label,c in zip(range(num_clusters),['g','b','c','m','y','b','orange','darkgreen','pink','teal']):
#     cluster_c = dataset[labels==label]
#     ax.scatter(*cluster_c.T, c=c, s=3)
#     # ax.scatter(*cluster_two.T, c='g', s=3)
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# plt.show()

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

################# Signal Visualizer #################

# def __plot_signals(signals, output_test, output_training, channel_labels, distance):
#         """

#         Args:
#             signals:
#             channel_labels:

#         Returns:
#             object:

#         """
                
#         colors = list(CSS4_COLORS.keys())
#         i = 0
#         fig = plt.figure(figsize=(40, 40))
#         print(f"Distance Between Signals {distance}")
        
        
#         ax = fig.add_subplot(4, 4, i + 1)
#         i += 1

#         color_index = 0
#         ax.set_title("Interpreted Signal with Output Class "+str(output_test))
#         for channel, label in zip(signals[0], channel_labels):
#             ax.plot(channel, color=colors[color_index + 20], label=label)
#             color_index += 1
            
#         plt.legend()

#         ax = fig.add_subplot(4, 4, i + 1)
#         i += 1

#         color_index = 0
#         ax.set_title("Example Signal with Output Class "+str(output_training))
#         for channel, label in zip(signals[1], channel_labels):
#             ax.plot(channel, color=colors[color_index + 20], label=label)
#             color_index += 1

#         plt.legend()
#         plt.show()
#         return plt





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

        ax.set_title(f"Code: {code}"+ " - Label "+str(y[centroid].item()))
        
        # print(torch.sum(X[centroid],dim=0).shape)

        for channel, label in zip(torch.sum(X[centroid],dim=0).unsqueeze(0), channel_labels):
            ax.plot(channel, color=colors[2], label="Motion")
            color_index += 1
                
        plt.legend()
        i += 1

# %%

### Plot Sum of Signals ###

from ipywidgets import interact_manual, widgets

def plot_signal_sum(point, code):
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


interact_manual(plot_signal_sum,point=widgets.IntSlider(min=0, max=500, step=1, value=0), code= widgets.IntSlider(min=0, max=10, step=1, value=0));


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

    for channel, label in zip(torch.mean(X[centroid],dim=0).unsqueeze(0), channel_labels):
        ax.plot(channel, color=colors[color_index], label=label)
        color_index += 1
                    
    plt.legend()


interact_manual(plot_signal_mean,point=widgets.IntSlider(min=0, max=100, step=1, value=0), code= widgets.IntSlider(min=0, max=10, step=1, value=0));





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


interact_manual(plot_signal_spectogram_sum,point=widgets.IntSlider(min=0, max=100, step=1, value=0), code= widgets.IntSlider(min=0, max=16, step=1, value=0));





# %%
