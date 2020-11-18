
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
        new_df[3].append(0)
        
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
print(codebook)

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

def purturbate (signal, range):
    # segment = None
    dif = float("-inf")
    result = model(signal.unsqueeze(0))
    result = result[0].item()
    x = 0
    y = range
        
    curr_range_min = 0
    curr_range_max = range
        
    while curr_range_max < 4000:
        new_signal = signal.clone()
        new_signal[:,curr_range_min:curr_range_max] = 0  #zero out
            
        new_result = model(new_signal.unsqueeze(0))
        # print(new_result)    
        new_dif = abs(new_result - result)
        if new_dif > dif:
            # segment = torch.abs(torch.sum())  #signal at range
            dif = new_dif
            x = curr_range_min
            y = curr_range_max
            
        curr_range_min += range
        curr_range_max += range
                
    # print(x)
    # print(y)
    return x, y
                
# %%
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)
        
# %%
import plotly
import plotly.graph_objs as go
import plotly.express as px
from ipywidgets import interact, widgets
from matplotlib.patches import Rectangle 

def analyze_ood(signal, range):
    output = signal.unsqueeze(0)
    for i in modules:
        layer = modules[i]
        layer.eval().to(device)
        output = layer(output)
        
    features = torch.flatten(output,start_dim=1)
    # print(features.shape)
    
    coordinates = pca.transform(features.cpu().detach().numpy())
    
    plotly.offline.init_notebook_mode()
    
    # print(coordinates)
    
    new_df = pd.DataFrame({'X':[coordinates[0][0]],'Y':[coordinates[0][1]]
                           ,'Z':[coordinates[0][2]],'label':[200]}) 
    
    new_df = df.append(new_df)
    
    # Configure the trace.
    trace = px.scatter_3d(new_df,
            x='X', 
            y='Y',  
            z='Z', 
            color= 'label',
    )
        
    trace.show()
    
    fig = plt.figure(figsize=(40, 40))
    
    x1,y1 = purturbate(signal,range)
    
    #print(x1)
    #print(y1)
    
    close = closest_node(coordinates,points)
    
    x2,y2 = purturbate(X[close],range)
    
    # new_points = list(points.copy())
    
    # new_points.pop(close)
    
    # close_2 = closest_node(coordinates,new_points)
    
    # x3,y3 = purturbate(X[close_2],250)
    
    ##
    
    ax = fig.add_subplot(411)
    # print(signal)
    signal = torch.abs(torch.sum(signal, dim =0))
    # print((signal).shape)
    # print(signal)
    ax.plot(signal)

    ax.set_xlabel('Time')

    ax.set_ylabel('Motion')
    
    rect = Rectangle((x1, 0), y1, 20, color ='red',fc=(1,0,0,0.2), ec=(0,0,0,1)) 
    plt.ylim((0,20))

    ax.add_patch(rect)
    
    plt.show()
    
    ##
    fig = plt.figure(figsize=(40, 40))

    ax = fig.add_subplot(411)

    close_signal = torch.abs(torch.sum(X[close], dim =0))
    ax.plot(close_signal)

    ax.set_xlabel('Time')

    ax.set_ylabel('Motion')
    
    rect = Rectangle((x2, 0), y2, 20, color ='red',fc=(1,0,0,0.2), ec=(0,0,0,1)) 
    plt.ylim((0,20))

    ax.add_patch(rect)
    
    plt.show()


    
# %%


analyze_ood(torch.FloatTensor(np.random.rand(3,4000)), 100)   


# %%
