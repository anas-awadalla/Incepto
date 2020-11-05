# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
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
# torch.cuda.set_device(0)
# device = torch.device('cuda:0')


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

from matplotlib._color_data import CSS4_COLORS
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


class IllegalDataDimensionException(Exception):
    pass


class SignalInterpreter(object):
    """Summary of class here.

        Longer class information....
        Longer class information....

        Attributes:
            likes_spam: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """

    def __init__(self, model, X, y, gpu, channel_labels):
        """

        Args:
            model:
            dataset:
            gpu:
            channel_labels:
        """
        # data_loader = DataLoader(dataset, batch_size=len(dataset))
        # itr = iter(data_loader)
        # self.X, self.y = next(itr)
        
        self.X = X
        self.y = y
        
        self.model = model.eval()

        if len(self.X.shape) != 3:
            raise IllegalDataDimensionException("Expected data to have dimensions 3 but got dimension ",
                                                str(len(self.X.shape)))

        if len(self.X.shape) != len(channel_labels):
            raise IllegalDataDimensionException("Expected channel_labels to contain ", str(len(self.X.shape)),
                                                " labels but got ", str(len(channel_labels)), " labels instead.")

        self.channel_labels = channel_labels

        if gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(gpu))
            
        model.to(self.device)

        counter = 0
        self.modules = {}
        for top, module in model.named_children():
            if type(module) == torch.nn.Sequential:
                self.modules.update({"" + top: module})
                counter += 1

        print(f"Total Interpretable layers: {counter}")

        self.X.to(self.device)

        layer_map = {0: self.X.flatten(start_dim=1)}
        index = 1
        output = self.X

        for i in self.modules:
            layer = self.modules[i]
            layer.eval().to(self.device)
            output = layer(output)
            layer_map.update({index: output.flatten(start_dim=1)})
            index += 1

        assert counter == len(layer_map) - 1
        print("Intermediate Forward Pass Through " + str(len(layer_map) - 1) + " Layers")

        print("Generating PCA...")
        self.pca_layer_result = []

        self.pca = PCA(n_components=3)
        self.pca_result = self.pca.fit_transform(layer_map[len(layer_map) - 1].detach().numpy())
        for i in range(len(channel_labels)):
            self.pca_layer_result.append(self.pca_result[:,i])
        
        print("Generation Complete")

    def interpret_signal(self, signal):
        """

        Args:
            signal:

        Returns:

        """
        
        output = signal.unsqueeze(0).to(self.device)

        for i in self.modules:
            layer = self.modules[i]
            layer.eval().to(self.device)
            output = layer(output)
            
        output = torch.flatten(output,start_dim=1)
        
        reduced_signal = self.pca.transform(output.detach())
        example_index, dist = self.__closest_point(reduced_signal)
        example_signal = self.X[example_index]
        
        return self.__plot_signals([signal, example_signal], int(torch.round(torch.sigmoid(self.model(signal.unsqueeze(0).to(self.device)))).item()), self.y[example_index].item(), self.channel_labels, dist)

    def __closest_point(self, point):
        """

        Args:
            point:
            points:

        Returns:

        """
        points = np.asarray(self.pca_result)
        deltas = points - point
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        index = np.argmin(dist_2)
        return index, dist_2[index]

    def __plot_signals(self, signals, output_test, output_training, channel_labels, distance):
        """

        Args:
            signals:
            channel_labels:

        Returns:
            object:

        """
                
        colors = list(CSS4_COLORS.keys())
        i = 0
        fig = plt.figure(figsize=(40, 40))
        print(f"Distance Between Signals {distance}")
        
        
        ax = fig.add_subplot(4, 4, i + 1)
        i += 1

        color_index = 0
        ax.set_title("Interpreted Signal with Output Class "+str(output_test))
        for channel, label in zip(signals[0], channel_labels):
            ax.plot(channel, color=colors[color_index + 20], label=label)
            color_index += 1
            
        plt.legend()

        ax = fig.add_subplot(4, 4, i + 1)
        i += 1

        color_index = 0
        ax.set_title("Example Signal with Output Class "+str(output_training))
        for channel, label in zip(signals[1], channel_labels):
            ax.plot(channel, color=colors[color_index + 20], label=label)
            color_index += 1

        plt.legend()
        plt.show()
        return plt

# %%

s = SignalInterpreter(model, X, y, -1,["x","y","z"])


# %%
# Sanity Checks
s.interpret_signal(X[20])

s.interpret_signal(X[210])
