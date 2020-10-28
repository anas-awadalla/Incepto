from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import torch
import os
from torch.utils.data import Dataset, DataLoader
import itertools

class IllegalDataDeminsionException(Exception):
    pass


class SignalInterpreter(object):
    
    def __init__(self, model, dataset, gpu, channel_labels):
        
        
        data_loader = DataLoader(dataset,batch_size=len(dataset))
        itr = iter(data_loader)
        self.X,self.y = next(itr)
        
        if len(self.X.shape) != 3:
            raise IllegalDataDeminsionException("Expected data to have deminsions 3 but got deminsion ",str(len(self.X.shape)))
        
        if len(self.X.shape) != len(channel_labels):
            raise IllegalDataDeminsionException("Expected channel_labels to contain ",str(len(self.X.shape))," labels but got ",str(len(channel_labels))," labels instead.")
        
        if gpu == "-1":
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:'+str(gpu))

        counter = 0
        modules = {}
        for top, module in model.named_children():
            if type(module) == torch.nn.Sequential:
                modules.update({""+top:module})
                counter+=1
                
        print(f"Total Interpreatable layers: {counter}")

        self.X.to(device);

        layer_map = {0:self.X.flatten(start_dim=1)}
        index = 1
        output = self.X

        for i in modules:
            layer = modules[i]
            layer.eval().cpu()
            output = layer(output)
            layer_map.update({index:output.flatten(start_dim=1)})
            index += 1    

        assert counter == len(layer_map)-1
        print("Intermediate Forward Pass Through "+str(len(layer_map)-1)+" Layers")

        print("Generating PCA...")
        self.pca_layer_result = []

        self.pca = PCA(n_components=3)
        pca_result = self.pca.fit_transform(layer_map[len(layer_map)-1].detach().numpy())
        for i in pca_result[0]:
            self.pca_layer_result.append(pca_result[:,i])

            
        print("Generation Complete")
        
    def interpret_signal(self, signal):
        reduced_signal = self.pca.transform(signal.unsequeeze(0).detach.numpy())
        example_index = self.__closest_point(reduced_signal)
        example_signal = self.X[example_index]
        return self.__plot_signals([signal,example_signal],self.channel_labels)
    
    def __closest_point(self, point, points):
        points = np.asarray(points)
        deltas = points - point
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)
    
    def __plot_signals(self, signals, channel_labels):
        colors = list(mcd.CSS4_COLORS.keys())
        i = 0
        fig = plt.figure(figsize=(40,40))    
        ax = fig.add_subplot(4, 4, i + 1)
        i += 1
        
        color_index = 0
        ax.set_title("Interpreted Signal")
        for channel, label in zip(signals[0], channel_labels):
            ax.plot(channel,color= colors[color_index+20],label=label)
            color_index += 1
        
        ax = fig.add_subplot(4, 4, i + 1)
        i += 1
        
        color_index = 0
        ax.set_title("Example Signal")
        for channel, label in zip(signals[1], channel_labels):
            ax.plot(channel,color= colors[color_index+20],label=label)
            color_index += 1

        plt.legend()
        plt.show()
        return plt
        