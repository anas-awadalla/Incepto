from collections.abc import Iterable
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
import param
import pandas as pd 
import numpy as np
import seaborn as sns
import panel as pn
from torchvision import datasets, transforms
from dashboard import DashboardDataElements


# configure seaborn settings
sns.set()
sns.set(style="ticks", color_codes=True)
# sns.set_context("notebook")
sns.set({ "figure.figsize": (12/1.5,8/1.5) })
sns.set_style("whitegrid", {'axes.edgecolor':'gray'})
plt.rcParams['figure.dpi'] = 360

# add a grid to the plots 
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True

pn.extension()


"""
in_distribution: as torch dataset
out_distribution: as a single or list of torch datasets
image: is the data image data
"""

def analyze_adv(adv_data, real_data, targets, channel_labels, data_labels=None):

    dde = DashboardDataElements(name='')
    dashboard_title = '### Adversarial Data Analysis'
    dashboard_desc = 'Debuging NNs Facing Adversarial Data'

    
    if is_image:
        data = [((next(iter(in_distrbution_sample))[0]))]
        for i in samples:
            data.append(((next(iter(i))[0])))
            
        print("Generating Graph Layout...")

        tabs = []
        for i, label in zip(data, data_labels):
            i = sum(i)/len(i)
            i = i.permute(1,2,0).cpu().detach().numpy()
            tabs.append((label,pn.Column(dde.pixel_dist_img(i),
                                         dde.color_dist_img(i), )))
                                         #dde.multi_dem_color_hist(i))))
                
        dashboard = pn.Column(dashboard_title,dashboard_desc, dde.param,  pn.Tabs(*tabs))
        print("Conducting the following comparisions for image data: Color distribution, Pixel distribution, and Variance of laplacian operators")
    
    else:
        print("Conducting the following comparisions for signal data: Min/Max/Mean/StDev, Energy, and Power")
        assert(signal_frequency is not None), "Signal Frequency cannot be None"
        
        data = [((next(iter(in_distrbution_sample))[0]))]
        for i in samples:
            data.append(((next(iter(i))[0])))
            
        print("Generating Graph Layout...")

        tabs = []
        for i, label in zip(data, data_labels):
            i = sum(i)/len(i)
            i = i.cpu().detach().numpy()
            tabs.append((label,pn.Column(dde.plt_energy_spec(i,signal_frequency,channel_labels), 
                                         dde.plt_power_spec(i,signal_frequency,channel_labels), 
                                         dde.mean_plot(i,channel_labels), dde.min_plot(i,channel_labels), 
                                         dde.max_plot(i,channel_labels), dde.stdev_plot(i,channel_labels) )))
                
        dashboard = pn.Column(dashboard_title,dashboard_desc, dde.param,  pn.Tabs(*tabs))

    

    # display the dashboard, with all elements and data embedded so 
    # no 'calls' to a data source are required
    dashboard.show()
    