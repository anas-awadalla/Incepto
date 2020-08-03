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

def analyze(in_distribution, out_of_distribution, channel_labels, is_image=False, signal_frequency=None, data_labels=None):
            
    # Check Dimensions
    if not isinstance(out_of_distribution, Iterable):
        out_of_distribution = [out_of_distribution]
        
    if data_labels is None:
        data_labels = ["in_distribution"]
        for i in range(len(out_of_distribution)):
            data_labels.append("out_distribution_"+str(i))
    
    print("Getting sample from datasets...")
    samples = []
    # in_distribution_subset = torch.utils.data.Subset(in_distribution, [0])   
    in_distrbution_sample = torch.utils.data.DataLoader(in_distribution, batch_size=len(in_distribution), num_workers=0, shuffle=True)
    for i in out_of_distribution:
        samples.append(torch.utils.data.DataLoader(i, batch_size=len(i), num_workers=0, shuffle=True))
    print("Checking Data Demensions...")

    in_dist_shape = list(next(iter(in_distrbution_sample))[0].size())
    for i in (samples):
        assert (list(next(iter(i))[0].size())[1:] == in_dist_shape[1:]), ("Dimensions are not consistent, was looking for dimensions: "+str(in_dist_shape))
    print("Dimensions are consistent - Shape: "+str(in_dist_shape[1:]))
    

    dde = DashboardDataElements(name='')
    dashboard_title = '### Out of Distribution Data Analysis'
    dashboard_desc = 'Comparing Different Datasets'

    
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
    