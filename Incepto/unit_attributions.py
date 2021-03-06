# -*- coding: utf-8 -*-
"""Unit Inspections.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v1KIVK_UAvVpivXx5S3nWpgFfMGaWzlC
"""

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# !(stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1) && exit
# pip install ninja 2>> install.log
# git clone https://github.com/SIDN-IAP/global-model-repr.git tutorial_code 2>> install.log

!git clone https://github.com/davidbau/dissect.git

!pip install captum

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dissect.netdissect import nethook, imgviz, show, segmenter, renormalize, upsample, tally, pbar
torch.cuda.set_device(0)

# Pretrained model
model = torchvision.models.densenet121()
model.classifier = nn.Linear(1024, 7)
d = torch.load("/content/drive/My Drive/Interpretability Experiments/kaggle90-fc.pt")
model.load_state_dict(state_dict=d)

transform=transforms.Compose([
            transforms.Resize((224 ,224)),
            transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

from matplotlib.pyplot import imshow
from PIL import Image
image_path = "/content/2-original.jpg" #@param {type: "string"}
ood_img = Image.open(image_path)

ood_tensor_img = transform(ood_img)

model.cpu().eval()
output = model(ood_tensor_img.unsqueeze(0))
print(torch.softmax(output,dim = 1))
print("Output: ", torch.argmax(output).item())
imshow(ood_img)

# counter to keep count of the conv layers
counter = 0
modules = {}
for top,module in model.features.named_children():
    for block,m1 in module.named_children():
      for layer,m in m1.named_children():
        if type(m) == nn.Conv2d:
            modules.update({"features."+top+"."+block+"."+layer:m})
            counter += 1

print(f"Total convolutional layers: {counter}")

from operator import itemgetter

def get_topk_units(model, k, input):
  activations = {}
  for layer in modules:
    attr_algo = LayerActivation(model, modules[layer])
    attributions = attr_algo.attribute(input.unsqueeze(0).cuda())[0]
    for i, unit in enumerate(attributions):
      activations.update({(layer,i):sum(sum(unit)).item()})

  i = 0

  results = []
  for (key, value), top in zip(sorted(activations.items(), key = itemgetter(1), reverse = True),range(5)):
        results.append((key, value))


  return results


layers = get_topk_units(model,5,ood_tensor_img)

print(layers)

model = nethook.InstrumentedModel(model)
model.eval()

import pandas as pd
import glob
import os

#make a dataset
data = pd.read_csv("/content/drive/My Drive/mHealth Privacy/Evaluating Models/Data/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")

print(data)
base_skin_dir = os.path.join('/content/drive/My Drive/mHealth Privacy/Evaluating Models/Data/skin-cancer-mnist-ham10000/HAM10000_images_total')


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

data['cell_type'] = data['dx'].map(lesion_type_dict.get)
data['cell_type_idx'] = pd.Categorical(data['cell_type']).codes

data[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

data['cell_type'].value_counts()

from PIL import Image
class dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, transform=None):
        'Initialization'
        self.df = df
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = Image.open("/content/drive/My Drive/mHealth Privacy/Evaluating Models/Data/skin-cancer-mnist-ham10000/HAM10000_images_total/"+self.df['image_id'][index]+".jpg")
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

ham_data = dataset(data, transform=transform)
dataloader_imgs = DataLoader(ham_data, batch_size=100,shuffle=True)

torch.cuda.empty_cache()

itr = iter(dataloader_imgs)

try:
  batch = next(itr)
except:
  batch = next(itr)

iv = imgviz.ImageVisualizer((256, 256), source=ham_data, percent_level=0.99)

model.cuda();

try:
  os.system("rm -rf results")
except:
  None

sample_size = 100
layername = layers[0][0][0]
model.retain_layer(layername)
data, y = batch
def max_activations(batch, *args):
    image_batch = data.cuda()
    _ = model(image_batch)
    acts = model.retained_layer(layername)
    return acts.view(acts.shape[:2] + (-1,)).max(2)[0]

def mean_activations(batch, *args):
    image_batch = data.cuda()
    _ = model(image_batch)
    acts = model.retained_layer(layername)
    return acts.view(acts.shape[:2] + (-1,)).mean(2)

topk = tally.tally_topk(
    max_activations,
    dataset=ham_data,
    sample_size=sample_size,
    batch_size=100,
    cachefile='results/cache_mean_topk.npz'
)

top_indexes = topk.result()[1]

print(model.retained_layer(layername).shape)

show.blocks([
      ['unit %d' % u,
       layername ,
      'img %d' % i,
      'pred: %s' % batch[1][i].item(),
      [iv.masked_image(
          batch[0][i],
          model.retained_layer(layername)[i],
          u,level=0)]
      ]
    for u in [layers[4][0][1]]
    for i in top_indexes[u, :10]
])

model.stop_retaining_layers([layername])

model.retain_layer(layername)

output = model(ood_tensor_img.unsqueeze(0).cuda());

show.blocks([
      ['unit %d' % u,
       layername ,
      'pred: %s' % torch.argmax(output).item(),
      [iv.masked_image(
          ood_tensor_img,
          model.retained_layer(layername)[0],
          u,level=0)]
      ]
    for u in [77]
    # for i in top_indexes[u, :8]
])
