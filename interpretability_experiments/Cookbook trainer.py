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
device = torch.device('cpu')


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
# X = torch.FloatTensor(X.cpu().float())


# %%
from parkinsonsNet import Network

model = torch.load("/home/anasa2/pre_trained/parkinsonsNet-rest_mpower-rest.pth")


# %%

import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        # argmin = argmin.cuda() # device here
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)
    
# %%
class multiCookbook(nn.Module):
    def __init__(self,k,dim):
        super(multiCookbook, self).__init__()

        self.cookbook1 = NearestEmbed(k,dim)
        
        
        self.cookbook2 = NearestEmbed(k,dim)
        
        
        self.cookbook3 = NearestEmbed(k,dim)
        
        
    def forward(self, x):
        
        # z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        # emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)

        # a_e = torch.cat([self.cookbook1(x[:,0][:,0:1000].detach())[0],self.cookbook1(x[:,0][:,1000:2000].detach())[0]
        #                ,self.cookbook1(x[:,0][:,2000:3000].detach())[0],self.cookbook1(x[:,0][:,3000:4000].detach())[0]]
        #               , dim=1)
        # b_e = torch.cat([self.cookbook2(x[:,1][:,0:1000].detach())[0],self.cookbook2(x[:,1][:,1000:2000].detach())[0]
        #                ,self.cookbook2(x[:,1][:,2000:3000].detach())[0],self.cookbook2(x[:,1][:,3000:4000].detach())[0]]
        #               , dim=1)
        # c_e = torch.cat([self.cookbook3(x[:,2][:,0:1000].detach())[0],self.cookbook3(x[:,2][:,1000:2000].detach())[0],self.cookbook3(x[:,2][:,2000:3000].detach())[0]
        #                ,self.cookbook3(x[:,2][:,3000:4000].detach())[0]]
        #               , dim=1)
        
        a_e,_ = self.cookbook1(x[:,0].detach())
        b_e,_ = self.cookbook2(x[:,1].detach())
        c_e,_ = self.cookbook3(x[:,2].detach())

        
        emb = torch.transpose(torch.cat([a_e.unsqueeze(0), b_e.unsqueeze(0), c_e.unsqueeze(0)], dim=0),0,1)
              
        # a_q = torch.cat([self.cookbook1.forward(x[:,0][:,0:1000],weight_sg=True)[0],self.cookbook1.forward(x[:,0][:,1000:2000],weight_sg=True)[0], self.cookbook1.forward(x[:,0][:,2000:3000],weight_sg=True)[0]
        #                ,self.cookbook1.forward(x[:,0][:,3000:4000],weight_sg=True)[0]]
        #               , dim=1)
        # b_q = torch.cat([self.cookbook2.forward(x[:,1][:,0:1000],weight_sg=True)[0],self.cookbook2.forward(x[:,1][:,1000:2000],weight_sg=True)[0], self.cookbook2.forward(x[:,1][:,2000:3000],weight_sg=True)[0]
        #                ,self.cookbook2.forward(x[:,1][:,3000:4000],weight_sg=True)[0]]
        #               , dim=1)
        # c_q = torch.cat([self.cookbook3.forward(x[:,2][:,0:1000],weight_sg=True)[0],self.cookbook3.forward(x[:,2][:,1000:2000],weight_sg=True)[0], self.cookbook3.forward(x[:,2][:,2000:3000],weight_sg=True)[0]
        #                ,self.cookbook3.forward(x[:,2][:,3000:4000],weight_sg=True)[0]]
        #               , dim=1)
        
        a_q,_ = self.cookbook1.forward(x[:,0],weight_sg=True)
        b_q,_ = self.cookbook2.forward(x[:,1],weight_sg=True)
        c_q,_ = self.cookbook3.forward(x[:,2],weight_sg=True)


        
        z_q = torch.transpose(torch.cat([a_q.unsqueeze(0), b_q.unsqueeze(0), c_q.unsqueeze(0)], dim=0),0,1)      
                
        # # z_q, _ = emb(z_e, weight_sg=True).view(-1, 4000)
        # emb, _ = emb(z_e.detach()).view(-1, 4000)        
        
        return z_q, emb
        
# %%
from parkinsonsNet import Network

class quantizedModel(nn.Module):
    def __init__(self):
        super(quantizedModel, self).__init__()
        
        self.classifier = torch.load("/home/anasa2/pre_trained/parkinsonsNet-rest_mpower-rest.pth")
        
        # counter = 0
        # self.modules = {}
        # for top, module in self.classifier.named_children():
        #     if type(module) == torch.nn.Sequential:
        #         self.modules.update({""+top:module.})
        #         counter+=1
        
        # print(f"Total convolutional layers: {counter}")
        self.cookbook = NearestEmbed(20,4000)
        
        

        # self.cookbook = multiCookbook(20,4000)

    def forward(self, x):
        emb = self.cookbook(torch.sum(x,dim=1).detach())
        z_q = self.cookbook.forward(torch.sum(x,dim=1),weight_sg=True)
        
        output = z_q
        
        for top, module in self.classifier.named_children():
            if type(module) == torch.nn.Sequential:
                module = module.cpu()
                output = module(output)
            
        return z_q, emb, output
    
# %%
counter = 0
modules = {}
for top, module in model.named_children():
    if type(module) == torch.nn.Sequential:
        modules.update({""+top:module})
        counter+=1
        
print(f"Total convolutional layers: {counter}")

features = []

# itr = iter(dataloader)
# batch = itr.next()
# X = batch["data"]
# output = X

# for batch in tqdm(dataloader):
#         output = torch.FloatTensor(batch["data"].float()).cpu()
#         for l in modules:
#             layer = modules[l]
#             layer.eval().cpu()
#             output = layer(output)
            
#         features.append(output)
# %%
def loss_function(x, recon_x, z_e, emb):
        vq_coef=0.2
        comit_coef=0.4
        
        ce_loss = F.cosine_embedding_loss(recon_x, x,torch.tensor(0.5))
        vq_loss = F.mse_loss(emb, z_e.detach())
        commit_loss = F.mse_loss(z_e, emb.detach())

        return  ce_loss + vq_coef*vq_loss + comit_coef*commit_loss


# %%
qmodel = quantizedModel()

#%%
for param in qmodel.classifier.parameters():
    param.requires_grad = False
    
optimizer = torch.optim.Adam(qmodel.cookbook.parameters(),lr=1e-4)
# loss = torch.nn.CrossEntropyLoss()
# loss = torch.nn.CosineEmbeddingLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=5, gamma=0.5)

# dissimilar = torch.tensor(-1).cuda()

# %%
from tqdm.notebook import tqdm

qmodel = qmodel.train()#.cuda()


for epoch in range(100):
    scheduler.step()
    
    running_feature_loss = 0.0
    running_wave_loss = 0.0
    
    for i,batch in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        X = batch["data"]
        y = batch["result"]
        
        X = torch.FloatTensor(X.float())
        
        normal_output = X
        for l in modules:
            layer = modules[l].cpu()
            # layer.eval()
            normal_output = layer(normal_output)
            
        normal_output = Variable(normal_output, requires_grad=False)
                        
        z_q, emb, result = qmodel(X)
        
        
        feature_loss = loss_function(normal_output,result,X,emb)
            
                
        # feature_loss = loss(result, normal_output,dissimilar)
        
        # wave_loss = loss(X,wave,dissimilar)
        
        running_feature_loss += feature_loss
        # running_wave_loss += wave_loss
        
        total_loss = feature_loss#+wave_loss
        
        total_loss.backward()
        
        optimizer.step()
    # print(normal_output)
    # print(result)
        
    print("Epoch "+str(epoch))
    print("Total Feature Loss: ",str(running_feature_loss.item()))
    # print("Total Wave Loss: ",str(running_wave_loss.item()))
        
        
        

# %%
print(X[0][1])
print(qmodel.cookbook.cookbook2.weight)
print(qmodel.cookbook.cookbook2(X[0][1][3000:].unsqueeze(0).cuda())[0])

# %%
import matplotlib.pyplot as plt

plt.plot(X[0][0].cpu().detach().numpy())
plt.show()

i = qmodel.cookbook.cookbook1(X[3][0].unsqueeze(0))[0]
plt.plot(i.cpu().detach().numpy()[0])
plt.show()

# plt.plot(X[1][2].cpu().detach().numpy())
# plt.show()

# i = qmodel.cookbook.cookbook3(X[1][2].unsqueeze(0).cuda())[0]
# plt.plot(i.cpu().detach().numpy()[0])
# plt.show()

# %%
