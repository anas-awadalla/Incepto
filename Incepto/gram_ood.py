from __future__ import division,print_function
import sys
from tqdm import tqdm_notebook as tqdm
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
import calculate_log as callog
import warnings
warnings.filterwarnings('ignore')


def gram_metrics(model, train_in_dist, test_in_dist, ood_data, in_dist_name, ood_data_names, gpu=0, batch_size=128):
    torch.cuda.set_device(gpu)
    torch_model = model
    torch_model.cuda()
    torch_model.params = list(torch_model.parameters())
    torch_model.eval()
    train_loader = torch.utils.data.DataLoader(train_in_dist,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_in_dist,
        batch_size=batch_size, shuffle=True)
    
    data_train = list(torch.utils.data.DataLoader(
        train_in_dist,
        batch_size=1, shuffle=False))
    
    data = list(torch.utils.data.DataLoader(
    test_in_dist,
    batch_size=1, shuffle=False))
    
    ood_loaders = []
    
    for d in ood_data:
        ood_loaders.append(list(d,
        batch_size=1, shuffle=False))
        
    
    torch_model.eval()
    # correct = 0
    # total = 0
    # for x,y in test_loader:
    #     x = x.cuda()
    #     y = y.numpy()
    #     correct += (y==np.argmax(torch_model(x).detach().cpu().numpy(),axis=1)).sum()
    #     total += y.shape[0]
    # print("Accuracy: ",correct/total)
    
    train_preds = []
    train_confs = []
    train_logits = []
    for idx in range(0,len(data_train),128):
        batch = torch.squeeze(torch.stack([x[0] for x in data_train[idx:idx+128]]),dim=1).cuda()
        
        logits = torch_model(batch)
        confs = F.softmax(logits,dim=1).cpu().detach().numpy()
        preds = np.argmax(confs,axis=1)
        logits = (logits.cpu().detach().numpy())

        train_confs.extend(np.max(confs,axis=1))    
        train_preds.extend(preds)
        train_logits.extend(logits)

    test_preds = []
    test_confs = []
    test_logits = []

    for idx in range(0,len(data),128):
        batch = torch.squeeze(torch.stack([x[0] for x in data[idx:idx+128]]),dim=1).cuda()
        
        logits = torch_model(batch)
        confs = F.softmax(logits,dim=1).cpu().detach().numpy()
        preds = np.argmax(confs,axis=1)
        logits = (logits.cpu().detach().numpy())

        test_confs.extend(np.max(confs,axis=1))    
        test_preds.extend(preds)
        test_logits.extend(logits)

    def detect(all_test_deviations,all_ood_deviations, verbose=True, normalize=True):
        average_results = {}
        for i in range(1,11):
            random.seed(i)
            
            validation_indices = random.sample(range(len(all_test_deviations)),int(0.1*len(all_test_deviations)))
            test_indices = sorted(list(set(range(len(all_test_deviations)))-set(validation_indices)))

            validation = all_test_deviations[validation_indices]
            test_deviations = all_test_deviations[test_indices]

            t95 = validation.mean(axis=0)+10**-7
            if not normalize:
                t95 = np.ones_like(t95)
            test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
            ood_deviations = (all_ood_deviations/t95[np.newaxis,:]).sum(axis=1)
            
            results = callog.compute_metric(-test_deviations,-ood_deviations)
            for m in results:
                average_results[m] = average_results.get(m,0)+results[m]
        
        for m in average_results:
            average_results[m] /= i
        if verbose:
            callog.print_results(average_results)
        return average_results


    def cpu(ob):
        for i in range(len(ob)):
            for j in range(len(ob[i])):
                ob[i][j] = ob[i][j].cpu()
        return ob
        
    def cuda(ob):
        for i in range(len(ob)):
            for j in range(len(ob[i])):
                ob[i][j] = ob[i][j].cuda()
        return ob

    class Detector:
        def __init__(self):
            self.all_test_deviations = None
            self.mins = {}
            self.maxs = {}
            self.classes = range(10)
        
        def compute_minmaxs(self,data_train,POWERS=[10]):
            for PRED in tqdm(self.classes):
                train_indices = np.where(np.array(train_preds)==PRED)[0]
                train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]),dim=1)
                mins,maxs = torch_model.get_min_max(train_PRED,power=POWERS)
                self.mins[PRED] = cpu(mins)
                self.maxs[PRED] = cpu(maxs)
                torch.cuda.empty_cache()
        
        def compute_test_deviations(self,POWERS=[10]):
            all_test_deviations = None
            for PRED in tqdm(self.classes):
                test_indices = np.where(np.array(test_preds)==PRED)[0]
                test_PRED = torch.squeeze(torch.stack([data[i][0] for i in test_indices]),dim=1)
                test_confs_PRED = np.array([test_confs[i] for i in test_indices])
                mins = cuda(self.mins[PRED])
                maxs = cuda(self.maxs[PRED])
                test_deviations = torch_model.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis]
                cpu(mins)
                cpu(maxs)
                if all_test_deviations is None:
                    all_test_deviations = test_deviations
                else:
                    all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
                torch.cuda.empty_cache()
            self.all_test_deviations = all_test_deviations
            
        def compute_ood_deviations(self,ood,POWERS=[10]):
            ood_preds = []
            ood_confs = []
            
            for idx in range(0,len(ood),128):
                batch = torch.squeeze(torch.stack([x[0] for x in ood[idx:idx+128]]),dim=1).cuda()
                logits = torch_model(batch)
                confs = F.softmax(logits,dim=1).cpu().detach().numpy()
                preds = np.argmax(confs,axis=1)
                
                ood_confs.extend(np.max(confs,axis=1))
                ood_preds.extend(preds)  
                torch.cuda.empty_cache()
            # print("Done")
            
            all_ood_deviations = None
            for PRED in tqdm(self.classes):
                ood_indices = np.where(np.array(ood_preds)==PRED)[0]
                if len(ood_indices)==0:
                    continue
                ood_PRED = torch.squeeze(torch.stack([ood[i][0] for i in ood_indices]),dim=1)
                ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices])
                mins = cuda(self.mins[PRED])
                maxs = cuda(self.maxs[PRED])
                ood_deviations = torch_model.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)/ood_confs_PRED[:,np.newaxis]
                cpu(self.mins[PRED])
                cpu(self.maxs[PRED])            
                if all_ood_deviations is None:
                    all_ood_deviations = ood_deviations
                else:
                    all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
                torch.cuda.empty_cache()
            average_results = detect(self.all_test_deviations,all_ood_deviations)
            return average_results, self.all_test_deviations, all_ood_deviations
        
        def G_p(ob, p):
            temp = ob.detach()
            
            temp = temp**p
            temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
            temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
            temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
            
            return temp

        detector = Detector()
        detector.compute_minmaxs(data_train,POWERS=range(1,11))

        detector.compute_test_deviations(POWERS=range(1,11))
        
        for name, d in zip(ood_data_names,ood_loaders):
            print(name)
            results = detector.compute_ood_deviations(d,POWERS=range(1,11))
        
        