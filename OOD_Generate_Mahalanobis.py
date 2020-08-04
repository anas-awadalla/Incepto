"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
import models
from densenet121 import DenseNet121
import parkinsonsNet
import os
import lib_generation
from parkinsonsNet import Network

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

def extract_features(pre_trained_net, in_distribution, in_dist_name, out_dist_list, out_of_distribution, in_transform, gpu, batch_size, num_classes):
    # set the path to pre-trained model and output
    outf = "/output/"
    outf = outf + "model" + '_' + in_dist_name + '/'
    if os.path.isdir(outf) == False:
        os.mkdir(outf)
    
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(gpu)
    
    # load networks
    model= torch.load(pre_trained_net, map_location = "cuda:" + str(gpu))

    model.cuda()
    print('loaded model')
    
    
    # load target dataset
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(in_distribution)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    if in_transform is not None:
        train_loader = torch.utils.data.DataLoader(in_distribution, batch_size=batch_size, shuffle=True, sampler=train_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(in_distribution, batch_size=batch_size, shuffle=True, sampler=valid_sampler, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(in_distribution, batch_size=batch_size, shuffle=True, sampler=train_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(in_distribution, batch_size=batch_size, shuffle=True, sampler=valid_sampler, **kwargs)

    print('loaded target data: ', in_dist_name)
    
    
    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(*(list(next(iter(train_loader))[0].size()))).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    
    
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader, model_name="model")
    
    print('Generate dataloaders...')

    out_test_loaders=[]
    for out_dist in out_of_distribution:
            out_test_loaders.append(torch.utils.data.DataLoader(out_dist, batch_size=batch_size, shuffle=True, **kwargs))
    
    print('get Mahalanobis scores', num_output)
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            print('layer_num', i)
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, num_classes, outf, \
                                                        True, "model", sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            
        for out_test_loader, out_dist in zip(out_test_loaders,out_dist_list):
            print('Out-distribution: ' + out_dist) 
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader, num_classes, outf, \
                                                             False, "model", sample_mean, precision, i, magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), in_dist_name, out_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)