"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import data_loader
import numpy as np
import models
import os
import lib.adversary as adversary
from models.densenet121 import DenseNet121
from models.parkinsonsNet import Network

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

# parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
# parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
# parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
# parser.add_argument('--dataroot', default='./data', help='path to dataset')
# parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
# parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
# parser.add_argument('--net_type', required=True, help='resnet | densenet')
# parser.add_argument('--gpu', type=int, default=0, help='gpu index')
# parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CWL2')
# args = parser.parse_args()
# print(args)

def generate_adv_samples(model, net_type, dataset_name, dataset, gpu, adv_type, num_classes, random_noise, min_pixel, max_pixel, in_transform=None, batch_size=200, verbose=0):
    outf = './adv_output/'
    # set the path to pre-trained model and output
    pre_trained_net = model
    if os.path.isdir(outf) == False:
        os.mkdir(outf)
    outf = outf + net_type + '_' + dataset_name + '/'
    if os.path.isdir(outf) == False:
        os.mkdir(outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(gpu)
     
    if adv_type == 'FGSM':
        adv_noise = 0.05
    elif adv_type == 'BIM':
        adv_noise = 0.01
    elif adv_type == 'DeepFool':
        if net_type == 'resnet':
            if dataset == 'cifar10':
                adv_noise = 0.18
            elif dataset == 'cifar100':
                adv_noise = 0.03
            else:
                adv_noise = 0.1
        else:
            if dataset == 'cifar10':
                adv_noise = 0.6
            elif dataset == 'cifar100':
                adv_noise = 0.1
            else:
                adv_noise = 0.5

    # model= torch.load(pre_trained_net)

    # load networks
    if net_type == 'densenet':
        if dataset == 'svhn':
            model = models.DenseNet3(100, int(num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(gpu)))
        else:
            model = torch.load(pre_trained_net, map_location = "cuda:" + str(gpu))
        in_transform = transforms.Compose([transforms.ToTensor(), \
                                           transforms.Normalize((125.3/255, 123.0/255, 113.9/255), \
                                                                (63.0/255, 62.1/255.0, 66.7/255.0)),])
        min_pixel = -1.98888885975
        max_pixel = 2.12560367584
        if dataset == 'cifar10':
            if adv_type == 'FGSM':
                random_noise_size = 0.21 / 4
            elif adv_type == 'BIM':
                random_noise_size = 0.21 / 4
            elif adv_type == 'DeepFool':
                random_noise_size = 0.13 * 2 / 10
            elif adv_type == 'CWL2':
                random_noise_size = 0.03 / 2
        elif dataset == 'cifar100':
            if adv_type == 'FGSM':
                random_noise_size = 0.21 / 8
            elif adv_type == 'BIM':
                random_noise_size = 0.21 / 8
            elif adv_type == 'DeepFool':
                random_noise_size = 0.13 * 2 / 8
            elif adv_type == 'CWL2':
                random_noise_size = 0.06 / 5
        else:
            if adv_type == 'FGSM':
                random_noise_size = 0.21 / 4
            elif adv_type == 'BIM':
                random_noise_size = 0.21 / 4
            elif adv_type == 'DeepFool':
                random_noise_size = 0.16 * 2 / 5
            elif adv_type == 'CWL2':
                random_noise_size = 0.07 / 2
                
    elif net_type == 'resnet':
        model = models.ResNet34(num_c=num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(), \
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        min_pixel = -2.42906570435
        max_pixel = 2.75373125076
        if dataset == 'cifar10':
            if adv_type == 'FGSM':
                random_noise_size = 0.25 / 4
            elif adv_type == 'BIM':
                random_noise_size = 0.13 / 2
            elif adv_type == 'DeepFool':
                random_noise_size = 0.25 / 4
            elif adv_type == 'CWL2':
                random_noise_size = 0.05 / 2
        elif dataset == 'cifar100':
            if adv_type == 'FGSM':
                random_noise_size = 0.25 / 8
            elif adv_type == 'BIM':
                random_noise_size = 0.13 / 4
            elif adv_type == 'DeepFool':
                random_noise_size = 0.13 / 4
            elif adv_type == 'CWL2':
                random_noise_size = 0.05 / 2
        else:
            if adv_type == 'FGSM':
                random_noise_size = 0.25 / 4
            elif adv_type == 'BIM':
                random_noise_size = 0.13 / 2
            elif adv_type == 'DeepFool':
                random_noise_size = 0.126
            elif adv_type == 'CWL2':
                random_noise_size = 0.05 / 1 
                
    elif net_type == 'densenet121':
        model = DenseNet121(num_classes=num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(gpu)).state_dict())
        in_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.7630069, 0.5456578, 0.5700767), (0.14093237, 0.15263236, 0.17000099))])
        
        min_pixel = -10
        max_pixel = 10
        
        if adv_type == 'FGSM':
            random_noise_size = 0.25 / 4
        elif adv_type == 'BIM':
            random_noise_size = 0.13 / 2
        elif adv_type == 'DeepFool':
            random_noise_size = 0.126
        elif adv_type == 'CWL2':
            random_noise_size = 0.05 / 1 
    else:
        random_noise_size = random_noise
        min_pixel = min_pixel
        max_pixel = max_pixel
        
            
    model.cuda()
    print('load model: ' + net_type)
    
    print('load target data ')#, dataset)

    validation_split = 0.2
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    if in_transform is not None:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    else:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    # # load dataset
    # _, test_loader = data_loader.getTargetDataSet(dataset, batch_size, in_transform, dataroot)
    
    print('Attack: ' + adv_type  +  ', Dist: ' + "dataset" + '\n') #ADD OTHER PARAM
    model.eval()
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0
    
    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss()#.cuda()

    selected_list = []
    selected_index = 0
    for data, target in test_loader:
        data, target = data.type(torch.FloatTensor).cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()

        noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).cuda())
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
            
        # generate adversarial
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        if adv_type == 'FGSM': 
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float()-0.5)*2
            if net_type == 'densenet':
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023)),
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994)),
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        elif adv_type == 'BIM': 
            gradient = torch.sign(inputs.grad.data)
            for k in range(5):
                inputs = torch.add(inputs.data, adv_noise, gradient)
                inputs = torch.clamp(inputs, min_pixel, max_pixel)
                inputs = Variable(inputs, requires_grad=True)
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                gradient = torch.sign(inputs.grad.data)
                if net_type == 'densenet':
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
                else:
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        if adv_type == 'DeepFool':
            _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cpu(), \
                                             num_classes, step_size=adv_noise, train_mode=False)
            adv_data = adv_data.cuda()
        elif adv_type == 'CWL2':
            _, adv_data = adversary.cw(model, data.data.clone(), target.data.cpu(), 1.0, 'l2', crop_frac=1.0)
        else:
            adv_data = torch.add(inputs.data, adv_noise, gradient)
            
        adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
        
        # measure the noise 
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)


        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cpu()
            # for dashboard generation
            viz_adv_data = adv_data_tot
            viz_data = data
            viz_target = target
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)

        output = model(Variable(adv_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
        
        output = model(Variable(noisy_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()
        
        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
            
        total += data.size(0)
    print(adv_data[0][0].shape)


    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (outf, net_type, dataset, adv_type))
    torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (outf, net_type, dataset, adv_type))
    torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (outf, net_type, dataset, adv_type))
    torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (outf, net_type, dataset, adv_type))
    if verbose == 0:
        print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
        print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
        print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
        print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))
    else:
        print("None")
        #build UI Dashboard

