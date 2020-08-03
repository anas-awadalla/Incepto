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

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='/home/anasa2/deep_Mahalanobis_detector/output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet | parkinsonsNet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)

def main():
    # set the path to pre-trained model and output
    pre_trained_net = '/home/anasa2/deep_Mahalanobis_detector/pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
    elif args.dataset == 'svhn':
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
    elif args.dataset == 'ham10000':
        #out_dist_list = ['cifar10', 'imagenet_resize', 'face', 'face_age', 'isic-2017', 'isic-2016']
        #out_dist_list = ['cifar10', 'face', 'face_age', 'isic-2017', 'isic-2016']
        #out_dist_list = ['cifar10', 'cifar100', 'svhn', 'imagenet_resize', 'lsun_resize', 'face', 'face_age', 'isic-2017', 'isic-2016']
        out_dist_list = ['ham10000-avg-smoothing','ham10000-brightness','ham10000-contrast','ham10000-dilation','ham10000-erosion','ham10000-med-smoothing','ham10000-rotation','ham10000-shift']
    elif args.dataset == 'mpower-rest':
        out_dist_list = ['mHealth','MotionSense','oodParkinsonsData',"mpower-rest"]
    elif args.dataset == 'mpower-outbound':
        out_dist_list = ['mHealth','MotionSense','oodParkinsonsData',"mpower-outbound"]
    elif args.dataset == 'mpower-return':
        out_dist_list = ['mHealth','MotionSense','oodParkinsonsData',"mpower-return"]
    # load networks
    if args.net_type == 'densenet':
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        else:
            model = torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    elif args.net_type == 'densenet121':
        model = DenseNet121(num_classes=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)).state_dict())
        in_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.7630069, 0.5456578, 0.5700767), (0.14093237, 0.15263236, 0.17000099))])
    elif args.net_type == 'parkinsonsNet-rest':
        model= torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
        in_transform = None
    elif args.net_type == 'parkinsonsNet-return':
        model= torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
        in_transform = None
    elif args.net_type == 'parkinsonsNet-outbound':
        model= torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
        in_transform = None
    model.cuda()
    print('load model: ' + args.net_type)
    
    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    
    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    if (args.net_type == 'parkinsonsNet-rest') | (args.net_type == 'parkinsonsNet-return') | (args.net_type == 'parkinsonsNet-outbound'):
        temp_x = torch.rand(8,3,4000).cuda()
        temp_x = Variable(temp_x) 
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, args.num_classes, feature_list, train_loader, model_name=args.net_type)
    
    print('Generate dataloaders...')

    out_test_loaders=[]
    for out_dist in out_dist_list:
            out_test_loaders.append(data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot))
    
    print('get Mahalanobis scores', num_output)
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            print('layer_num', i)
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, args.num_classes, args.outf, \
                                                        True, args.net_type, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            
        for out_test_loader, out_dist in zip(out_test_loaders,out_dist_list):
            # out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
            print('Out-distribution: ' + out_dist) 
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader, args.num_classes, args.outf, \
                                                             False, args.net_type, sample_mean, precision, i, magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.dataset , out_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)
    
if __name__ == '__main__':
    main()