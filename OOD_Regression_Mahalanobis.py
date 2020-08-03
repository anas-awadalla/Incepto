"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse

from sklearn.linear_model import LogisticRegressionCV

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--net_type', required=True, help='resnet | densenet | parkinsonsNet')
args = parser.parse_args()
print(args)

def main():
    # initial setup
    dataset_list = ['cifar10', 'cifar100', 'svhn']

    if args.net_type == 'densenet121':
        dataset_list = ['ham10000']
    elif args.net_type == "parkinsonsNet-rest":
        dataset_list = ['mpower-rest'] 
    elif args.net_type == "parkinsonsNet-return":
        dataset_list = ['mpower-return']
    elif args.net_type == "parkinsonsNet-outbound":
        dataset_list = ['mpower-outbound']   
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    
    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('In-distribution: ', dataset)
        outf = '/home/anasa2/deep_Mahalanobis_detector/output/' + args.net_type + '_' + dataset + '/'
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        if dataset == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        elif dataset == 'ham10000':
            #out_list = ['cifar10', 'cifar100', 'svhn', 'imagenet_resize', 'lsun_resize', 'face', 'face_age', 'isic-2017', 'isic-2016'] #face #face_age
            out_list = ['ham10000-avg-smoothing','ham10000-brightness','ham10000-contrast','ham10000-dilation','ham10000-erosion','ham10000-med-smoothing','ham10000-rotation','ham10000-shift'] #face #face_age
        elif dataset == 'mpower-rest':
            out_list = ['mHealth','MotionSense','oodParkinsonsData','mpower-rest']
        elif dataset == 'mpower-return':
            out_list = ['mHealth','MotionSense','oodParkinsonsData','mpower-return']
        elif dataset == 'mpower-outbound':
            out_list = ['mHealth','MotionSense','oodParkinsonsData','mpower-outbound']

        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            print('Out-of-distribution: ', out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                #X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out)
                #X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                X_val, Y_val, X_test, Y_test = lib_regression.stratified_split(total_X, total_Y)
                #X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
                #Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
                #X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
                #Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
                #partition = int(len(X_val)/2)
                #X_train = np.concatenate((X_val[:partition], X_val[:partition]))
                #Y_train = np.concatenate((Y_val[:partition], Y_val[:partition]))
                #X_val_for_test = np.concatenate((X_val[partition:], X_val[partition:]))
                #Y_val_for_test = np.concatenate((Y_val[partition:], Y_val[partition:]))
                X_train, Y_train, X_val_for_test, Y_val_for_test = lib_regression.stratified_split(X_val, Y_val, test_size=0.5)
                lr = LogisticRegressionCV(n_jobs=-1, max_iter=1000).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_tnr < results['TMP']['TNR']:
                    best_tnr = results['TMP']['TNR']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
        
    # print the results
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        if dataset_list[count_in] == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        elif dataset == 'ham10000':
            #out_list = ['cifar10', 'cifar100', 'svhn', 'imagenet_resize', 'lsun_resize', 'face', 'face_age', 'isic-2017', 'isic-2016']
            out_list = ['ham10000-avg-smoothing','ham10000-brightness','ham10000-contrast','ham10000-dilation','ham10000-erosion','ham10000-med-smoothing','ham10000-rotation','ham10000-shift']
        elif dataset == 'mpower-rest':
            out_list = ['mHealth','MotionSense','oodParkinsonsData','MotionSense','mpower-rest']
            
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1

if __name__ == '__main__':
    main()