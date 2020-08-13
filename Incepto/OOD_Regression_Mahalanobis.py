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

# parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
# parser.add_argument('--net_type', required=True, help='resnet | densenet | parkinsonsNet')
# args = parser.parse_args()
# print(args)

def train_detector(dataset_list, out_list):
    net_type = "model"
    outf = "/output/"

    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    
    for dataset in dataset_list:
        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            print('Out-of-distribution: ', out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.stratified_split(total_X, total_Y)
                
                X_train, Y_train, X_val_for_test, Y_val_for_test = lib_regression.stratified_split(X_val, Y_val, test_size=0.5)
                lr = LogisticRegressionCV(n_jobs=-1, max_iter=1000).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]

                y_pred = lr.predict_proba(X_val_for_test)[:, 1]

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

# if __name__ == '__main__':
#     main()