# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 09:52:07 2018

@author: Emanuele
"""

import numpy as np
import pandas as pd
import sklearn.utils as skutils


"""
 Takes as input:
    filename:string, path to the .csv file. The header must not be present. Each entry
              should be a row of m entries;
    split:string, 3 split methods are possible and need do be specified in 'split' variable:                     
            'train': all data is reserved to train;
            'train-test': split between train and test, according to 'train_percentage';
            'validation': data is split among train, test and validation: 
                          their percentage is chosen according to the percantge specified
                          in 'train_percentage' (assigned to train), 'validation_percentage'
                          (assigned to validation) and the percentage that is left to test.;
    train_percentage:float, number between 0 and 1 (included) that specifies the size of
                      the dataset reserved to train purpose. It is ignored if 'split' is set
                      to 'train' (i.e. all the dataset is reserved to train);
    validation_percentage:float, number between 0 and 1 (included) that specifies the size of
                           the dataset reserved to validation purpose. It is ignored if 'split' is 
                           set to 'train-test' (i.e. no validation is performed);
    normalize:boolean, if True, normalize data with min-max method (i.e. subtract minimum and
               divide by (max-min));
    time-difference:boolean, if True, each datapoint is subtracted with the previous one to
                     compose the next input (i.e. new_data(t) = data(t)-data(t-1)). This means
                     the dataset dimension is diminished by one.
"""
def data_split(filename,
               split='train', 
               train_percentage=.7, 
               validation_percentage=.1,
               shuffle=False,
               normalize=False,
               time_difference=False):
    
    data = pd.read_csv(filename, delimiter=',', header=0)
    data = (data.iloc[:,:]).values

    if normalize is True:
        
        data = (data-np.min(data))/(np.max(data)-np.min(data))
    
    # if the flag is enabled, turn the dataset into the variation of each time 
    #  step with the previous value (loose the firt sample)
    if time_difference is True:
        
        data = data[1:] - data[:-1]
    
    if shuffle is True:
        
        data = skutils.shuffle(data)
    
    n_samples, n_features = data.shape

    if split == 'train':

        x = data[:,:]

        return x

    elif split == 'train-test':

        train_size = int(train_percentage * n_samples)
        x_train = data[:train_size,:]
        x_test = data[train_size:,:]

        return x_train, x_test

    elif split == 'validation':

        train_size = int(train_percentage * n_samples)
        validation_size = int(validation_percentage * n_samples)

        x_train = data[:train_size,:]
        x_validation = data[train_size:train_size+validation_size,:]
        x_test = data[train_size+validation_size:,:]

        return x_train, x_validation, x_test
    