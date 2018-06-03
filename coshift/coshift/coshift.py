# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:25:34 2018

@author: Emanuele

Covariance shift calculator

Process train and test sets to check whether the covariance shift problem is
 present. Can tune sensibility of the model and plot ROC curves for each 
 variable.
 
 how the analysis of the covariance shift works for (train, test):
    
    1. Preprocessing: This step involves imputing all missing values and label 
        encoding of all categorical variables;
    2. Creating a random sample of your training and test data separately and 
        adding a new feature origin which has value train or test depending on 
        whether the observation comes from the training dataset or the test dataset;
    3. Now combine these random samples into a single dataset. Note that the 
        shape of both the samples of training and test dataset should be nearly 
        equal, otherwise it can be a case of an unbalanced dataset;
    4. Now create a model taking one feature at a time while having ‘origin’ as
        the target variable on a part of the dataset (say ~75%);
    5. Now predict on the rest part(~25%) of the dataset and eventually calculate
       the value of AUC-ROC;
    6. Two steps are possible: either using the accuracy of the decisor on splitting 
       train and test or the value of AUC-ROC for a particular feature is greater
       than 0.80, we classify that feature as drifting.
        
We use decision trees for binary classification to identify wether the covariance
 shift is beyond the threshold. The random forest's model is taken from sklearn,
 but as soon as it isready my own model on github, that one will be used.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# take train and test set, resize the biggest one to the dimension of the other
#  add a binary feature (0 if the sample is from train, 1 if it is from test)
#  and finally return them shuffled into a single dataset.
# if 'target' is not set to None, discard the column 'target'
def add_objective(train, test, target=None):
    
    # drop target column
    if target is not None:
        
        try:
            print("bisc")
            train.drop(target, 1);
            test.drop(target, 1);
        except:
            print("Dropping ",target, " column resulted in an error");
    
    # sample a random subset of the smallest dataset
    if train.shape[0] >= test.shape[0]:
        train = train.sample(test.shape[0]);
    else:
        test = test.sample(train.shape[0]);
        
    # reset the indexing for both the dataset
    train = train.reset_index(drop=True);        
    test = test.reset_index(drop=True);
        
    num_samples = min([train.shape[0], test.shape[0]]);
    
    # add binary feature 'obj' that identifies train and test samples
    # 1 if the sample is from train, 0 if from test
    train['target'] = pd.Series(np.ones(num_samples));
    test['target'] = pd.Series(np.zeros(num_samples));    
        
    # concatenate train and test and shuffle them 
    try:
        result = pd.concat([train, test]);
        result = result.sample(frac=1).reset_index(drop=True);
    except:
        print("Concatenation failed");
    
    return result;

# calculate the importance that each feature has in discriminating train and test
# this means that given a dataset where the binary feature 'target' is already
#  calculated, we evaluate how good is each feature, together with a decision tree,
#  in splitting them.
# takes as input:
#   train, test: the datasets fro train and test in pandas format (samples, dimensions)
# returns:
#   accuracy of each feature in predicting which sample belongs to train/test
#   tpr/fpr: true positive rate and false positive rate of each predictor (for each feature)
def coshift(train, test):
          
    # create the classifier
    clf = tree.DecisionTreeClassifier();
       
    # # extract target and data from train
    train_target = train['target']; 
    train_data = train.drop('target', axis=1);
    
    # extract target and data from test
    test_target = test['target'];
    test_data = test.drop('target', axis=1);
    
    # accuracy, tpr, fpr of each feature in splitting train and test
    accuracy = pd.Series(np.zeros(train_data.shape[1]), index=train_data.columns);
    tpr = pd.Series(np.zeros(train_data.shape[1]), index=train_data.columns);
    fpr = pd.Series(np.zeros(train_data.shape[1]), index=train_data.columns);
    
    # calculate the denominator for tpr and fpr
    count_true = len(test[test['target'].astype(int)==1]); # divide by (TP+FN)
    count_false = len(test[test['target'].astype(int)==0]); # divide by (FP+TN)
    
    # for each feature, use a decision tree and try to predict 'target' in train     
    for feat in train_data:
        
        clf.fit(train_data[feat].values.reshape(train_data.shape[0], 1), train_target);
           
        # for each feature's classificator, get the accuracy on the test set
        for i in range(test_target.values.shape[0]):
            
            # predict the value          
            prediction = clf.predict(test_data[feat].values[i])[0];
            
            if int(prediction) == int(test_target.iloc[i]):
                accuracy[feat] += 1;
                
            # calculate the tpr, fpr
            # tpr = TP/(TP+FN)
            # fpr = FP/(FP+TN)
            if int(prediction) == 1:
                
                if int(test_target.iloc[i]) == 1:                        
                    tpr[feat] += 1; # TP, true positive

                else:
                    fpr[feat] += 1; # FP, flase positive
        
        # normalize each tpr by its denominator            
        tpr[feat] /= count_true;
        fpr[feat] /= count_false;
            

                
    # divide the total number of correct classifications by the number of samples
    accuracy /= test_target.shape[0];
    
    return accuracy, tpr, fpr;

# create some random data to test the covariance shift
# n_samples is the number of samples, dim is the number of dimensions
def rand_data(n_samples, dim):
    
    data = np.random.rand(n_samples, dim);
    data = np.append(data, np.random.randint(0,2,(n_samples,1)), axis=1);
    df = pd.DataFrame(data, columns=[*('d'+str(i) for i in range(dim)), 'target']);
    
    return df;

# split train and test (this is just a mask to scikit function 'train_test_split')
def split(data, train_percentage=.2):
    
    train, test = train_test_split(data, test_size=train_percentage);
    
    return train, test;

# transform non-numerical data in data that is usabe by the tree
def label_encoding(data):
    
    number = LabelEncoder();
    
    for i in data.columns:
        
        if (data[i].dtype == 'object'):
            data[i] = number.fit_transform(data[i].astype('str'));
            data[i] = data[i].astype('object');
            
    return data;

# substitute missing values: use the 'mode' for categorical, 'mean' for numerical
def missing_values(data):
    
    for i in data.columns:
        
        if data[i].dtype == 'object':
            data[i] = data[i].fillna(data[i].mode().iloc[0]);
            
        if (data[i].dtype == 'int' or data[i].dtype == 'float'):
            data[i] = data[i].fillna(np.mean(data[i]));
    
    return data;

# print ROC curve, given a list of TPR and FPR (all of the same lenght)
def ROC(TPR, FPR, ordered=True):
    
    # order score, TPR and FPR in ascending order
    if ordered is False:
        pass;
    
    # calculate the area under the curve
    roc_auc = metrics.auc(FPR, TPR);
    
    # set title, labels, axis and limit of the axis
    plt.title('Receiver Operating Characteristic');
    plt.plot(FPR, TPR, 'b', label = 'AUC = %0.2f' % roc_auc);
    plt.legend(loc = 'lower right');
    plt.plot([0, 1], [0, 1],'r--');
    plt.xlim([0, 1]);
    plt.ylim([0, 1]);
    plt.ylabel('True Positive Rate');
    plt.xlabel('False Positive Rate');
    
    # show the ROC
    plt.show();
    