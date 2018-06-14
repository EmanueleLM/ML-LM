# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 21:03:04 2018

@author: Emanuele

PCA algorithm for both numpy and pandas datatypes.

 If the pandas version is employed, it works with both numerical and categorical
  features. In the latter case, the concept of "unalikability" is employed as measure
  of the variance of the data with itself, with the covariance calculated using 
  the mean, when a variable (or both) is categorical.
"""

import numpy as np
import pandas as pd

# PCA with input matrix (i.e. non-labeled data).
#
# Returns the dimensions that caputure the most variance in the data and the indices
#  of the columns that are returned
#
# Two methods can be used: 'best', which returns the k (out of d, where d is the
#  number of dimensions of the data) dimensions with the highest variance, or 
#  'cumulative', that takes all the dimensions that contributes to the total variance
#  up to a predefined value k, which is expressed as a float between 0 and 1.
#
# please note that the data must be in the format (samples, columns)
def pca(data, method="best", k=1):
       
    # covariance, eigenvectors, eigenvalues
    cov_matrix = np.cov(data.T);  # covariance matrix needs to transpose input
    eig_val, eig_vec = np.linalg.eig(cov_matrix);
    
    eig_val = np.abs(eig_val);
    
    # list of indices of the k most important features
    eig_indices = list();
    
    if (method == "best"):
        
        # create a list of (eigenvector, eigenvalue, index) tuples
        eig_list = [(eig_vec[i], eig_val[i], i) for i in range(len(eig_val))];
         
        # order by explained variance (descending)
        eig_list.sort(key=lambda x: x[1], reverse=True);
        
        # take the first k dimensions
        out = list(i[0] for i in eig_list[:k]);
        
        # append the index of the i-th most important feature
        eig_indices = list(i[-1] for i in eig_list[:k]);         
        
    elif (method == "cumulative"):
        
        # create a list of (eigenvector, eigenvalue, index) tuples
        eig_list = [(eig_vec[i], eig_val[i], i) for i in range(len(eig_val))];
               
        # order by descending explained variance
        eig_list.sort(key=lambda x: x[1], reverse=True);
        
        # take the first d dimensions s.t. their explained variance is at least
        #  a fraction k of the total variance
        out = list();
        var_total = np.sum(eig_val)*k;
        var_partial = .0;
               
        for i in range(len(eig_list)):
            
            if var_partial > var_total:
                break;
                
            out.append(eig_list[i][0]);
            var_partial += eig_list[i][1];
            
            # append the index of the i-th most important feature
            eig_indices.append(i);
        
    else:
        
        print("Warning: undefined method ", method, " for PCA analysis.");
        return data, eig_indices;
    
    # convert back to numpy matrix
    out = np.array(out).T;
      
    return out, eig_indices;
    
    

# PCA with categorical data and Pandas DataFrame as input.
#
# The method is very the same as the previous one (pca), except that this one
#  handles categorical attributes and transforms each entry into the the frequency
#  of the class, then calculates the 'unalikability' for each class and uses it 
#  as a measure of categorical variance. Please refer to the README in the official
#  repo for more info about 'unalikability' (and the ref. to the original paper).
#
# Please note that you must specify which columns are categorical: columns are
#  a list of integers, each of those represents the i-th column in the DataFrame
# if get_pamdas is set to True, a DataFrame object is returned
def pca_labeled(data, columns=[], method="best", k=1, get_pandas=False):   
    
    cols = data.columns;
    
    # case columns is not defined: take all 'objects' types as categorical
    if columns != []:
        
        # tranform non-numerical instances to calculate 'unalikability'
        for j in columns:
            
            data.iloc[:,j] = data.map(data.iloc[:,j].value_counts(normalize=True));
    
    else:
        
        for j in data.columns:
                      
            if data[j].dtypes == np.object:
                
                data[j] = data[j].map(data[j].value_counts(normalize=True));
                
    # call pca now that all the data is numerical
    out, k_dims = pca(data.values.astype(float), method=method, k=k);
    
    if get_pandas is True:
        
        out = pd.DataFrame(out, columns=cols[k_dims]);
    
    return out;