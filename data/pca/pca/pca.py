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

# PCA with input matrix (i.e. non-labeled data)
# returns the dimensions that caputure the most variance
# two methods are possible: 'k-most', which returns the k (out of d, where d is the
#  number of dimensions of the data) dimensions with the highest variance, or 
#  'percentage', that takes all the dimensions that contributes to the total variance
#  up to a predefined value k, which is expressed as a float between 0 and 1. 
def pca(data, method="k-most", k=1.):
    
    # covariance, eigenvectors, eigenvalues
    cov_matrix = np.cov(data);
    eig_val, eig_vec = np.linalg.eig(cov_matrix);
    
    eig_val = np.abs(eig_val);
    
    if (method == "k-most"):
        
        # create a list of (eigenvector, eigenvalue, index) tuples
        eig_list = [(eig_vec[i], eig_val[i], i) for i in range(len(eig_val))];
               
        # order by descending explained variance
        eig_list.sort(key=lambda x: x[1], reverse=True);
                
        # take the first k dimensions
        out = list(i[0] for i in eig_list[:k]);
                 
    elif (method == "percentage"):
        
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
        
    else:
        
        print("Warning: undefined method ", method, " for PCA analysis.");
        return data;
    
    return out;
    
    

# pandas DataFrame as input
def pca_labeled():
    
    # non-numerical objects unalikability
    pass;