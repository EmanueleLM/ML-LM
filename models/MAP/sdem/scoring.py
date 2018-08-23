# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:21:08 2018

@author: Emanuele

Scoring and change point detection functions. References
 "Yamanishi, Kenji, and Jun-ichi Takeuchi. "A unifying framework for detecting 
  outliers and change points from non-stationary time series data." 
  Proceedings of the eighth ACM SIGKDD international conference on Knowledge 
  discovery and data mining. ACM, 2002."
"""

import numpy as np

# generalization of the one-dimensional (univariate) normal distribution to 
#  higher dimensions
def multivariate_gaussian(x, mean, cov):
    
    d = x.shape[0];
     
    # covariance matrices have positive (or null) determinant
    det = np.linalg.det(cov);
    inv = np.linalg.inv(cov);
          
    g_exp = (-1/2) * np.dot(np.dot((x-mean), inv), (x-mean));
     
    p_x = (np.exp(g_exp))/np.sqrt((2*np.pi)**d * det);
     
    return p_x;

# 
def outlier_detection_gaussian(x, mu, gamma):
    
    res = -np.log2(multivariate_gaussian(x, mu, gamma));
    
    return res;

def change_point_detection(X, MU, GAMMA):
    
    n_samples = X.shape[1];
    
    total_scoring = .0;
    
    for i in range(n_samples):
        
        total_scoring += outlier_detection_gaussian(X[:,i], MU[:,i], GAMMA[:,:,i]);
        
    return total_scoring/n_samples;
        
        