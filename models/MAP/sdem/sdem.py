# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:52:14 2018

@author: Emanuele

Sequentially discounting Expectation Maximization algorithm, references
 "Yamanishi, Kenji, and Jun-ichi Takeuchi. "A unifying framework for detecting 
  outliers and change points from non-stationary time series data." 
  Proceedings of the eighth ACM SIGKDD international conference on Knowledge 
  discovery and data mining. ACM, 2002."
"""

import numpy as np

def SDEM(X, n_mixtures, alpha=1, discount=1, init_method="random"):
    
    n_sample = 1/X.shape[1];
    
    # initialize centers of the gaussian mixtures
    centers = np.array([]);    
    
    for _ in range(n_mixtures):
        
        centers= np.append(centers, 
                           np.array([np.random.uniform(X[i].min(), X[i].max()) 
                           for i in range(X.shape[0])]));
    
    centers = centers.reshape(n_mixtures, X.shape[0]);

    # initialize the covariance matrix
    cov = np.zeros(shape=(centers.shape[0], centers.shape[0]));
    
    for i in range(cov.shape[0]):
        
        for j in range(cov.shape[0]):
            
            cov[i][j] = np.sum(X-centers[i], axis=1)*np.sum(X-centers[j], axis=1).T/n_sample
        
        