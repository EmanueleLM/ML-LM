# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:29:27 2018

@author: Emanuele

Sequentially discounting Expectation Maximization algorithm, references
 "Yamanishi, Kenji, and Jun-ichi Takeuchi. "A unifying framework for detecting 
  outliers and change points from non-stationary time series data." 
  Proceedings of the eighth ACM SIGKDD international conference on Knowledge 
  discovery and data mining. ACM, 2002."
"""

import copy as cp
import numpy as np

def SDAR(X, discount=.1, k=10):
    
    n_sample = X.shape[1];
    
    # initialization: parameters 'mu_hat' and 'C_j'
    mu_hat = np.zeros(shape=(X.shape[0])); 
    C_j_prec = np.zeros(shape=(X.shape[0], k)); # previous window
    C_j = np.zeros(shape=(X.shape[0], k));
    
    # parameters of the model
    A = np.zeros(shape=(X.shape[0], k));
    W = np.zeros(shape=(k,k));
    
    for i in range(k, n_sample):
            
        mu_hat = (1-discount)*mu_hat + discount*X[:,i];
        
        for j in range(k):
            
            C_j[:,j] = (1-discount)*C_j[:,j] + discount*np.dot(X[:,i]-mu_hat, (X[:,i-j]-mu_hat).T);
         
        # stack the old and new windows horizontaly
        window = np.hstack([C_j_prec, C_j]);
            
        for j in range(X.shape[0]):
            
            Y = C_j[j];
            
            for wind in range(k):
                
                W[wind] = window[1+wind:1+wind+k]; 
            
            A[j] = np.linalg.solve(W, Y);
            
        C_j_prec = cp.deepcopy(C_j);
        
        

