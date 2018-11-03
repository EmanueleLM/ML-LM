# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:44:09 2018

@author: Gabriele

Sequentially discounting Expectation Maximization algorithm, references
 "Yamanishi, Kenji, and Jun-ichi Takeuchi. "A unifying framework for detecting 
  outliers and change points from non-stationary time series data." 
  Proceedings of the eighth ACM SIGKDD international conference on Knowledge 
  discovery and data mining. ACM, 2002."
  
"""

import numpy as np
import copy as cp

# sdar algorithm
def sdar(X, discount = 0.1, k = 10):

    d, n = X.shape
    
    # initialize covariance functions estimators
    C = np.random.uniform(low=-1., high=1., size=(k, 1))   
    C_prev = np.random.uniform(low=-1., high=1., size=(k, 1));     
    
    # parameters of the model (used in eq. (6) of the paper)
    weights = np.random.uniform(low=.0, high=1., size=(k, 1))
    A = np.zeros(shape=(k, k)) 
    
    # bootstrap of the estimations of our model
    mu_hat = np.zeros(shape=(1, d))      
    sigma_hat = np.zeros(shape=(d,d))
    
    # list of results (last two equations after eq. (6) of the paper)
    SIGMA_HAT = list()
    X_HAT = list()
    
    for t in range (k, n):
        
        mu_hat = (1-discount) * mu_hat + discount * X[:, t]
               
        for j in range (k):
                    
            C[j] = (1 - discount) * C[j] + discount * np.dot(X[:, t] - mu_hat, (X[:, t-j] - mu_hat).T) 
            
        window = np.concatenate((C_prev, C)).flatten()
        
        for a in range(k):
            
            A[a,:] = window[k-a-1:2*k-a-1]
         
        weights = np.linalg.solve(A, C)
          
        # copy previous window into the new one
        C_prev = cp.copy(C)
    
        # estimate next prediction  
        X_hat = np.multiply(weights, (X[:,t-k:t]-mu_hat).T) + mu_hat
        
        sigma_hat = (1 - discount)*sigma_hat + discount*np.multiply((X[:,t].T-X_hat), (X[:,t].T-X_hat).T)
        
        # append estimations to the results
        X_HAT.append(X_hat)
        SIGMA_HAT.append(sigma_hat)
        
    return X_HAT, SIGMA_HAT
        



        
        
    
        
        
            
            
    
        
        