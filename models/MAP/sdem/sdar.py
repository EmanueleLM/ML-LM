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

def SDAR(X, discount=.1, k=10):
    
    n_sample = X.shape[1];
    
    # initialization: parameters 'mu_hat' and 'C_j'
    mu_hat = np.zeros(shape=(X.shape[0])); 
    C_j_prec = np.zeros(shape=(X.shape[0], k)); # previous window
    C_j = np.zeros(shape=(X.shape[0], k));
       
    # parameters of the model
    A = np.zeros(shape=(X.shape[0], k));
    W = np.zeros(shape=(k,k));
    
    # solutions to the algorithm
    x_hat = np.zeros(shape=(X.shape[0], n_sample-k));
    means_gaussian = np.zeros(shape=(X.shape[0], n_sample-k));
    sigma = np.zeros(shape=(X.shape[0], X.shape[0], n_sample-k));
    
    for i in range(k, n_sample):
            
        mu_hat = (1-discount)*mu_hat + discount*X[:,i];
        
        # assign the new mean of the gaussian
        means_gaussian[:,i] = mu_hat;
        
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
        
        # predict the parameters for the next sample
        x_hat[:,i] = np.sum(np.dot(A[j], X[:,i-k:i]-mu_hat), axis=1) + mu_hat;
        # assign the new covariance matrix of the gaussian
        sigma[:,:,i] = (1-discount)*sigma[:,:,max(0, i-1)] + discount*np.sum((X[:,i]-x_hat[:,i])**2);
        
    return means_gaussian, sigma;
        
        

