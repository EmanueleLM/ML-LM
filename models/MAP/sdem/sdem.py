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

import matplotlib.pyplot as plt
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
    

def SDEM(X, n_mixtures, alpha=1., discount=.1):
    
    n_sample = X.shape[1];
    
    # relative 'weight' of each gaussian distribution
    weights = np.array([1/n_mixtures for _ in range(n_mixtures)]);
       
    # centers of the gaussian mixtures, 'mu' in the paper
    centers = np.array([]);    
    
    # initialize centers of the gaussian mixtures
    for _ in range(n_mixtures):
        
        centers= np.append(centers, 
                           np.array([np.random.uniform(X[i].min(), X[i].max()) 
                           for i in range(X.shape[0])]));
    
    centers = centers.reshape(n_mixtures, X.shape[0]);
    plt.plot(centers[:,0], centers[:,1], '^g');  
    
    # initialization (to zero) of the parameter 'mu_hat'
    mu_hat = np.zeros(shape=centers.shape);  
    
    # initialize the covariance matrices (isotropic): this is the 'lambda' parameter
    cov = np.zeros(shape=(n_mixtures, X.shape[0], X.shape[0]));
    
    # initialize 'lambda_hat' parameter of the paper
    lambda_hat = np.zeros(shape=(n_mixtures, X.shape[0], X.shape[0]));
    
    for n in range(n_mixtures):
        
        cov[n] = np.multiply(np.eye(X.shape[0]), np.random.rand(X.shape[0]));
        lambda_hat[n] = np.multiply(np.eye(X.shape[0]), np.random.rand(X.shape[0]));
                           
    # gamma initialization
    gamma = np.zeros(n_mixtures);
    
    for i in range(n_sample):
        
        # total 'weighted' probability
        p_tot = np.sum([weights[n]*multivariate_gaussian(X[:,i], centers[n], cov[n]) for n in range(n_mixtures)]);
                
        for j in range(n_mixtures):
                                  
            # expectation:
            # probability of x to belong to the j-th mixture
            p = weights[j]*multivariate_gaussian(X[:,i], centers[j], cov[j]);
                        
            gamma[j] = ((1-alpha*discount)*(p/p_tot)) + (alpha*discount)/n_mixtures;
            
            # maximization:
            weights[j] = ((1-discount)*weights[j]) + discount*gamma[j];
            
            mu_hat[j] = ((1-discount)*mu_hat[j]) + discount*gamma[j]*X[:,i];
            centers[j] = mu_hat[j]/weights[j];
            
            lambda_hat[j] = ((1-discount)*lambda_hat[j]) + discount*gamma[j]*np.sum(X[:,i]**2);
            cov[j] = (lambda_hat[j]/weights[j]) - np.sum(centers[j]**2);
                                              
    return centers;
            
            