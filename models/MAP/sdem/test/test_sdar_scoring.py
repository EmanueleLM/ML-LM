# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:36:04 2018

@author: Gabriele

Sequentially discounting Expectation Maximization algorithm, references
 "Yamanishi, Kenji, and Jun-ichi Takeuchi. "A unifying framework for detecting 
  outliers and change points from non-stationary time series data." 
  Proceedings of the eighth ACM SIGKDD international conference on Knowledge 
  discovery and data mining. ACM, 2002."
  
"""

import numpy as np
from scipy.stats import multivariate_normal

# multivariate gaussian estimation
def multivariate_gaussian(x, mean, cov):

    d = cov.shape[0];
    
    try: 
        det = np.linalg.det(cov);
        inv = np.linalg.inv(cov);
        
        g_exp = (-1/2) * np.dot(np.dot((x-mean).T, inv), (x-mean));
             
        p_x = np.exp(g_exp)/np.sqrt(((2*np.pi)**d) * det);
        
        return p_x[0,0];
    
    # covariance matrix has positive (or null) determinant
    except np.linalg.LinAlgError:
         
        return 0.
    
# Scoring function, section 3. of the paper
def outlier_detect_gauss(x, mu, gamma):
    
    res = multivariate_normal(mu.flatten(), gamma, allow_singular=True).pdf(x)
    #res = multivariate_gaussian(x, mu, gamma)
    res = -np.log2(res)
    
    return res  
  
# Change Point Detection, section 4. of the paper    
def change_point_detect(X, T):
    
    scoring = list()
       
    for i in range (T, len(X)):
        
        scoring.append(np.sum(np.vstack(X[i-T:i])/T))
            
    return scoring
    
    