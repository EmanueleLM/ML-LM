# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 15:21:53 2018

@author: Emanuele

SDEM test with random bidimensional data
"""

# solve the relative import issues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../");

import matplotlib.pyplot as plt
import numpy as np

import sdem as sdem

if __name__ == "__main__":
    
    # create 3 random gaussian distributions
    data1 = np.random.multivariate_normal([0., 0.], [[1., .0],[.0, 1.]], 50).T;
    data2 = np.random.multivariate_normal([10., 4.], [[1., .0],[.0, 1.]], 50).T;
    data3 = np.random.multivariate_normal([6., 12.], [[1., .0],[.0, 1.]], 50).T;    
    
    # plot the data
    plt.plot(data1[0,:], data1[1,:], '*b');
    plt.plot(data2[0,:], data2[1,:], '*y');  
    plt.plot(data3[0,:], data3[1,:], '*m'); 
    
    # stack and shuffle the data 
    data = np.hstack([data2, data1, data3]);
    np.random.shuffle(data.T);
    
    # call the sdem function and return the centers of the gaussians
    centers = sdem.SDEM(data, 3, alpha=1.5, discount=0.02);
    
    # plot the centers
    plt.plot(centers[:,0], centers[:,1], '^r');