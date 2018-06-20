# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 08:41:44 2018

@author: Emanuele

test for clusters' initialization techniques 
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../");

from utils import cinit as cinit

import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    
    # plot test for clusters
    data = np.random.rand(100, 2); # 2-dim input points
    
    # create random centers of the clusters
    centers = cinit.random(data, k=10);
    
    plt.plot(data[:,0], data[:,1], 'r*', 
             centers[:,0], centers[:,1], 'g^');
             
    # create centers of the clusters with kmeans++ initialization's method
    centers = cinit.kmeans_plus_plus(data, k=10, distance='L2');
    
    plt.plot(data[:,0], data[:,1], 'r*', 
             centers[:,0], centers[:,1], 'bs');
    