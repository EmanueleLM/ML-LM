# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:57:30 2018

@author: Emanuele

K-Means implementation
"""

from distance import distance as dist
from utils import cinit as cinit

import numpy as np

# dictionary to pretty-invoke distances' routines
distances_dict = {'L2': dist.L2, 'L1': dist.L1, 'L0': dist.L0, 
                  'LInf': dist.LInf};

# dictionary to pretty-invoke clusters initialization's routines                  
cluster_init_dict = {'random': cinit.random, 'kmeans++': cinit.kmeans_plus_plus};

# input:
#  data, actual points we want to cluster;
#  k, number of clusters we want to split the data in;
#  distance, measure of distance we want to employ to assign points to clusters;
#  init, initialization method for the center of each cluster;
#  epsilon, convergence measure.
def kmeans(data, k=10, distance='L2', init='random', epsilon=1e-3):
       
    # initialize centers of each cluster
    centers = cinit.cluster_init_dict[init](data, k, distance);
    
    # init a list of indices for each point in the dataset, for each cluster
    clusters = list([list() for _ in range(k)]);
    
    convergence = False;
    
    while convergence is False:
        
        # check if convergence criterion is met
        total_distance = .0; # sum of inter-cluster distances of current iteration
        old_total_distance = .0; # sum of inter-cluster distances from prev iteration
        
        for (cen, clu) in (centers, clusters):
            
            total_distance += np.sum(distances_dict[distance](cen, data[clusters[clu]]), axis=0);
        
        if np.abs(total_distance-old_total_distance) > epsilon:
            
            break;
        
        else:
            
            old_total_distance = total_distance;
        
        
        
        