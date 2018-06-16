# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 12:56:13 2018

@author: Emanuele

Clusters initialization routines
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../");

from distance import distance as dist

# dictionary to pretty-invoke distances' routines
distances_dict = {'L2': dist.L2, 'L1': dist.L1, 'L0': dist.L0, 
                  'LInf': dist.LInf};

import numpy as np

# random initialization procedure
def random(dims):
    
    clusters = np.random.rand(*(dims));
    
    return clusters;
 
# kmeans++ initialization procedure
def kmeans_plus_plus(data, k, distance):
    
    clusters = list(data[np.random.choice(data.shape[0])]);
    
    for _ in range(k-1):
        
        # create pdf for each datapoint wrt the last element of the cluster
        p_cluster = distances_dict[distance](clusters[-1], data);
        p_cluster = p_cluster/np.sum(p_cluster, axis=0);
        
        # select new point of the cluster by picking from the pdf
        clusters.append(np.random.choice(range(data.shape[0]), size=1, p=p_cluster));
            
    return clusters;