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

# random initialization procedure.
# returns the centers as datapoints.
def random(data, k, distance=None):
    
    centers = np.random.choice(range(data.shape[0]), size=k, replace=False);
    
    return data[centers];

# kmeans++ initialization procedure.
def kmeans_plus_plus(data, k, distance):
    
    # select the firt point at random from data
    centers = list();
    center = np.random.choice(range(data.shape[0]), size=1);
    centers.append(data[center]);
      
    for _ in range(k-1):
        
        # create pdf for each datapoint wrt the last element of the cluster
        p_cluster = distances_dict[distance](centers[-1][0], data);
        p_cluster = p_cluster/np.sum(p_cluster, axis=0);
        
        # select new point of the cluster by picking from the pdf
        center = np.random.choice(range(data.shape[0]), size=1, p=p_cluster);
        centers.append(data[center]);
        np.delete(data, center);
        
    # flatten the second dimension since it is a 1
    centers = np.squeeze(np.array(centers), axis=1);
            
    return centers;