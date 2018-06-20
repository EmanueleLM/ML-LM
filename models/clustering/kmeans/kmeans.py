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
#
# output:
#  clusters, k resulting clusters;
#  (optional) assignments, for each datapoint, the i-th cluster it belongs to 
#
def kmeans(data, k=5, distance='L2', init='random', epsilon=1e-3, assignments=False):
       
    # initialize centers of each cluster
    centers = cluster_init_dict[init](data, k, distance);
      
    # init a list of indices for each point in the dataset, for each cluster
    clusters_assignments = np.array([]);
    intra_cluster_distance = np.array(-epsilon);
       
    convergence = False;
    
    # initial intra-clusters distance
    old_total_distance = np.inf;
       
    # check if convergence criterion is met
    while convergence is False:
                                            
        # assign each point to the closest cluster        
        pt_to_cluster_distance = np.array([]);
        
        for c in centers:
            
            d = distances_dict[distance](c, data);
                      
            if pt_to_cluster_distance.shape != (0,):
                
                # stack the distances, from each center, vertically
                pt_to_cluster_distance = np.vstack([pt_to_cluster_distance, d]);
                
            else:
                
                pt_to_cluster_distance = d;
                       
        # obtain distance from each nearest cluster and index of the cluster itself
        intra_cluster_distance = np.amin(pt_to_cluster_distance, axis=0);
        clusters_assignments = np.argmin(pt_to_cluster_distance, axis=0);
                                     
        # update the cluster's center
        for i in range(k):
            
            # a cluster may contain zero points assigned to it, in that case the
            #  cluster's center is unchanged
            if len(np.where(clusters_assignments==i)) > 0:
                           
                # for each cluster, 'reduce' by index assignment to get new center
                data_to_cluster_i = data[np.where(clusters_assignments==i)];
                            
                centers[i] = data_to_cluster_i.mean(axis=0);

                   
        if np.abs(np.sum(intra_cluster_distance, axis=0)-old_total_distance) < epsilon:
            
            break;
            
        else:
            
            # update total intra-clusters' distance
            old_total_distance = np.sum(intra_cluster_distance, axis=0);
        
                             
    if assignments is True:
        
        return centers, clusters_assignments;
    
    return centers;
            
            
            
                       
        
        