# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:57:30 2018

@author: Emanuele

K-Means implementation
"""

from distance import distance as dist

distances_dict = {'L2': dist.L2, 'L1': dist.L1, 'L0': dist.L0, 
                  'LInf': dist.LInf}; 

def kmeans(data, k=10, distance='L2'):
    
    pass;