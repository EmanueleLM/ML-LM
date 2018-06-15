# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:57:30 2018

@author: Emanuele

K-Means implementation
"""

import distance as dist

distances_dict = {'L2': dist.L2, 'L1': dist.L1, 'LMin': dist.LMin, 
                  'LMax': dist.LMax,, 'Ln': dist.Ln}; 

def k-means(data, k=10, distance='L2'):
    
    pass;