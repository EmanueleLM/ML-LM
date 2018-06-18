# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:03:20 2018

@author: Emanuele

Implementation of distances: returns a vector of distances between a single point
 (e.g. the cluster) and other points (that can be whole the dataset).
"""

import numpy as np

# L2 or Euclidean norm between two points
def L2(pt, pts):
    
    res = np.power((pts-pt), 2); # SIMD operations
    res = np.sqrt(res.sum(axis=tuple((i for i in range(1,pt.ndim+1)))));
    
    return res;

# L1 norm    
def L1(pt, pts):
    
    res = np.sum(np.abs(pts-pt), axis=1); # SIMD operations
    
    return res;
  
# L0 which takes the minimum element from each pt to pts distance
def L0(pt, pts):
    
    res = np.abs(pts-pt); # SIMD operations
    res = np.min(res, axis=1);
    
    return res;
 
# LInf which takes the maximum element from each pt to pts distance
def LInf(pt, pts):
    
    res = np.abs(pts-pt); # SIMD operations
    res = np.max(res, axis=1);
    
    return res;