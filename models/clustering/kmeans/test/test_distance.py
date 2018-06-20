# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:46:44 2018

@author: Emanuele

k-means test for distances
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../");

from distance import distance as dist

import numpy as np

if __name__ == "__main__":
    
    # test result
     
    # create a vector where each entry has increasing number of ones
    x = np.array([[(1 if i>=j else 0) for i in range(10)] for j in range(10)]);
    x = x.T;
    
    # L2
    assert np.all(dist.L2(x, x[0]) == np.array([np.sqrt(i) for i in range(10)])), "L2 test has failed";
    
    # L1
    assert np.all(dist.L1(x, x[0]) == np.array([(i) for i in range(10)])), "L1 test has failed";
    
    # L0
    assert np.all(dist.L0(x, x[0]) == np.array([0 for i in range(10)])), "L0 test has failed";
    
    # LInf
    assert np.all(dist.LInf(x, x[0]) == np.array([(1 if i!=0 else 0) for i in range(10)])), "LInf test has failed";
    
    # test dimensions
    
    x = np.random.rand(*(i for i in range(10, 30, 5)));
       
    # L2
    assert len(dist.L2(x, x[0])) == (x.shape[0]), "L2 test has failed: dimensions mismatch";
    assert (dist.L2(x, x[0]).ndim) == 1, "L2 test has failed: dimensions mismatch";
    
    # L1
    assert len(dist.L1(x, x[0])) == (x.shape[0]), "L1 test has failed: dimensions mismatch";
    assert (dist.L1(x, x[0]).ndim) == 1, "L1 test has failed: dimensions mismatch";

    
    # L0
    assert len(dist.L0(x, x[0])) == (x.shape[0]), "L0 test has failed: dimensions mismatch";
    assert (dist.L0(x, x[0]).ndim) == 1, "L0 test has failed: dimensions mismatch";
    
    # LInf
    assert len(dist.LInf(x, x[0])) == (x.shape[0]), "LInf test has failed: dimensions mismatch";
    assert (dist.LInf(x, x[0]).ndim) == 1, "LInf test has failed: dimensions mismatch";

