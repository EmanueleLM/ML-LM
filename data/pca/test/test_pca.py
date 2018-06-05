# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 23:51:38 2018

@author: Emanuele

PCA test
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../");

import numpy as np
import pandas as pd
from pca import pca, pca_labeled

if __name__ == "__main__":
    
    # PCA test
    # create numpy dataset (10 samples, 100 dimensions)
    x = np.random.rand(10, 100);
    
    # apply pca to extract the 5 most relevant features
    dims, k_dims = pca(x, k=5);
    
    # PCA labeled test
    # now create a pandas dataset with some categorical features
    x_objects = np.chararray((10, 15));
    
    # create some numerical data
    x_num = np.random.rand(10, 15);
    
    # concatenate numbers and objects
    x = np.append(x_objects, x_num);
    
    # ...