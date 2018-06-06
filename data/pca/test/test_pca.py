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
    x = np.concatenate((x_objects, x_num), axis=1);
    
    # create pandas DataFrame
    df = pd.DataFrame(x, columns=[str(a) for a in range(30)], dtype=object);
    df.iloc[:, 15:] = df.iloc[:, 15:].astype(float); # convert to float the numbers
    
    # launch pca_labeled
    res = pca_labeled(df, k=5, get_pandas=True);
    
