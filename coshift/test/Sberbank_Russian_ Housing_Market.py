# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:52:38 2018

@author: Emanuele

Simple test for covariance shift module
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../coshift/");

import coshift as csh
import pandas as pd

def main(argv):
    
    # load train and test
    train = pd.read_csv("../data/train.csv");
    test = pd.read_csv("../data/test.csv");
    
    # add 'target' feature (identifies train and test), discard real target feature
    df = csh.add_objective(train, test, target='price_doc');
    
    # tranform 'object' data in data usable by the tree classifier
    df = csh.label_encoding(df);
    
    # deal with missing values
    df = csh.missing_values(df);
    
    # split train and test (70%/30%)
    train, test = csh.split(df, train_percentage=.3);
    
    # calculate accuracy on split, true positve rate and false positve rate for each feature
    acc, tpr, fpr = csh.coshift(train, test);
    


if __name__ == "__main__":
    main(sys.argv);