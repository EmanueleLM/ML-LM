# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:35:31 2018

@author: Gabriele

Sequentially discounting Expectation Maximization algorithm, references
 "Yamanishi, Kenji, and Jun-ichi Takeuchi. "A unifying framework for detecting 
  outliers and change points from non-stationary time series data." 
  Proceedings of the eighth ACM SIGKDD international conference on Knowledge 
  discovery and data mining. ACM, 2002."
  
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../");

import sdar as sdar_alg

import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

index = pd.read_csv('data/TOPIX_index.csv').values
plt_index = cp.copy(index) # unnormalized and used for plot purpose

#initialize dataset and parameters for the sdar test
index = (index-min(index))/(max(index)-min(index))
index = index.T
k = 4
res = sdar_alg.sdar(index, 0.005, k);
x_hat, sigma_hat = res[0], res[1]


# estimate outliers
sample_size = len(x_hat)
score = list()
score_plot = np.array([])

for i in range(sample_size):
    
    score.append(sdar_alg.outlier_detect_gauss(index[:,i+k], x_hat[i], sigma_hat[i]));

# estimate change point
T = 5
change_point = sdar_alg.change_point_detect(score, T)   


# plot series and change point on the same image
fig, ax1 = plt.subplots()

# data series 
# exclude the first few points since the prediction may be biased by the initialization
ax1.plot(plt_index[10:], 'b', label='TOPIX')
ax1.set_xlabel('Date')
ax1.set_ylabel('TOPIX')

# change point
# exclude the first few points since the prediction may be biased by the initialization
ax2 = ax1.twinx()
change_point = np.vstack(change_point)
ax2.plot(change_point[10:], 'r', label='change point')
ax2.set_ylabel('Change Point')
plt.legend(loc='best')

fig.tight_layout()
plt.show()

# save graph
#fig.savefig('graph.pdf')