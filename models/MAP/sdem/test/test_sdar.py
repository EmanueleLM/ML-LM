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

import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sdar_alg as sdar_alg
import scoring as prot


index = pd.read_csv('TOPIX_index.csv').values
plt_index = cp.copy(index) # unnormalized and used for plot purpose

#initialize dataset and parameters for the sdar test
index = (index-min(index))/(max(index)-min(index))
index = index.T
k = 8
res = sdar_alg.sdar(index, 0.005, k);
x_hat, sigma_hat = res[0], res[1]


# estimate outliers
sample_size = len(x_hat)
score = list()
score_plot = np.array([])

for i in range(sample_size):
    
    score.append(prot.outlier_detect_gauss(index[:,i+k], x_hat[i], sigma_hat[i]));

# estimate change point
T = 5
change_point = prot.change_point_detect(score, T) 

# estimate new change point by applying another sdar layer (eq. 10)
new_change_point = change_point-min(change_point)/(max(change_point)-min(change_point))

new_change_point = np.vstack(change_point).T
res_2 = sdar_alg.sdar(new_change_point, 0.005, k)
x_hat_2, sigma_hat_2 = res_2[0], res_2[1]

average_size=len(x_hat_2)
score_2= list()
score_2_plot = np.array([])

for i in range(average_size):
    
    score_2.append(prot.outlier_detect_gauss(new_change_point[:,i+k], x_hat_2[i], sigma_hat_2[i]))

T_1 = 5
change_point_2 = prot.change_point_detect(score_2, T_1)


#plot series and change point on the same image 
# the former is the index of time series, the latter the results from equation 10.
fig, ax1 = plt.subplots()

# data series
ax1.plot(plt_index[30:], 'b', label='TOPIX')
ax1.set_xlabel('Date')
ax1.set_ylabel('TOPIX')

# change points
ax2 = ax1.twinx()
change_point_2 = np.vstack(change_point_2)[30:]
ax2.plot(change_point_2, 'r', label='change point')
ax2.set_ylabel('Change Point')
plt.legend(loc='best')

fig.tight_layout()
plt.show()

# save graph
fig.savefig('graph.pdf')
