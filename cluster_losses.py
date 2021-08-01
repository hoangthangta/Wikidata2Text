import torch
import numpy as np
import sklearn
import math

from scipy.cluster.vq import kmeans, vq

def mse_loss(data, centroids):
    """
    Calculate mse loss between data and centroids by torch (a method of kmeans loss)
    Reference:
        [1] https://arxiv.org/pdf/1801.07648.pdf (formula 3, page 4)
        [2] https://discuss.pytorch.org/t/k-means-loss-calculation/22041
    # Arguments
        data: input data (torch tensor)
        single_value: a single value (torch tensor)
    # Return
        sum_loss, mean_loss (float, float)
    """
    
    data = torch.Tensor(data)
    centroids = torch.Tensor(centroids)
    loss_tensor = ((data[:, None]-centroids[None])**2).sum(2).min(1)
    return loss_tensor[0].sum().item(), loss_tensor[0].mean().item()

def rmse_loss(data, centroids):
    """
    Calculate rmse loss
    # Arguments
        data: input data (float)
        single_value: a single value (float)
    # Return
        sum_loss, mean_loss (float, float)
    """
    sum_loss, mean_loss = mse_loss(data, centroids)
    return math.sqrt(sum_loss), math.sqrt(mean_loss)

def kmeans_loss(data, n_clusters):
    """
    Calculate kmeans cluster loss (also euclidean_distance_mean_loss())
    # Arguments
        data: input data (list or numpy array)
        n_clusters: number of clusters (int)
    # Return
       sum_loss, mean_loss, centroids, idx (float, float, list, list)
       The mean (non-squared) Euclidean distance between the observations passed and the centroids generated.
       Note the difference to the standard definition of distortion in the context of the k-means algorithm,
       which is the sum of the squared distances.
    """
    data = np.array(data)
    data = data.astype('float')

    print(data)
    
    centroids, mean_loss = kmeans(data, n_clusters)
    sum_loss = mean_loss*len(data) # sum loss
    idx,_ = vq(np.array(data), centroids)
    return sum_loss, mean_loss, centroids, idx

def euclidean_distance_mean_loss(data, centroids):
    """
    Calculate euclidean distance mean loss between data and centroids (also similar to kmean_loss())
    # Arguments
        data: input data (torch tensor)
        single_value: a single value, usually the min value (torch tensor)
    # Return
       sum_loss, mean_loss (float, float)
    """ 
    data = torch.Tensor(data)
    centroids = torch.Tensor(centroids)
    sqrt_tensor = torch.sqrt(((data[:, None]-centroids[None])**2).sum(2)).min(1)
    return sqrt_tensor[0].sum().item(), sqrt_tensor[0].mean().item()

#...................................................................................
'''import sklearn
import math
import torch
import numpy as np
import time

from cluster_losses import *
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import mean_squared_error

data = torch.randn(100000, 16)
centroids = torch.randn(2, 16)

#data = [[2,2], [3,3], [4,4]]
#centroids = [[1,1],[2,2]]

print('data: ', data)
print('centroids: ', centroids)

print(mse_loss(data, centroids))
print(rmse_loss(data, centroids))
print(euclidean_distance_mean_loss(data, centroids))

print(kmean_loss(data, 16))

#current_time = time.time()
#finish_time = time.time()
#print(finish_time-current_time)'''
