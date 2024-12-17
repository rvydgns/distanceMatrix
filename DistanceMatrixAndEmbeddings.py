import sklearn
import sklearn.datasets
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt

def computeDistanceMatrix(X):
    dist_matrix = scipy.spatial.distance.cdist(X, X)
    dist_matrix = dist_matrix / dist_matrix.mean() * 10
    return torch.tensor(dist_matrix)

def calculateObjective(Y, dist_matrix):
    euclidean_dist = torch.cdist(Y, Y)
    objective_value = ((torch.triu(dist_matrix - euclidean_dist) ** 2).sum() / torch.triu(dist_matrix ** 2).sum()) ** 0.5
    return objective_value

def visualizeEmbeddings(Y_history, T):
    plt.figure(figsize=(12, 8))
    for idx, i in enumerate([0, 3, 4, 5, 6, 10]):
        ax = plt.subplot(2, 3, idx + 1)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_title(f'Checkpoint {i}')
        ax.scatter(*Y_history[i].T, c=T, cmap='tab10', s=40, alpha=0.5)
    plt.show()
