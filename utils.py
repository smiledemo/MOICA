import torch
import numpy as np
from sklearn.cluster import SpectralClustering

def spectral_clustering(features, n_clusters):
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(features)
    return clustering.labels_

def smooth_labels(labels, k=5):
    smoothed_labels = []
    for i in range(len(labels)):
        distances = np.linalg.norm(labels - labels[i], axis=1)
        nearest_neighbors = np.argsort(distances)[:k]
        smoothed_label = np.mean(labels[nearest_neighbors], axis=0)
        smoothed_labels.append(smoothed_label)
    return np.array(smoothed_labels)

def calculate_loss(outputs, targets, criterion):
    loss = criterion(outputs, targets)
    return loss
