import pandas as pd
import numpy as np
import itertools as itter
from node import Node
from scipy.spatial import Voronoi
from distance import *


def distance_matrix_euc(nodes: list[Node], ret_squared=False) -> pd.DataFrame:
    """Calculate the Euclidean distance matrix between all given Nodes

    Args:
        nodes (list[Node]): NOde object with coordination X and Y

    Returns:
        pd.DataFrame: A symmetric matrix of distances as a DataFrame
    """
    d = np.zeros((len(nodes), len(nodes)))
    for n in itter.combinations(nodes, 2):
        n1: Node = n[0]
        n2: Node = n[1]
        dist = n1.dist_to_other(n2, ret_squared)
        d[n1.index, n2.index] = dist
        d[n2.index, n1.index] = dist
    node_indices = [x.id for x in nodes]
    return pd.DataFrame(d, index=node_indices, columns=node_indices)


def distance_euc(n1: Node, n2: Node, ret_squared=False) -> float:
    dist = n1.dist_to_other(n2)
    if ret_squared:
        return dist
    return np.sqrt(dist)

# # calculate the centroid of each cluster by
# # getting the average of datapoints
# def get_centroids(clusters:list(Cluster), display=False):
#     centroids=[]
#     for c in clusters:
#         if(display):print(f'cluster {c}:', *clusters[c])
#         centroids.append(np.mean(clusters[c], axis=0))
#     if(display):
#         print('Centroids: ', *centroids)
#         print("\n")
#     return centroids

# calculate the medoid by choosing a point with the 
# minimum sum distance between other points in the cluster.
def get_medoid(nodes:list[Node],distances:pd.DataFrame):
    min_dist=np.Infinity
    n = len(nodes)
    medoid:Node=nodes[0]
    for n1 in nodes:
        dist = 0
        for n2 in nodes:
            dist += get_distance(n1, n2, distances)
            # dist += distances.loc[nodes[i], nodes[j]]
        if(dist<min_dist):
            min_dist=dist
            medoid = n1
    return medoid

def get_distance(n1:Node, n2:Node, distances:pd.DataFrame):
    return distances.loc[n1,n2]