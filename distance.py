import pandas as pd
import numpy as np
import itertools as itter
from node import Node
from scipy.spatial import Voronoi


def calculate_euclidean(nodes: list[Node], ret_squared=False) -> pd.DataFrame:
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
