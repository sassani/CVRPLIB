import pandas as pd
import numpy as np
import itertools as itter
from node import Node
from scipy.spatial import Voronoi


def calculate_euclidean(nodes: list[Node]) -> pd.DataFrame:
    d = np.zeros((len(nodes), len(nodes)))
    for n in itter.combinations(nodes, 2):
        n1: Node = n[0]
        n2: Node = n[1]
        dist = n1.dist_to_other(n2)
        d[n1.index, n2.index] = dist
        d[n2.index, n1.index] = dist
    node_indices = [x.id for x in nodes]
    return pd.DataFrame(d, index=node_indices, columns=node_indices)
