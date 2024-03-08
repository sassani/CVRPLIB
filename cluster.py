from __future__ import annotations
from node import Node
import heapq
import numpy as np


class Cluster:
    def __init__(self, node: Node | None):
        self.nodes = set()
        if node is not None:
            self.index = node.index
            self.nodes.add(node)
            self.total_demand = node.demand
            self.mean_demand = node.demand

    def add_node(self, _new: Node) -> None:
        self.nodes.add(_new)

    def merge(self, other: Cluster) -> None:
        self.nodes.update(other.nodes)
        self.total_demand += other.total_demand
        self.mean_demand = self.total_demand/len(self.nodes)

    # def cluster_center(self):
    #     min_dist = np.Infinity
    #     n = len(self.nodes)
    #     medoid: Node | None = None
    #     for n1 in self.nodes:
    #         dist = 0
    #         for n2 in self.nodes:
    #             dist += n1.dist_to_other(n2, type='rout')
    #             # dist += distances.loc[nodes[i], nodes[j]]
    #         if (dist < min_dist):
    #             min_dist = dist
    #             medoid = n1
    #     return medoid

    # def __str__(self) -> str:
    #     return f"""Cluster: {self.idx}
    #     Nodes:{str(sorted(list(n.id for n in self.nodes)))}
    #     Neighbors: {str(list(n.id for n in self.neighbors))}
    #     Total Demand: {self.total_demand}"""


if __name__ == '__main__':
    pass
