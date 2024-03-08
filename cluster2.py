from node import Node
import heapq
import numpy as np


class Cluster:
    def __init__(self, id: int = -1):
        self.nodes = set()
        # self.nodes_candidates:list[Node]=[]
        self.idx: int = id
        self.total_demand = 0
        self.neighbors: list[Node] = []
        # heapq.heapify(self.neighbors)

    def add_node(self, _new: Node) -> None:
        if _new not in self.nodes:
            print(f"cluster: {self.idx}, node: {_new.id}")
            self.nodes.add(_new)
            self.total_demand += _new.demand
            _new.cluster_idx = self.idx
            for node in _new.neighbors:
                # heapq.heappush(self.neighbors, (node.demand, node))
                if node.cluster_idx == -1:
                    self.neighbors.append(node)
                    node.candidate_clusters[self.idx] = node.id

    def remove_node(self, node: Node) -> float:
        return 0

    def cluster_center(self):
        min_dist = np.Infinity
        n = len(self.nodes)
        medoid: Node | None = None
        for n1 in self.nodes:
            dist = 0
            for n2 in self.nodes:
                dist += n1.dist_to_other(n2, type='rout')
                # dist += distances.loc[nodes[i], nodes[j]]
            if (dist < min_dist):
                min_dist = dist
                medoid = n1
        return medoid

    def __str__(self) -> str:
        return f"""Cluster: {self.idx}
        Nodes:{str(sorted(list(n.id for n in self.nodes)))}
        Neighbors: {str(list(n.id for n in self.neighbors))}
        Total Demand: {self.total_demand}"""


if __name__ == '__main__':
    pass
