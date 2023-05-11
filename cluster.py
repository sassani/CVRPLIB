from node import Node
import heapq


class Cluster:
    def __init__(self, id:int=None):
        self.nodes = set()
        self.idx:int = id
        self.total_demand = 0
        self.neighbors = []
        heapq.heapify(self.neighbors)

    def add_node(self, _new:Node)->None:
        if _new not in self.nodes:
            self.nodes.add(_new)
            self.total_demand += _new.demand
            _new.cluster = self.idx
            for n in _new.neighbors:
                heapq.heappush(self.neighbors, (n.demand, n))

    def remove_node(self, node:Node)->float:
        return 0

    def test(self):
        return heapq.nlargest(1, self.neighbors)
    
    def __str__(self) -> str:
        return f"""Cluster: {self.idx}
        Nodes:{str(list(n.id for n in self.nodes))}
        Neighbors: {str(list(n[1].id for n in self.neighbors))}
        Total Demand: {self.total_demand}"""


if __name__ == '__main__':
    pass