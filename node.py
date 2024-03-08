from __future__ import annotations
import numpy as np
from typing import Dict


class Node:
    def __init__(self, index, id=1, x_coord=0, y_coord=0, demand=0, is_depot=False):
        self.index = index
        self.id = id
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.demand = demand
        self.is_depot = is_depot
        self.neighbors = []
        self.cluster_idx: int = -1
        self.candidate_clusters = {}
        self.distances: Dict[Node, float]
        # self.route = "empty"
        # self.position = "empty"

    def reset_cluster(self):
        self.candidate_clusters = {}
        self.cluster_idx = -1

    def __lt__(self, other):
        return self.id < other.id

    def get_coords(self):
        return self.x_coord, self.y_coord

    def get_demand(self):
        return self.demand

    def get_id(self):
        return self.id

    def set_neighbors(self, neighbors: list[Node]) -> None:
        self.neighbors = list(neighbors)

    def dist_to_other(self, other: Node, ret_squared=False, type: str = 'euc', max_demand=1) -> float:
        match type:
            case 'euc':
                dist = (self.x_coord-other.x_coord)**2 + (self.y_coord-other.y_coord)**2
                if ret_squared:
                    return dist
                return np.sqrt(dist)
            case 'route':
                return self.distances[other]
            case 'demand_inv':
                return np.exp2(-0.5*np.abs(self.demand-other.demand)/max_demand)
        return -1

    # def __str__(self):
    #     return f"""Node: {self.id}
    #     Coordination: ({np.around([self.x_coord, self.y_coord],3)})
    #     Demand: {self.demand}
    #     Cluster: {self.cluster_idx}"""

    def __str__(self) -> str:
        return str(self.id)
