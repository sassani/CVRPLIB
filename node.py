from __future__ import annotations
import numpy as np


class Node:
    def __init__(self, index, id, x_coord, y_coord, demand=0, is_depot=False):
        self.index = index
        self.id = id
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.demand = demand
        self.is_depot = is_depot
        # self.route = "empty"
        # self.position = "empty"

    def __lt__(self, other):
        return self.id < other.id

    def get_coords(self):
        return self.x_coord, self.y_coord

    def get_demand(self):
        return self.demand

    def get_id(self):
        return self.id

    def dist_to_other(self, other: Node, ret_squared=False):
        dist = (self.x_coord-other.x_coord)**2 + \
            (self.y_coord-other.y_coord)**2
        if ret_squared:
            return dist
        return np.sqrt(dist)

    def __str__(self):
        return (str(self.id))
