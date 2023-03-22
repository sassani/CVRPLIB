import pandas as pd
import numpy as np
import itertools as itter
from node import Node
import networkx as nx
from scipy.spatial import Voronoi
import distance as dist
from data import read_file


class Map:
    def __init__(self) -> None:
            self.nodes = []
            self.distances = None

    # def create_map_by_vrp(self, path:str):
    #     nodes, meta_data = _read_vrp(path)
    #     self.info = meta_data
    #     self.nodes = []
    #     self.create_nodes(nodes)
    #     self.distances = dist.calculate_euclidean(self.nodes)

    def create_map_from_file(self, path:str, type:str):
        nodes, meta_data = read_file(path, type)
        self.info = meta_data
        self.nodes = []
        self.create_nodes(nodes)
        self.distances = dist.calculate_euclidean(self.nodes)

    def create_map_by_distances(self, distances):
         pass

    def create_nodes(self, df: pd.DataFrame):
        for item in df.iterrows():
            node = item[1]
            self.nodes.append(Node(item[0], node['ID'], node['NODE_COORD']
                              [0], node['NODE_COORD'][1], node['DEMAND'], node['IS_DEPOT']))

    def create_route():
        pass


if __name__ == '__main__':
    print('Running MAP class...')
    map = Map()
    map.create_map_from_file('sample_data/X-n5.vrp', 'cvrp')
    # map.calculate_euclidean()
    # map.distances = Map.calculate_euclidean(map.nodes)
    print(map.distances)
    print('ok')
