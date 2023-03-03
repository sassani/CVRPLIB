import pandas as pd
import numpy as np
import itertools as itter
from node import Node
import networkx as nx
from scipy.spatial import Voronoi
import distance as dist


class Map:
    def __init__(self, df_nodes: pd.DataFrame, info, df_distances: pd.DataFrame = None) -> None:
        self.info = info
        self.nodes = []
        self.create_nodes(df_nodes)
        self.distances = df_distances
        if (df_distances == None):
            self.distances = dist.calculate_euclidean(self.nodes)

    def create_nodes(self, df: pd.DataFrame):
        for item in df.iterrows():
            node = item[1]
            self.nodes.append(Node(item[0], node['ID'], node['NODE_COORD']
                              [0], node['NODE_COORD'][1], node['DEMAND'], node['IS_DEPOT']))

    def create_route():
        pass


if __name__ == '__main__':
    from data import read_vrp
    data, meta_data = read_vrp('X-n5.vrp')
    map = Map(data, meta_data)
    # map.calculate_euclidean()
    # map.distances = Map.calculate_euclidean(map.nodes)
    print(map.distances)
    print('ok')
