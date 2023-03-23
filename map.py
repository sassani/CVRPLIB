import pandas as pd
import numpy as np
import itertools as itter
from node import Node
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
import distance as dist
from data import read_file


class Map:
    def __init__(self) -> None:
            self.nodes = []
            self.distances = None
            self.network = nx.Graph()

    def create_map_from_file(self, path:str, type:str):
        nodes, meta_data = read_file(path, type)
        self.info = meta_data
        self.nodes = []
        self.create_nodes(nodes)
        self.distances = dist.calculate_euclidean(self.nodes)

    def create_net_by_distance(self, distance):
         pass         

    def create_nodes(self, df: pd.DataFrame):
        for item in df.iterrows():
            node = item[1]
            node = Node(item[0], node['ID'], node['NODE_COORD'][0], node['NODE_COORD'][1], node['DEMAND'], node['IS_DEPOT'])
            self.network.add_node(node, pos=(node.x_coord, node.y_coord), color='red'if node.is_depot else 'green')
            self.nodes.append(node)
            
    def create_net_by_voronoi(self):
         points = [(n.x_coord, n.y_coord) for n in self.nodes]
         vor = Voronoi(points)
         for n in vor.ridge_points:
            a:Node = self.nodes[n[0]]
            b:Node = self.nodes[n[1]]
            self.network.add_edge(a,b, weight=a.dist_to_other(b))

    def draw(self):
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        pos=nx.get_node_attributes(self.network,'pos')
        colors=nx.get_node_attributes(self.network,'color').values()
        nx.draw(self.network, pos, with_labels=True, font_weight=1)
        nx.draw_networkx_nodes(self.network, pos, node_color=colors)
        plt.show()

    def create_route():
        pass


if __name__ == '__main__':
    from scipy.spatial import Voronoi, voronoi_plot_2d
    print('Running MAP class...')
    map = Map()
    map.create_map_from_file('sample_data/X-n101-k25.vrp', 'cvrp')
    # map.create_map_from_file('sample_data/X-n5.vrp', 'cvrp')
    map.create_net_by_voronoi()
    map.draw()
    # print(map.distances)
    # print('ok')
