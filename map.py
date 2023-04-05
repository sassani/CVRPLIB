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

    def import_nodes_from_file(self, path:str, type:str):
        nodes, meta_data = read_file(path, type)
        self.map_info = meta_data
        self.info = meta_data
        self.nodes = []
        self.import_nodes_from_df(nodes)
        # self.distances = dist.calculate_euclidean(self.nodes)

    def import_nodes_from_df(self, df: pd.DataFrame):
        for item in df.iterrows():
            node = item[1]
            node = Node(item[0], node['ID'], node['NODE_COORD'][0], node['NODE_COORD'][1], node['DEMAND'], node['IS_DEPOT'])
            self.network.add_node(node, 
                                  pos=(node.x_coord, node.y_coord), 
                                  color='red'if node.is_depot else 'green',
                                  demand=node.demand)
            self.nodes.append(node)

    def create_net_by_distance(self, dist_matrix:str):
         data, _ = read_file(dist_matrix, 'dist')
         n = len(data)
         for row in np.arange(n):
            for col in np.arange(row+1,n):
                dist_val = data.iloc[row,col]
                if dist_val > 0:
                    self.network.add_edge(self.nodes[row], self.nodes[col], weight=dist_val)
            
    def create_net_by_voronoi(self):
         points = [(n.x_coord, n.y_coord) for n in self.nodes]
         vor = Voronoi(points)
         for n in vor.ridge_points:
            a:Node = self.nodes[n[0]]
            b:Node = self.nodes[n[1]]
            self.network.add_edge(a,b, weight=a.dist_to_other(b))

    def draw(self, size=(15,15)):
        fig, ax = plt.subplots(1,1, figsize=size)
        pos=nx.get_node_attributes(self.network,'pos')
        colors=nx.get_node_attributes(self.network,'color').values()
        labels=nx.get_node_attributes(self.network,'demand')

        nx.draw(self.network, pos, with_labels=True, font_weight=1)
        nx.draw_networkx_nodes(self.network, pos, node_color=colors)
        # nx.draw_networkx_labels(self.network, pos, labels=labels)
        plt.show()

    def create_route():
        pass


if __name__ == '__main__':
    from scipy.spatial import Voronoi, voronoi_plot_2d
    print('Running MAP class...')
    map = Map()
    # map.create_map_from_file('sample_data/X-n101-k25.vrp', 'cvrp')
    # map.create_map_from_file('sample_data/X-n5.vrp', 'cvrp')
    map.import_nodes_from_file('sample_data/X-n15.vrp', 'cvrp')
    map.create_net_by_distance('sample_data/X-n15.dist')
    # map.create_net_by_voronoi()
    map.draw((10,10))
    # print(map.distances)
    # print('ok')
