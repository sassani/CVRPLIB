import pandas as pd
import numpy as np
import itertools as itter
from node import Node
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
import distance as dist
from data import read_file, create_random_cvrp


class Map:
    def __init__(self) -> None:
        self.nodes = []
        self.distances = None
        self.network = nx.Graph()

    def import_nodes_from_file(self, path: str, type: str)->None:
        nodes, meta_data = read_file(path, type)
        # self.map_info = meta_data
        self.info = meta_data
        self.nodes = []
        self.nodes_df = nodes
        self._import_nodes_from_df(nodes)
        # self.distances = self._shortest_length_matrix()
        # self.paths = self._shortest_path_matrix()

    def create_random_nodes(self, width: int = 10, height: int = 10, dem_min: int = 1, dem_max: int = 100, n: int = 10, capacity=None, is_grided=False, seed=None)->None:
        nodes, meta_data = create_random_cvrp(width, height, dem_min,
                              dem_max, n, capacity, is_grided, seed)
        self._import_nodes_from_df(nodes)
        self.nodes_df = nodes
        self.info = meta_data

    def _import_nodes_from_df(self, df: pd.DataFrame)->None:
        for item in df.iterrows():
            node = item[1]
            node = Node(item[0], node['ID'], node['NODE_COORD'][0],
                        node['NODE_COORD'][1], node['DEMAND'], node['IS_DEPOT'])
            self.network.add_node(node,
                                  pos=(node.x_coord, node.y_coord),
                                  color='red'if node.is_depot else 'green',
                                  demand=node.demand,
                                  id=node.id)
            self.nodes.append(node)

    def create_net_by_distance(self, dist_matrix: str)->None:
        data, _ = read_file(dist_matrix, 'dist')
        n = len(data)
        for row in np.arange(n):
            for col in np.arange(row+1, n):
                dist_val = data.iloc[row, col]
                if dist_val > 0:
                    self.network.add_edge(
                        self.nodes[row], self.nodes[col], weight=dist_val)
        self._initialize_paths_and_lengths()

    def create_net_by_voronoi(self)->None:
        points = [(n.x_coord, n.y_coord) for n in self.nodes]
        vor = Voronoi(points)
        for n in vor.ridge_points:
            a: Node = self.nodes[n[0]]
            b: Node = self.nodes[n[1]]
            self.network.add_edge(a, b, weight=a.dist_to_other(b))
        self._initialize_paths_and_lengths()
        

    def draw(self, size=(5, 5), show_demands=False)->None:
        fig, ax = plt.subplots(1, 1, figsize=size)
        pos = nx.get_node_attributes(self.network, 'pos')
        colors = nx.get_node_attributes(self.network, 'color').values()
        labels = None
        if show_demands:
            labels = nx.get_node_attributes(self.network, 'demand')
        else:
            labels = nx.get_node_attributes(self.network, 'id')

        nx.draw(self.network, pos, with_labels=False, font_weight=1)
        nx.draw_networkx_nodes(self.network, pos, node_color=colors)
        nx.draw_networkx_labels(self.network, pos, labels=labels)
        plt.show()

    def _initialize_paths_and_lengths(self)->None:
        self.distances = pd.DataFrame.from_dict(dict(self._shortest_length_matrix())).sort_index() 
        self.paths = pd.DataFrame.from_dict(dict(self._shortest_path_matrix())).sort_index()
        self._setup_neighbors()

    def _setup_neighbors(self)->None:
        for node in self.nodes:
            node.set_neighbors(self.network.neighbors(node))

    def _shortest_path_matrix(self):
        dist = nx.all_pairs_shortest_path(self.network)
        return dist

    def _shortest_length_matrix(self)->float:
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.network))
        return lengths
    
    def farthest_nodes(self, df_distance:pd.DataFrame)->tuple[Node,Node]:
        n1:Node = df_distance.max().idxmax()
        n2:Node = df_distance.loc[n1].idxmax()
        return (n1, n2)
    
    def closest_nodes(self, df_distance:pd.DataFrame):
        pass


if __name__ == '__main__':
    from scipy.spatial import Voronoi, voronoi_plot_2d
    print('Running MAP class...')
    map = Map()
    map.create_random_nodes()
    # map.create_map_from_file('sample_data/X-n101-k25.vrp', 'cvrp')
    # map.create_map_from_file('sample_data/X-n5.vrp', 'cvrp')
    # map.import_nodes_from_file('sample_data/X-n15.vrp', 'cvrp')
    # map.create_net_by_distance('sample_data/X-n15.dist')
    # map.create_net_by_voronoi()
    map.draw((10, 10))
    # print(map.distances)
    # print('ok')
