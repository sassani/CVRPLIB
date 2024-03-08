from typing import cast
import pandas as pd
import numpy as np
import itertools as itter
from node import Node
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
# import distance as dist
from data import read_file, create_random_cvrp


class Map:
    def __init__(self, seed:int|None=None) -> None:
        self.seed = seed
        self.nodes = []
        self.distances:pd.DataFrame
        self.network = nx.Graph()
        self.cluster_colors = self.set_colors()

    def import_nodes_from_file(self, path: str, type: str) -> None:
        nodes, meta_data = read_file(path, type)
        # self.map_info = meta_data
        self.info = meta_data
        self.nodes = []
        self.nodes_df = nodes
        self._import_nodes_from_df(nodes)
        # self.distances = self._shortest_length_matrix()
        # self.paths = self._shortest_path_matrix()

    def create_random_nodes(self,
                            width: int = 10,
                            height: int = 10,
                            dem_min: int = 1,
                            dem_max: int = 100,
                            n: int = 10,
                            capacity=None,
                            is_grided=False,
                            depots:list[int]=[1]) -> None:
        nodes, meta_data = create_random_cvrp(width, height, dem_min,
                                              dem_max, n, capacity, is_grided, self.seed)
        self.max_demand = dem_max
        for n in depots:
            nodes.loc[n-1, 'IS_DEPOT'] = True
            nodes.loc[n-1, 'DEMAND'] = 0
        self._import_nodes_from_df(nodes)
        self.nodes_df = nodes
        self.info = meta_data

    def _import_nodes_from_df(self, df: pd.DataFrame) -> None:
        max_demand=0
        for item in df.iterrows():
            node = item[1]
            node = Node(item[0], node['ID'], node['NODE_COORD'][0],
                        node['NODE_COORD'][1], node['DEMAND'], node['IS_DEPOT'])
            self.network.add_node(node,
                                  pos=(node.x_coord, node.y_coord),
                                  color='#FF0000'if node.is_depot else '#00FF15',
                                  demand=node.demand,
                                  id=node.id)
            self.nodes.append(node)
            max_demand = node.demand if max_demand < node.demand else max_demand
        self.size = len(self.nodes)
        self.max_demand = max_demand

    def create_net_by_distance(self, dist_matrix: str) -> None:
        data, _ = read_file(dist_matrix, 'dist')
        n = len(data)
        for row in np.arange(n):
            for col in np.arange(row+1, n):
                dist_val = data.iloc[row, col]
                if dist_val > 0:
                    a: Node = self.nodes[row]
                    b: Node = self.nodes[col]
                    self.network.add_edge(a, b, 
                                          dist=dist_val,
                                          dist_demand = a.dist_to_other(b, type='demand_inv', max_demand=self.max_demand))
        self._initialize_paths_and_lengths()

    def create_net_by_voronoi(self) -> None:
        points = [(n.x_coord, n.y_coord) for n in self.nodes]
        vor = Voronoi(points)
        for n in vor.ridge_points:
            a: Node = self.nodes[n[0]]
            b: Node = self.nodes[n[1]]
            self.network.add_edge(a, b, 
                                  dist=a.dist_to_other(b),
                                  dist_demand = a.dist_to_other(b, type='demand_inv', max_demand=self.max_demand))
        self._initialize_paths_and_lengths()

    def extract_distances_by_demand(self):
        pass


    def plot(self, node_label: str|None=None, node_color: str = 'type', ax=None, **kwds) -> None:
        ax = ax or plt.gca()
        pos = dict([(n, (n.x_coord, n.y_coord)) for n in self.network.nodes])
        labels_switch = {
            'index': dict([(n, n.index) for n in self.network.nodes]),
            'id': dict([(n, n.id) for n in self.network.nodes]),
            'demand': dict([(n, n.demand) for n in self.network.nodes]),
            'cluster': dict([(n, n.cluster_idx) for n in self.network.nodes])
        }
        colors_switch = {
            'type': ['#FF0000' if n.is_depot else '#00FF36' for n in self.network.nodes],
            'cluster': [self.cluster_colors.get(n.cluster_idx, '#EEE9E9') for n in self.network.nodes]
        }
        labels = labels_switch.get(node_label) if node_label is not None else None
        colors = colors_switch.get(node_color)

        nx.draw(self.network, pos,
                with_labels=True if node_label else False,
                labels=labels,
                node_color=colors,
                font_size=8, node_size=150, ax=ax, **kwds)

    def _initialize_paths_and_lengths(self) -> None:
        self.distances = pd.DataFrame.from_dict(
            dict(self._shortest_length_matrix())).sort_index()
        self.paths = pd.DataFrame.from_dict(
            dict(self._shortest_path_matrix())).sort_index()
        for node in self.nodes:
            node.distances = self.distances.loc[node]
        self._setup_neighbors()

    def _setup_neighbors(self) -> None:
        for node in self.nodes:
            node.set_neighbors(self.network.neighbors(node))

    def _shortest_path_matrix(self):
        dist = nx.all_pairs_shortest_path(self.network)
        return dist

    def _shortest_length_matrix(self):
        lengths = nx.all_pairs_dijkstra_path_length(self.network, weight='dist_demand')
        return lengths

    def farthest_nodes(self):
        n1 = self.distances.max().idxmax()
        n2 = self.distances.loc[n1].idxmax()
        return (n1, n2)

    # def closest_nodes(self, df_distance: pd.DataFrame):
    #     pass

    def get_distance(self,n1:Node, n2:Node):
        return n1.distances[n2]
        return self.distances.loc[n1,n2]

    def set_colors(self, num_clusters: int = 10):
        # TODO: needs to be dynamic, based on the input number of clusters
        return {
            0: '#FFFF00',
            1: '#8B0000',
            2: '#FF3E96',
            3: '#8B2252',
            4: '#00C78C',
            5: '#CDCD00',
            6: '#00868B',
            7: '#00F5FF',
            8: '#FF6347',
            9: '#FFA54F',
            10: '#4F94CD'
        }


if __name__ == '__main__':
    # from scipy.spatial import Voronoi, voronoi_plot_2d
    print('Running MAP class...')
    map = Map()
    map.create_random_nodes()
    map.create_net_by_voronoi()
    # map.create_map_from_file('sample_data/X-n101-k25.vrp', 'cvrp')
    # map.create_map_from_file('sample_data/X-n5.vrp', 'cvrp')
    # map.import_nodes_from_file('sample_data/X-n15.vrp', 'cvrp')
    # map.create_net_by_distance('sample_data/X-n15.dist')
    # map.create_net_by_voronoi()
    map.plot(node_color='cluster')
    plt.show()
    # print(map.distances)
    # print('ok')
