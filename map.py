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
        self.set_colors()

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
                            is_grided=False, seed=None) -> None:
        nodes, meta_data = create_random_cvrp(width, height, dem_min,
                                              dem_max, n, capacity, is_grided, seed)
        nodes.loc[0, 'IS_DEPOT'] = True
        self._import_nodes_from_df(nodes)
        self.nodes_df = nodes
        self.info = meta_data

    def _import_nodes_from_df(self, df: pd.DataFrame) -> None:
        for item in df.iterrows():
            node = item[1]
            node = Node(item[0], node['ID'], node['NODE_COORD'][0],
                        node['NODE_COORD'][1], node['DEMAND'], node['IS_DEPOT'])
            self.network.add_node(node,
                                  pos=(node.x_coord, node.y_coord),
                                  color='#FF0000'if node.is_depot else '#00FF36',
                                  demand=node.demand,
                                  id=node.id)
            self.nodes.append(node)
        self.size = len(self.nodes)

    def create_net_by_distance(self, dist_matrix: str) -> None:
        data, _ = read_file(dist_matrix, 'dist')
        n = len(data)
        for row in np.arange(n):
            for col in np.arange(row+1, n):
                dist_val = data.iloc[row, col]
                if dist_val > 0:
                    self.network.add_edge(
                        self.nodes[row], self.nodes[col], weight=dist_val)
        self._initialize_paths_and_lengths()

    def create_net_by_voronoi(self) -> None:
        points = [(n.x_coord, n.y_coord) for n in self.nodes]
        vor = Voronoi(points)
        for n in vor.ridge_points:
            a: Node = self.nodes[n[0]]
            b: Node = self.nodes[n[1]]
            self.network.add_edge(a, b, weight=a.dist_to_other(b))
        self._initialize_paths_and_lengths()

    def plot(self, node_label: str = None, node_color: str = 'type', ax=None, **kwds) -> None:
        ax = ax or plt.gca()
        pos = dict([(n, (n.x_coord, n.y_coord)) for n in self.network.nodes])
        labels_switch = {
            'id': dict([(n, n.id) for n in self.network.nodes]),
            'demand': dict([(n, n.demand) for n in self.network.nodes]),
            'cluster': dict([(n, n.cluster_idx) for n in self.network.nodes])
        }
        colors_switch = {
            'type': ['#FF0000' if n.is_depot else '#00FF36' for n in self.network.nodes],
            'cluster': [self.cluster_colors.get(n.cluster_idx, '#EEE9E9') for n in self.network.nodes]
        }
        labels = labels_switch.get(node_label)
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
        self._setup_neighbors()

    def _setup_neighbors(self) -> None:
        for node in self.nodes:
            node.set_neighbors(self.network.neighbors(node))

    def _shortest_path_matrix(self):
        dist = nx.all_pairs_shortest_path(self.network)
        return dist

    def _shortest_length_matrix(self) -> float:
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.network))
        return lengths

    def farthest_nodes(self, df_distance: pd.DataFrame) -> tuple[Node, Node]:
        n1: Node = df_distance.max().idxmax()
        n2: Node = df_distance.loc[n1].idxmax()
        return (n1, n2)

    def closest_nodes(self, df_distance: pd.DataFrame):
        pass

    def set_colors(self, num_clusters: int = 10):
        # TODO: needs to be dynamic, based on the input number of clusters
        self.cluster_colors = {
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
    map.create_random_nodes(seed=10)
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
