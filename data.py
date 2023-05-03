from io import TextIOWrapper
import re
import numpy as np
import pandas as pd
from itertools import product
from random import sample


def read_file(path: str, type: str) -> tuple([pd.DataFrame, dict]):
    match type:
        case 'cvrp':
            return _read_cvrp(path)
        case 'dist':
            return _read_distance(path)
        case _:
            pass


def _read_cvrp(path: str) -> tuple([pd.DataFrame, dict]):
    data = {}
    with open(path, 'r') as f:
        # Read meta data (if any):
        meta_data = _get_metadata(f)
    #
        meta_data['filepath'] = path.split('.')[0]
        meta_data['name'] = meta_data.get('name', '')
        meta_data['comment'] = meta_data.get('comment', '')
        meta_data['type'] = meta_data.get('type', '')
        meta_data['edge_weight_type'] = meta_data.get('edge_weight_type', '')
        meta_data['dimension'] = int(meta_data.get('dimension', 0))
        meta_data['capacity'] = int(meta_data.get('capacity', 0))

        # Read main data:
        temp = np.array(f.read().splitlines())
        dim = meta_data['dimension']
        data['NODE_COORD'] = temp[0:dim]
        data['DEMAND'] = temp[dim+1:2*dim+1]
        meta_data['depots'] = list(
            map(lambda s: s.replace('\t', ''), temp[2*dim+1:3*dim+1][1:-2]))
        data = pd.DataFrame(data)
        data['ID'] = data.apply(lambda s: (
            s['NODE_COORD'].split('\t'))[0], axis=1)
        data['NODE_COORD'] = data.apply(lambda s: tuple(
            np.array(s['NODE_COORD'].split('\t'))[1:3].astype(float)), axis=1)
        data['DEMAND'] = data.apply(
            lambda s: s['DEMAND'].split('\t')[1], axis=1)
        data['IS_DEPOT'] = data.apply(
            lambda s: True if s['ID'] in meta_data['depots'] else False, axis=1)
        # data = data.set_index('ID')

    return (data, meta_data)


def _get_metadata(f: TextIOWrapper):
    meta_data = {}
    while True:
        line = f.readline().rstrip('\n')
        key_val = line.split(':')
        if (len(key_val) < 2):  # End of meta data section
            break
        key = re.sub(r"\s+", "", key_val[0])
        string_value = re.search(r"(\".+\")", key_val[1])
        if string_value:
            value = string_value.group().replace("\"", "")
        else:
            value = re.sub(r"\s+", "", key_val[1])
        meta_data[key.lower()] = value
    return meta_data


def _read_distance(path: str) -> tuple([pd.DataFrame, dict]):
    meta_data = {}
    data = {}
    with open(path, 'r') as f:
        data = pd.read_csv(path, delimiter='\t', header=0, index_col=0)

    return (data, meta_data)


def _create_grid(width: int = 10, height: int = 10):
    node_coord = []
    for y in np.arange(height):
        for x in np.arange(width):
            node_coord.append((x, y))
    return node_coord


def create_random_cvrp(width: int = 10, height: int = 10, dem_min: int = 1, dem_max: int = 100, n: int = 10, capacity=None, is_grided=False):
    node_coord = []
    if is_grided:
        node_coord = _create_grid(width, height)
        n = width * height
    else:
        node_coord = sample(
            list(product(np.random.uniform(0,width,n), np.random.uniform(0,height,n), repeat=1)), k=n)

    meta_data = {}
    meta_data['filepath'] = 'memory'
    meta_data['name'] = f'X-n{n}k'
    meta_data['comment'] = 'This grid was made randomly'
    meta_data['type'] = 'CVRP'
    meta_data['edge_weight_type'] = 'EUC_2D'
    meta_data['dimension'] = n

    demand = np.random.choice(np.arange(dem_min, dem_max+1), size=n)
    if (capacity is None):
        capacity = demand.mean()*4
    meta_data['capacity'] = capacity

    data = pd.DataFrame(np.arange(1, n+1), columns=['ID'])
    data['NODE_COORD'] = node_coord
    data['DEMAND'] = demand
    # TODO: randomize this column
    data['IS_DEPOT'] = np.full(n, False)
    return (data, meta_data)


if __name__ == '__main__':
    data, meta_data = read_file('sample_data/X-n1001-k43.vrp', 'cvrp')
    print(data)
    print(meta_data)

    g = create_random_cvrp(n=1500)
    print(g)
    # g = Generators.create_random_grid()
    # print(g)
    # distance, _ = read_file('sample_data/X-n15.dist', 'dist')
    # print(distance)
