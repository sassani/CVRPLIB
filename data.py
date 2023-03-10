import re
import numpy as np
import pandas as pd


def read_vrp(path:str):
    meta_data = {}
    data = {}
    with open(path, 'r') as f:
        # Read meta data:
        while True:
            line = f.readline().rstrip('\n')
            key_val = line.split(':')
            if(len(key_val) < 2):  # End of meta data section
                break
            key = re.sub(r"\s+", "", key_val[0])
            string_value = re.search(r"(\".+\")", key_val[1])
            if string_value:
                value = string_value.group().replace("\"", "")
            else:
                value = re.sub(r"\s+", "", key_val[1])
            meta_data[key.lower()] = value

        meta_data['name'] = meta_data.get('name', '')
        meta_data['comment'] = meta_data.get('comment', '')
        meta_data['type'] = meta_data.get('type', '')
        meta_data['edge_weight_type'] = meta_data.get('edge_weight_type', '')
        meta_data['dimension'] = int(meta_data.get('dimension', 0))
        meta_data['capacity'] = int(meta_data.get('capacity', 0))
        
        # Read main data:
        temp = np.array(f.read().splitlines())
        dim = meta_data['dimension']
        data['NODE_COORD'] = temp[0:meta_data['dimension']]
        data['DEMAND'] = temp[dim+1:2*dim+1]
        meta_data['depots'] = temp[2*dim+1:3*dim+1][1:-2]
        # meta_data['depots'] = temp[2*dim+1:3*dim+1][1:-2].astype(int)
        data = pd.DataFrame(data)
        data['ID'] = data.apply(lambda s: (s['NODE_COORD'].split('\t'))[0], axis=1)
        # data['ID'] = data.apply(lambda s: (s['NODE_COORD'].split('\t'))[0], axis=1).astype(int)
        data['NODE_COORD'] =  data.apply(lambda s: tuple(np.array(s['NODE_COORD'].split('\t'))[1:3].astype(float)), axis=1)
        data['DEMAND'] =  data.apply(lambda s: s['DEMAND'].split('\t')[1], axis=1)
        data['IS_DEPOT'] = data.apply(lambda s: True if s['ID'] in meta_data['depots'] else False, axis=1)
        # data = data.set_index('ID')

    return (data, meta_data)


if __name__ == '__main__':
    data, meta_data = read_vrp('X-n1001-k43.vrp')
    print(data)
    print(meta_data)

