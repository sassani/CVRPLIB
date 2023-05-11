from map import Map
from cluster import Cluster

# class Solver:
#     def __init__(self) -> None:
#         pass

#     def run(map:Map, num_clusters):
#         pass

def run(map:Map, num_clusters=2):
    clusters=[]
    # TODO: need a function to select init nodes (now it only work with 2 nodes!)
    init_nodes = list(map.farthest_nodes(map.distances))
    for n in range(num_clusters):
        cl:Cluster = Cluster(n)
        cl.add_node(init_nodes[n])
        clusters.append(cl)
        print(clusters[n])

    return clusters 

    