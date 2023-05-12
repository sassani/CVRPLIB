from map import Map
from cluster import Cluster
from node import Node
import heapq

# TODO: needs a generic solutions for different optimization methods.
# class Solver:
#     def __init__(self) -> None:
#         pass

#     def run(map:Map, num_clusters):
#         pass


def run(map: Map, num_clusters=2):
    clusters = []
    # TODO: At this point we want to keep the overall average of each cluster
    # as close as possible to the overall average of the map.
    # We will add other parameters based on the optimization's target.
    optimum_value = sum([d.demand for d in map.nodes])/map.size
    # heapq.heapify(clusters)
    res_nodes = map.size

    init_nodes = get_init_points(map, num=2)
    for k in range(num_clusters):
        cl: Cluster = Cluster(k)
        cl.add_node(init_nodes[k])
        res_nodes -= 1
        clusters.append(cl)
        # heapq.heappush(clusters, (cl.total_demand, cl))
        # print(clusters[n])

    while (res_nodes > 0):
        # get the list of clusters ordered by total demand
        # l = heapq.nsmallest(num_clusters, clusters)
        clusters.sort(key=lambda c: c.total_demand)
        c: Cluster
        for c in clusters:
            # print(f"id:{c.idx}, tdem:{c.total_demand}")
            # print(best_candidate(c))
            candidate = best_candidate(c, optimum_value)
            if candidate == None:
                continue
            c.add_node(candidate)
            res_nodes -= 1

    return clusters


def get_init_points(map: Map, num: int = 2):
    # TODO: need a function to select init nodes (now it only work with 2 nodes!)
    # It can be depot nodes too.
    return list(map.farthest_nodes(map.distances))


def best_candidate(cl: Cluster, opt_val: float = 0) -> Node:
    scores = []
    # heapq.heapify(scores)
    node: Node
    for node in cl.neighbors:
        if node.cluster_idx >-1:
            continue
        new_val = (cl.total_demand + node.demand)/(len(cl.nodes)+1)
        dist_to_opt = abs(new_val-opt_val)
        mu = [dist_to_opt, node]
        scores.append(mu)
            # heapq.heappush(scores, (mu, node))
    # return heapq.nsmallest(1, scores)
    if len(scores) == 0:
        return None
    scores.sort(key=lambda x: x[0])
    return scores[0][1]
