from map import Map
from cluster import Cluster
from node import Node
import heapq
import numpy as np


# TODO: needs a generic solutions for different optimization methods.
# class Solver:
#     def __init__(self) -> None:
#         pass

#     def run(map:Map, num_clusters):
#         pass


def run(map: Map, num_clusters=2, cluster_type='random'):
    _reset_nodes(map)
    clusters = []
    # TODO: At this point we want to keep the overall average of each cluster
    # as close as possible to the overall average of the map.
    # We will add other parameters based on the optimization's target.
    optimum_value = sum([d.demand for d in map.nodes])/map.size
    # heapq.heapify(clusters)
    res_nodes = map.size

    init_nodes = _get_init_points(map, k=num_clusters, type=cluster_type)

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
            candidate = _best_candidate(c, optimum_value)
            if candidate == None:
                continue
            c.add_node(candidate)
            res_nodes -= 1

    return clusters

def _reset_nodes(map):
    for n in map.nodes:
        n.reset_cluster()

def __random_selectgion(map:Map, k:int)->list[Node]:
    rng = np.random.default_rng(seed=map.seed)
    return list(rng.choice(map.nodes, size=k, replace=False))

def __clusters_medoids(map: Map, k):
    medoids:list[Node]=[]
    # pass
#     if k == 2:
#         return list(map.farthest_nodes())
#     medoids:list[Node] = __random_selectgion()
#     E=1e-3
#     err = 1
#     iteration = 1
#     clusters:dict[int, Cluster] = {}
#     while err>E:# repeat until no change for new medoids
#         for n in np.arange(k):# initialize k clusters containers
#             clusters[n]=[]
            
#         for point in map.nodes: # candidate all points inside the cluster i
#             dists=[]
#             for c in clusters: # calculate the distance between all points with the candidate 
#                 dists.append(map.get_distance(point, medoids))
#             clusters[np.argmin(dists)].append(point) # select the candidate point with the minimum sum distance
        
#         if(verbose):print(f"Iteration {iteration}")
#         temp_cent = get_medoids(clusters, display=verbose)
#         errs=[]
#         for i in np.arange(k): # to find the maximum change between old and new medoids
#             errs.append(distance_euc(temp_cent[i],medoids[i]))
#         err = np.max(errs)# get the maximum error (distance)
#         medoids = temp_cent
#         iteration+=1
    return medoids


def _get_init_points(map: Map, k: int = 2, type='random') -> list[Node]:
    # TODO: need a function to select init nodes (now it only work with 2 nodes!)
    # It can be depot nodes too.
    match type:
        case 'random':
            return __random_selectgion(map, k)
        case 'k_medoids':
            return __clusters_medoids(map, k)
        case _:
            return __random_selectgion(map, k)


def _best_candidate(cl: Cluster, opt_val: float = 0) -> Node|None:
    scores = []
    # heapq.heapify(scores)
    node: Node
    for node in cl.neighbors:
        if node.cluster_idx > -1:
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


# def maximal_dispersion(nodes:list(Node), num_points, num_iterations):
#     # Generate initial random points
#     points = np.random.rand(num_points, 2)  # 2D plane

#     for _ in range(num_iterations):
#         # Assign points to the nearest cluster centroid
#         distances = np.sqrt(np.sum((points[:, np.newaxis] - points) ** 2, axis=2))
#         assignments = np.argmin(distances, axis=1)

#         # Update cluster centroids
#         for i in range(num_points):
#             points[i] = np.mean(points[assignments == i], axis=0)

#     return points
