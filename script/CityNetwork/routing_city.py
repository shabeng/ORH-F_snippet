# Heavily based on Sebastian Martin
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from script.CityNetwork.city_creation import create_city


class RoutingCity:
    def __init__(self, network, speeds=(130, 110, 90, 50, 50, 20, 0, 0)):
        self.routing_network = network
        # Times of the existing roads in the city
        self.routing_road_times = np.zeros_like(self.routing_network.get_graph())
        self.calc_road_times(speeds)

        # Times and paths of the shortest route between each pair if nodes
        self.routing_times_matrix = np.zeros_like(self.routing_road_times)
        self.routing_path = np.zeros_like(self.routing_road_times)
        self.calc_time_path_matrix()

    def calc_road_times(self, speeds):
        for (orig_id, dest_id), road in self.routing_network.get_roads().items():
            self.routing_road_times[orig_id, dest_id] = 3.6 * road.get_distance() / speeds[road.get_type()]
        return

    def calc_time_path_matrix(self):
        nodes_num = len(self.routing_network.get_nodes())
        if self.routing_road_times.shape != (nodes_num, nodes_num):
            print('The given timings do not fit the network.')
            return
        for (i, j) in self.routing_network.get_roads().keys():
            if self.routing_road_times[i, j] < 0:
                print(f'{i} => {j} : negative timing {self.routing_road_times[i, j]}')
                return

        graph = csr_matrix(self.routing_road_times)
        for node_id in range(nodes_num):
            dist_matrix, predecessors = dijkstra(csgraph=graph, directed=True, indices=node_id, return_predecessors=True)
            self.routing_times_matrix[node_id, :] = dist_matrix
            self.routing_path[node_id, :] = predecessors


if __name__ == '__main__':
    from PreliminaryExperiments.pocML.solutionCollection.exp8 import system_param_81 as sp8
    from PreliminaryExperiments.pocML.solutionCollection.exp9 import system_param_9 as sp9
    # System 8
    t_ij_8 = sp8.t_ij
    # System 9
    t_ij_9 = sp9.t_ij
    print('Hello World!')
