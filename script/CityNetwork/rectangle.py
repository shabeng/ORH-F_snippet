# Heavily based on Sebastian Martin
import numpy as np


class Node:
    def __init__(self, id_num, x_cor, y_cor):
        """
        Object of a node in a network.
        :param id_num: int, location id which is the index of the location in the nodes list of the network (converted from
        the (i,j) tuple of the location position in the rectangle block). For example -
        rectangle positions:        location ids:
        (0,0)   (0,1)               0   1
                            ==>
        (1,0)   (1,1)               2   3
        :param x_cor: float, the x coordinate on the axis plot of the block
        :param y_cor: float, the y coordinate on the axis plot of the block
        """
        self.n_loc_id = id_num
        self.n_x = x_cor
        self.n_y = y_cor

    def get_coor(self):
        return np.array([self.n_x, self.n_y])

    def get_x_coor(self):
        return self.n_x

    def get_y_coor(self):
        return self.n_y

    def get_loc_id(self):
        return self.n_loc_id


class Road:
    def __init__(self, orig_node_id, dest_node_id, road_distance, road_type):
        """
        Object of a road in a network, like an edge with distance and direction.
        :param orig_node_id: int, the location id of the starting location of the road
        :param dest_node_id: int, the location id of the ending location of the road
        :param road_distance: float, Euclidean distance between the coordinates of the locations
        :param road_type: int, specify the speed in the road
        """
        self.r_orig_loc = orig_node_id
        self.r_dest_loc = dest_node_id
        self.r_distance = road_distance
        self.r_type = road_type

    def get_locations(self):
        return self.r_orig_loc, self.r_dest_loc

    def get_distance(self):
        return self.r_distance

    def get_type(self):
        return self.r_type


class Network:
    def __init__(self, nodes_lst, roads_dict):
        self.net_nodes = nodes_lst
        self.net_roads = roads_dict
        self.net_graph = np.zeros((len(nodes_lst), len(nodes_lst)))
        for (loc_orig, loc_dest), road in roads_dict.items():  # (loc_id_orig, loc_id_dest): Road object
            self.net_graph[loc_orig, loc_dest] = road.r_distance

    def get_nodes(self):
        return self.net_nodes

    def get_node_by_id(self, id_num):
        return self.net_nodes[id_num]

    def get_roads(self):
        return self.net_roads

    def get_graph(self):
        return self.net_graph

    def add_node(self, node_obj):
        loc_id = node_obj.get_loc_id()
        self.net_nodes[loc_id] = node_obj

    def add_road(self, road_obj):
        self.net_roads[road_obj.get_locations()] = road_obj

    def resize_nodes(self, new_nodes_num):
        additional_nodes = new_nodes_num - len(self.get_nodes())
        self.net_nodes = self.get_nodes() + [0] * additional_nodes


def create_rectangle_network(width, height, distance_x, distance_y):
    def cor_to_loc(rec_i, rec_j):
        """
        Function to convert from the rectangle position of a location to the location id like in the network nodes list.
        :param rec_i: int, the row index in the rectangle, starts with 0
        :param rec_j: int, the column index in the rectangle, starts with 0
        :return:
        """
        return rec_j + rec_i * width

    # Create nodes
    nodes_num = width * height
    nodes_lst = [0] * nodes_num
    for i in range(height):
        for j in range(width):
            loc_id = cor_to_loc(i, j)
            x = ((j + 1) - ((width + 1) / 2)) * distance_x
            y = (((height + 1) / 2) - (i + 1)) * distance_y
            nodes_lst[loc_id] = Node(loc_id, x, y)

    # Create roads
    roads_dict = {}
    for i in range(height):
        for j in range(width):
            for k in range(2):
                if k == 0 and i != (height - 1):
                    # Vertical roads: up-down
                    loc_id_1 = cor_to_loc(i, j)
                    loc_id_2 = cor_to_loc(i + 1, j)

                elif k == 1 and j != (width - 1):
                    # Horizontal roads: left-right
                    loc_id_1 = cor_to_loc(i, j)
                    loc_id_2 = cor_to_loc(i, j + 1)

                else:
                    continue

                node_coor_1 = nodes_lst[loc_id_1].get_coor()
                node_coor_2 = nodes_lst[loc_id_2].get_coor()
                road_distance = np.linalg.norm(node_coor_2 - node_coor_1, ord=2)
                roads_dict[(loc_id_1, loc_id_2)] = Road(loc_id_1, loc_id_2, road_distance, 5 - 1)
                roads_dict[(loc_id_2, loc_id_1)] = Road(loc_id_2, loc_id_1, road_distance, 5 - 1)

    return Network(nodes_lst, roads_dict)


def create_square_network(width, distance):
    return create_rectangle_network(width=width, height=width, distance_x=distance, distance_y=distance)


if __name__ == '__main__':
    squr_network = create_square_network(width=2, distance=200)
    rec_network = create_rectangle_network(width=2, height=3, distance_x=200, distance_y=200)
