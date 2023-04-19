# Heavily based on Sebastian Martin
from script.CityNetwork.rectangle import create_square_network, Node, Road, Network
from script.utilis.utilis_plot import color_rgbs_range

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# np.random.seed(146)


def cor_to_loc(rec_num, width, sub_width, rec_i=None, rec_j=None, rec_loc_id=None):
    """
    Function to convert from the rectangle position of a location to the location id like in the network nodes list.
    :param rec_i: int, the row index in the rectangle, starts with 0
    :param rec_j: int, the column index in the rectangle, starts with 0
    :param rec_loc_id: int, location id in the rectangle
    :param rec_num: int, the rectangle index in the city layout
    :param width
    :param sub_width
    :return:
    """
    # city center rectangle
    move_add = 0
    if rec_num != 0:
        # city suburbs rectangle
        move_add = width ** 2 + (rec_num - 1) * (sub_width ** 2)

    if rec_loc_id is not None:
        return move_add + rec_loc_id

    else:
        return move_add + rec_j + rec_i * sub_width if rec_num != 0 else rec_j + rec_i * width


def loc_to_rec_num(loc_num, width=8, sub_width=4):
    d = {0: range(width**2)}
    for i in range(1, 9):
        d[i] = range(d.get(i - 1, None)[-1] + 1, d.get(i - 1, None)[-1] + 1 + sub_width ** 2)
    for key, locations in d.items():
        if loc_num in locations:
            return key


def loc_to_rec_num_sub_center(loc_num, width=8, sub_width=4):
    rec_num = loc_to_rec_num(loc_num, width=width, sub_width=sub_width)
    if rec_num == 0:
        arr = np.arange(width**2).reshape((width, width))
        i = 0
        for x in range(2):
            for y in range(2):
                if loc_num in arr[4 * x:4 * (x + 1), 4 * y:4 * (y + 1)]:
                    return i
                else:
                    i += 1
    else:
        return rec_num + 3


def create_city(width, distance):
    np.random.seed(146)
    sub_width = int(width / 2)

    # square cities:
    city = create_square_network(width=width, distance=distance)
    suburbs = [create_square_network(width=sub_width, distance=distance) for i in range(8)]
    nodes_num = (width ** 2) + 8 * (sub_width ** 2)
    city.resize_nodes(nodes_num)
    '''
    4   3   2
    5   0   1
    6   7   8
    '''
    # Reposition the adjacent suburbs around the city center:
    for (sub_rec_num, direction_x, direction_y) in [(1, 1, 0), (3, 0, 1), (5, -1, 0), (7, 0, -1)]:
        # Add the nodes
        offset = np.random.rand() * width * 0.55
        sub_rec_net_obj = suburbs[sub_rec_num - 1]
        for node in sub_rec_net_obj.get_nodes():
            x_new = node.get_x_coor() + direction_x * distance * ((width - 1) / 2 + (sub_width + 1) / 2 + offset)
            y_new = node.get_y_coor() + direction_y * distance * ((width - 1) / 2 + (sub_width + 1) / 2 + offset)
            loc_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=node.get_loc_id())
            city.add_node(Node(id_num=loc_id_new, x_cor=x_new, y_cor=y_new))
        # Add the roads
        for (loc_orig, loc_dest), road in sub_rec_net_obj.get_roads().items():  # (locIdOrig, locIdDest): Road obj
            loc_orig_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_orig)
            loc_dest_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_dest)
            city.add_road(Road(loc_orig_id_new, loc_dest_id_new, road.get_distance(), road.get_type()))

    # Reposition the diagonal suburbs around the city center:
    for (sub_rec_num, direction_x, direction_y) in [(2, 1, 1), (4, -1, 1), (6, -1, -1), (8, 1, -1)]:
        # Add the nodes
        offset = np.random.rand() * width * 0.3
        sub_rec_net_obj = suburbs[sub_rec_num - 1]
        for node in sub_rec_net_obj.get_nodes():
            x, y = node.get_coor()
            x_new = (x - y) / (2**0.5) + \
                    direction_x * distance * ((width - 1) / 2 + offset + (sub_width + 1) / (2 * 2**0.5))
            y_new = (y + x) / (2**0.5) + \
                    direction_y * distance * ((width - 1) / 2 + offset + (sub_width + 1) / (2 * 2**0.5))
            loc_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=node.get_loc_id())
            city.add_node(Node(id_num=loc_id_new, x_cor=x_new, y_cor=y_new))
        # Add the roads
        for (loc_orig, loc_dest), road in sub_rec_net_obj.get_roads().items():  # (locIdOrig, locIdDest): Road obj
            loc_orig_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_orig)
            loc_dest_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_dest)
            city.add_road(Road(loc_orig_id_new, loc_dest_id_new, road.get_distance(), road.get_type()))

    # Add highway roads between city and suburbs
    connections = [  # city1, i1, j1, city2, i2, j2, type
        # perimeter - links between suburbs
        (1, 1 - 1, int(sub_width / 2) - 1, 2, sub_width - 1, int(sub_width / 2) - 1, 2 - 1),
        (2, 1 - 1, int(sub_width / 2) - 1, 3, int(sub_width / 2) - 1, sub_width - 1, 2 - 1),
        (3, int(sub_width / 2) - 1, 1 - 1, 4, int(sub_width / 2) - 1, sub_width - 1, 2 - 1),
        (4, int(sub_width / 2) - 1, 1 - 1, 5, 1 - 1, int(sub_width / 2) - 1, 2 - 1),
        (5, sub_width - 1, int(sub_width / 2) - 1, 6, 1 - 1, int(sub_width / 2) - 1, 2 - 1),
        (6, sub_width - 1, int(sub_width / 2) - 1, 7, int(sub_width / 2) - 1, 1 - 1, 2 - 1),
        (7, int(sub_width / 2) - 1, sub_width - 1, 8, int(sub_width / 2) - 1, 1 - 1, 2 - 1),
        (8, int(sub_width / 2) - 1, sub_width - 1, 1, sub_width - 1, int(sub_width / 2) - 1, 2 - 1),
        # link to city center from every suburb:
        (1, np.random.choice(range(sub_width)), 1 - 1, 0, np.random.choice(range(width)), width - 1, 2 - 1),
        (2, np.random.choice(range(sub_width)), 1 - 1, 0, 1 - 1, width - 1, 2 - 1),
        (3, sub_width - 1, np.random.choice(range(sub_width)), 0, 1 - 1, np.random.choice(range(width)), 2 - 1),
        (4, sub_width - 1, np.random.choice(range(sub_width)), 0, 1 - 1, 1 - 1, 2 - 1),
        (5, np.random.choice(range(sub_width)), sub_width - 1, 0, np.random.choice(range(width)), 1 - 1, 2 - 1),
        (6, np.random.choice(range(sub_width)), sub_width - 1, 0, width - 1, 1 - 1, 2 - 1),
        (7, 1 - 1, np.random.choice(range(sub_width)), 0, width - 1, np.random.choice(range(width)), 2 - 1),
        (8, 1 - 1, np.random.choice(range(sub_width)), 0, width - 1, width - 1, 2 - 1)
    ]
    # dist_between_os = 0
    # dist_between_oc = 0
    for (rec_num1, rec_i1, rec_j1, rec_num2, rec_i2, rec_j2, road_type) in connections:
        loc1_id = cor_to_loc(rec_num1, width, sub_width, rec_i=rec_i1, rec_j=rec_j1)
        loc2_id = cor_to_loc(rec_num2, width, sub_width, rec_i=rec_i2, rec_j=rec_j2)
        node_coor_1 = city.get_node_by_id(loc1_id).get_coor()
        node_coor_2 = city.get_node_by_id(loc2_id).get_coor()
        road_distance = np.linalg.norm(node_coor_2 - node_coor_1, ord=2)
        city.add_road(Road(loc1_id, loc2_id, road_distance, road_type))
        city.add_road(Road(loc2_id, loc1_id, road_distance, road_type))
        # if rec_num2 != 0:
        #     dist_between_os += road_distance
        # else:
        #     dist_between_oc += road_distance
        # print(f'City {rec_num1} and City {rec_num2}: distance = {road_distance}')
    # print(f'Between outskirts distance: {dist_between_os / 8} | Between center outskirts: {dist_between_oc / 8}')
    # print(f'Between outskirts time: {3.6 * (dist_between_os / 8) / 55} | Between center outskirts time: {3.6 * (dist_between_oc / 8) / 55}')

    # Add highway around city center
    for i in range(width - 1):  # row
        for j in [1 - 1, width - 1]:  # column
            for k in range(2):
                if k == 0:
                    # hw top-down
                    loc1_id = cor_to_loc(0, width, sub_width, rec_i=i, rec_j=j)
                    loc2_id = cor_to_loc(0, width, sub_width, rec_i=i + 1, rec_j=j)
                else:
                    # hw left-right
                    loc1_id = cor_to_loc(0, width, sub_width, rec_i=j, rec_j=i)
                    loc2_id = cor_to_loc(0, width, sub_width, rec_i=j, rec_j=i + 1)
                node_coor_1 = city.get_node_by_id(loc1_id).get_coor()
                node_coor_2 = city.get_node_by_id(loc2_id).get_coor()
                road_distance = np.linalg.norm(node_coor_2 - node_coor_1, ord=2)
                city.add_road(Road(loc1_id, loc2_id, road_distance, 1 - 1))
                city.add_road(Road(loc2_id, loc1_id, road_distance, 1 - 1))

    return Network(city.get_nodes(), city.get_roads())


def create_symmetric_city(width, distance):
    sub_width = int(width / 2)

    # square cities:
    city = create_square_network(width=width, distance=distance)
    suburbs = [create_square_network(width=sub_width, distance=distance) for i in range(8)]
    nodes_num = (width ** 2) + 8 * (sub_width ** 2)
    city.resize_nodes(nodes_num)
    '''
    4   3   2
    5   0   1
    6   7   8
    '''
    # Reposition the adjacent suburbs around the city center:
    for (sub_rec_num, direction_x, direction_y) in [(1, 1, 0), (3, 0, 1), (5, -1, 0), (7, 0, -1)]:
        # Add the nodes
        offset = width * 0.24
        sub_rec_net_obj = suburbs[sub_rec_num - 1]
        for node in sub_rec_net_obj.get_nodes():
            x_new = node.get_x_coor() + direction_x * distance * ((width - 1) / 2 + (sub_width + 1) / 2 + offset)
            y_new = node.get_y_coor() + direction_y * distance * ((width - 1) / 2 + (sub_width + 1) / 2 + offset)
            loc_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=node.get_loc_id())
            city.add_node(Node(id_num=loc_id_new, x_cor=x_new, y_cor=y_new))
        # Add the roads
        for (loc_orig, loc_dest), road in sub_rec_net_obj.get_roads().items():  # (locIdOrig, locIdDest): Road obj
            loc_orig_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_orig)
            loc_dest_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_dest)
            city.add_road(Road(loc_orig_id_new, loc_dest_id_new, road.get_distance(), road.get_type()))

    # Reposition the diagonal suburbs around the city center:
    for (sub_rec_num, direction_x, direction_y) in [(2, 1, 1), (4, -1, 1), (6, -1, -1), (8, 1, -1)]:
        # Add the nodes
        offset = width * 0.15
        sub_rec_net_obj = suburbs[sub_rec_num - 1]
        for node in sub_rec_net_obj.get_nodes():
            x, y = node.get_coor()
            x_new = (x - y) / (2**0.5) + \
                    direction_x * distance * ((width - 1) / 2 + offset + (sub_width + 1) / (2 * 2**0.5))
            y_new = (y + x) / (2**0.5) + \
                    direction_y * distance * ((width - 1) / 2 + offset + (sub_width + 1) / (2 * 2**0.5))
            loc_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=node.get_loc_id())
            city.add_node(Node(id_num=loc_id_new, x_cor=x_new, y_cor=y_new))
        # Add the roads
        for (loc_orig, loc_dest), road in sub_rec_net_obj.get_roads().items():  # (locIdOrig, locIdDest): Road obj
            loc_orig_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_orig)
            loc_dest_id_new = cor_to_loc(sub_rec_num, width, sub_width, rec_loc_id=loc_dest)
            city.add_road(Road(loc_orig_id_new, loc_dest_id_new, road.get_distance(), road.get_type()))

    # Add highway roads between city and suburbs
    connections = [  # city1, i1, j1, city2, i2, j2, type
        # perimeter - links between suburbs
        (1, 1 - 1, int(sub_width / 2), 2, sub_width - 1, int(sub_width / 2), 2 - 1),
        (2, 1 - 1, int(sub_width / 2), 3, int(sub_width / 2) - 1, sub_width - 1, 2 - 1),
        (3, int(sub_width / 2) - 1, 1 - 1, 4, int(sub_width / 2) - 1, sub_width - 1, 2 - 1),
        (4, int(sub_width / 2) - 1, 1 - 1, 5, 1 - 1, int(sub_width / 2) - 1, 2 - 1),
        (5, sub_width - 1, int(sub_width / 2) - 1, 6, 1 - 1, int(sub_width / 2) - 1, 2 - 1),
        (6, sub_width - 1, int(sub_width / 2) - 1, 7, int(sub_width / 2), 1 - 1, 2 - 1),
        (7, int(sub_width / 2), sub_width - 1, 8, int(sub_width / 2), 1 - 1, 2 - 1),
        (8, int(sub_width / 2), sub_width - 1, 1, sub_width - 1, int(sub_width / 2), 2 - 1),
        # link to city center from every suburb:
        (1, sub_width - 1, 1 - 1, 0, int(width / 2), width - 1, 2 - 1),
        (2, sub_width - 1, 1 - 1, 0, 1 - 1, width - 1, 2 - 1),
        (3, sub_width - 1, sub_width - 1, 0, 1 - 1, int(width / 2), 2 - 1),
        (4, sub_width - 1, sub_width - 1, 0, 1 - 1, 1 - 1, 2 - 1),
        (5, 1 - 1, sub_width - 1, 0, int(width / 2) - 1, 1 - 1, 2 - 1),
        (6, 1 - 1, sub_width - 1, 0, width - 1, 1 - 1, 2 - 1),
        (7, 1 - 1, 1 - 1, 0, width - 1, int(width / 2) - 1, 2 - 1),
        (8, 1 - 1, 1 - 1, 0, width - 1, width - 1, 2 - 1)
    ]

    # dist_between_os = 0
    # dist_between_oc = 0
    for (rec_num1, rec_i1, rec_j1, rec_num2, rec_i2, rec_j2, road_type) in connections:
        loc1_id = cor_to_loc(rec_num1, width, sub_width, rec_i=rec_i1, rec_j=rec_j1)
        loc2_id = cor_to_loc(rec_num2, width, sub_width, rec_i=rec_i2, rec_j=rec_j2)
        node_coor_1 = city.get_node_by_id(loc1_id).get_coor()
        node_coor_2 = city.get_node_by_id(loc2_id).get_coor()
        road_distance = np.linalg.norm(node_coor_2 - node_coor_1, ord=2)
        city.add_road(Road(loc1_id, loc2_id, road_distance, road_type))
        city.add_road(Road(loc2_id, loc1_id, road_distance, road_type))
        # if rec_num2 != 0:
        #     dist_between_os += road_distance
        # else:
        #     dist_between_oc += road_distance
        # print(f'City {rec_num1} and City {rec_num2}: distance = {road_distance}')
    # print(f'Between outskirts distance: {dist_between_os / 8} | Between center outskirts: {dist_between_oc / 8}')
    # print(f'Between outskirts time: {3.6*(dist_between_os / 8)/55} | Between center outskirts time: {3.6*(dist_between_oc / 8)/55}')

    # Add highway around city center
    for i in range(width - 1):  # row
        for j in [1 - 1, width - 1]:  # column
            for k in range(2):
                if k == 0:
                    # hw top-down
                    loc1_id = cor_to_loc(0, width, sub_width, rec_i=i, rec_j=j)
                    loc2_id = cor_to_loc(0, width, sub_width, rec_i=i + 1, rec_j=j)
                else:
                    # hw left-right
                    loc1_id = cor_to_loc(0, width, sub_width, rec_i=j, rec_j=i)
                    loc2_id = cor_to_loc(0, width, sub_width, rec_i=j, rec_j=i + 1)
                node_coor_1 = city.get_node_by_id(loc1_id).get_coor()
                node_coor_2 = city.get_node_by_id(loc2_id).get_coor()
                road_distance = np.linalg.norm(node_coor_2 - node_coor_1, ord=2)
                city.add_road(Road(loc1_id, loc2_id, road_distance, 1 - 1))
                city.add_road(Road(loc2_id, loc1_id, road_distance, 1 - 1))

    return Network(city.get_nodes(), city.get_roads())


def get_zones_xys_coor(net):
    x_coors, y_coors = [], []
    for node in net.get_nodes():
        if node != 0:
            x_coors.append(node.get_x_coor())
            y_coors.append(node.get_y_coor())
    return x_coors, y_coors


def plot_city(network, plot_roads=True, plot_areas=False, fig_size=None, node_c='blue', road_c='orange', area_c='green'):
    """
    Create a city visualization given a network instance.
    :param network: Network instance with roads and nodes dictionaries
    :param plot_roads
    :param plot_areas
    :param fig_size
    :param node_c
    :param road_c
    :param area_c
    :return: plt of the city
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()
    x_vals, y_vals = get_zones_xys_coor(network)
    ax.scatter(x_vals, y_vals, c=node_c)

    if plot_roads:
        ax = plot_city_roads(ax, network, road_c)

    if plot_areas:
        ax = plot_city_areas(ax, network, area_c)
    return fig, ax


def plot_city_roads(ax, network, color):
    """
    Create a city visualization given a network instance.
    :param ax
    :param network: Network instance with roads and nodes dictionaries
    :param color
    :return: plt of the city
    """
    for (orig_id, dest_id), road in network.get_roads().items():
        node_coor_1 = network.get_node_by_id(orig_id).get_coor()
        node_coor_2 = network.get_node_by_id(dest_id).get_coor()
        ax.plot([node_coor_1[0], node_coor_2[0]], [node_coor_1[1], node_coor_2[1]], c=color)
    return ax


def plot_city_areas(ax, net, color):
    """
    Create a city visualization given a network instance.
    :param ax
    :param net: Network instance with roads and nodes dictionaries
    :param color
    :return: plt of the city
    """
    nodes_center_lst = [24, 28, 56, 60]
    nodes_outskirts_lst = [76, 92, 108, 124, 140, 156, 172, 188]
    for c, node_ind in enumerate(nodes_center_lst):
        # Create a Rectangle patch
        mov_x = (-300) if c % 2 == 0 else (-400)
        x_len = 770 if c % 2 == 0 else 762.5
        mov_y = (-400) if c % 2 == 0 else (-400)
        y_len = 800 if c % 2 == 0 else 800
        rect = patches.Rectangle((net.get_nodes()[node_ind].get_x_coor() + mov_x,
                                  net.get_nodes()[node_ind].get_y_coor() + mov_y),
                                 x_len * 4, y_len * 4, linewidth=1, edgecolor=color, facecolor='none',
                                 linestyle='-')
        # Add the patch to the Axes
        ax.add_patch(rect)
    for j, node_ind in enumerate(nodes_outskirts_lst):
        angle = 0 if j % 2 == 0 else 45
        shift_y = 0 if j % 2 == 0 else 200 if node_ind != 188 else 180
        shift_x = shift_y * 2 if node_ind != 188 else shift_y * 2 + 80
        rec_len_x = 800 if node_ind != 188 else 780
        rec_len_y = 800 if node_ind != 188 else 780
        # Create a Rectangle patch
        rect = patches.Rectangle((net.get_nodes()[node_ind].get_x_coor() - 400 + shift_x,
                                  net.get_nodes()[node_ind].get_y_coor() - 400 - shift_y),
                                 rec_len_x * 4, rec_len_y * 4, angle, linewidth=1, edgecolor=color, facecolor='none',
                                 linestyle='--')
        # Add the patch to the Axes
        ax.add_patch(rect)
    return ax


def plot_route(ax, network, system, veh_id, route, is_time_color, color_bins):
    """
    Get an ax with the city (nodes with/without roads) and plot the route of one vehicle (several vehicles together is
    too messy).
    arrows? https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib
    :param ax: Axes instance
    :param network: Network instance of the city
    :param system: System instance that correspond to the given route
    :param veh_id: int of the vehicle serves the route
    :param route: list of int, each int is a request id number
    :param is_time_color: boolean, whether to use a different shade of blue that match the time
    :param color_bins: np.array, specifies the division of the time horizon to bins (for the colors)
    :return:
    """
    z = system.sys_vehs[veh_id - 1].get_zone()
    colors_vals = color_rgbs_range(num=color_bins.shape[0] + 1)  # +1 so we wont get too light color for the last bin
    for req in route:
        req_obj = system.sys_reqs[req - 1]
        t_k = req_obj.get_arrival_time()
        if is_time_color:
            c_ind = np.max(np.where(color_bins <= t_k)[0])
            c = colors_vals[c_ind]
        else:
            c = 'blue'
        o_k, d_k = req_obj.get_od_zones()
        ax.plot([network.get_node_by_id(z).get_x_coor(), network.get_node_by_id(o_k).get_x_coor()],
                [network.get_node_by_id(z).get_y_coor(), network.get_node_by_id(o_k).get_y_coor()],
                linestyle='--', color=c)
        ax.plot([network.get_node_by_id(o_k).get_x_coor(), network.get_node_by_id(d_k).get_x_coor()],
                [network.get_node_by_id(o_k).get_y_coor(), network.get_node_by_id(d_k).get_y_coor()],
                color=c)
        z = d_k
    ax.set_title(f'Vehicle {veh_id}')
    return ax


if __name__ == '__main__':
    city_network = create_city(width=8, distance=800)
    print('hello world!')

    # Plot the network - nodes + roads
    fig, _ = plot_city(city_network, plot_areas=True, fig_size=(5, 4))
    axis = plt.gca()
    axis.set_xlim(-9000, 9000)
    axis.set_ylim(-9000, 9000)
    plt.title('Original City')
    axis.axes.xaxis.set_visible(False)
    axis.axes.yaxis.set_visible(False)
    plt.savefig('/Users/shabeng/Documents/Education/University/Research/FairnessSharingSys/Paper'
                '/city_network_with_areas_not_collapse.png', dpi=1200)
    plt.show()

    # Plot nodes only
    # ax = plot_city(city_network, plot_roads=False)
    # plt.show()

    # Symmetric City
    city_symmetric_network = create_symmetric_city(width=8, distance=800)
    print('hello world!')

    # Plot the network - nodes + roads
    fig, _ = plot_city(city_symmetric_network, plot_areas=True)
    # plt.savefig('/Users/shabeng/Documents/Education/University/Research/FairnessSharingSys/PaperThesis/'
    #             'Experiments/Pass I/Graphs/city_network_with_areas_not_collapse.png', dpi=1200)
    axis = plt.gca()
    axis.set_xlim(-9000, 9000)
    axis.set_ylim(-9000, 9000)
    plt.title('Symmetric City')
    plt.show()

    # Plot the two system side by side
    f, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6.4*1.7*0.9, 4.8*0.9))
    for i, net in enumerate([city_network, city_symmetric_network]):
        ax = axes[i]
        x_coors, y_coors = get_zones_xys_coor(net)
        ax.scatter(x_coors, y_coors, c='blue')
        ax = plot_city_roads(ax, net, color='orange')
        ax = plot_city_areas(ax, net, color='green')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_xlim(-9000, 9000)
        ax.set_ylim(-9000, 9000)
        # ax.set_title('Asymmetric System' if i == 0 else 'Symmetric System', fontsize=12)
    # plt.suptitle('Comparison of City Networks', fontsize=16)
    plt.tight_layout()
    # plt.savefig('/Users/shabeng/Documents/Education/University/Research/FairnessSharingSys/PaperThesis/'
    #             'Experiments/Graphs/city_networks_comparison_with_areas_not_collapse_no_title.png', dpi=1200)
    plt.show()
