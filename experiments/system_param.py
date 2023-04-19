# Python Packages
import numpy as np
# My repo
from script.CityNetwork.city_creation import create_city, loc_to_rec_num, loc_to_rec_num_sub_center
from script.CityNetwork.routing_city import RoutingCity

# Params:
# Create City Layout
width = 8
distance = 800
city_network = create_city(width=width, distance=distance)
routing = RoutingCity(city_network)
t_ij = routing.routing_times_matrix.round()

# Instance Creation Parameters
T = 40 * 60  # 3600, 60*60
V = 15
warmup_reqs_num = 0

# Demand Probabilities
reqs_arr_p = 0.025
reqs_od_p_mat = np.zeros(t_ij.shape)
center_zones_num = width ** 2
center_zones_inxs = list(range(center_zones_num))
suburb_zones_num = t_ij.shape[0] - center_zones_num
reqs_od_p_mat[:center_zones_num, :center_zones_num] = \
    0.25 * (1 / center_zones_num) * (1 / center_zones_num)
reqs_od_p_mat[:center_zones_num, center_zones_num:] = \
    0.25 * (1 / center_zones_num) * (1 / suburb_zones_num)
reqs_od_p_mat[center_zones_num:, :center_zones_num] = \
    0.25 * (1 / center_zones_num) * (1 / suburb_zones_num)
reqs_od_p_mat[center_zones_num:, center_zones_num:] = \
    0.25 * (1 / suburb_zones_num) * (1 / suburb_zones_num)
reqs_od_p_mat = reqs_od_p_mat / reqs_od_p_mat.sum()

expiration_method = 'fixed'
expiration_dur = 6 * 60
expiration_dur_c = expiration_dur
expiration_dur_s = expiration_dur
vehicles_strategy = False
min_pi = 0

# Payment Params:
t_0 = 6 * 60  # np.unique(t_ij)
n = 1
norm_factor = 1
a = n * ((expiration_dur / t_0) + 1)
A = (a * t_0) / n
pay_func = lambda r, travel: \
    (A + (travel[r.get_od_zones()] >= t_0) * a * (travel[r.get_od_zones()] - t_0)) / norm_factor

# G script - set of request groups
requests_group = {** {i: {'group': loc_to_rec_num_sub_center(i, width=width, sub_width=int(width / 2)),
                          'neigh': loc_to_rec_num_sub_center(i, width=width, sub_width=int(width / 2))}
                      for i in center_zones_inxs},
                  ** {j: {'group': loc_to_rec_num_sub_center(j, width=width, sub_width=int(width / 2)),
                          'neigh': loc_to_rec_num_sub_center(j, width=width, sub_width=int(width / 2))}
                      for j in range(center_zones_num, center_zones_num + suburb_zones_num)},
                  ** {'neigh_num': 9 - 1 + 4, 'zone_num': t_ij.shape[0], 'group_type': 'single_o'}}
