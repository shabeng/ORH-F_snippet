from collections import Counter, defaultdict
import numpy as np
from script.ProblemObjects import requests as rq
from script.ProblemObjects import vehicles as vc


def group_definition_given_reqs_set(system_obj,
                                    condition_func=lambda req_obj, sys_obj: req_obj.get_id() > sys_obj.warmup):
    """
    Given a System object after sampling the request set, the function returns a dictionary mapping the groups name
    (the dict keys) to a list of requests of that group (the dict values).
    :param system_obj: System object after request set sampling!
    :param group_type: str, specify how to the determine the groups
    :param condition_func: Function (Req, Sys --> Bool)
                           get as input (request object + system object) and return a boolean to determine if to add to the
                           measure calculation
    :return:
    """
    # (1) Add the number of requests groups
    groups = defaultdict(list)
    group_type = system_obj.find_geo_attr('group_type')
    for req in system_obj.sys_reqs:
        if condition_func(req, system_obj):
            o, d = req.get_od_zones()
            group_o = system_obj.find_geo_attr('group', zone_id=o)
            group_d = system_obj.find_geo_attr('group', zone_id=d)
            if group_type == 'couple_od':
                group_name = f'{group_o},{group_d}'
            elif group_type == 'single_o':
                group_name = f'{group_o}'
            else:
                group_name = f'{group_d}'
            groups[group_name].append(req)
    return groups


def calc_r_q_G(system_obj, condition_func=lambda req_obj, sys_obj: req_obj.get_id() > sys_obj.warmup):
    """
    The function calculates the size and ratio of all the group of requests G_i, given the sampled set
    :param system_obj: system object
    :param condition_func: Function (Req, Sys --> Bool)
                              get as input a request object and return a boolean to determine if to add to the
                              measure calculation
    :return 2 1d arrays, where the i cell contains the size/ratio of group of requests G_i
    """
    # Assign the warm-up condition if request_condition is None, else assign the given condition
    # request_condition = \
    #     lambda request_object: request_object.k_id > warm_up_reqs if not request_condition else request_condition
    r_Gs = np.zeros(system_obj.find_geo_attr('group_num'))
    for req in system_obj.sys_reqs:
        # if req.k_id > warm_up_reqs:
        if condition_func(req, system_obj):
            i = req.find_request_type(system_obj)
            r_Gs[i] += 1
    return r_Gs, r_Gs / r_Gs.sum()


def calc_weight_abs_big(group_level_x, group_level_q):
    ab = 0
    qX = np.outer(group_level_x, group_level_q)
    for j in range(group_level_x.shape[0]):
        ab += np.abs(qX[j + 1:, j] - qX[j, j + 1:]).sum()
    return ab


def calc_weight_abs_small(group_level_x, group_level_q):
    ab = 0
    G = group_level_x.shape[0]
    for i in range(G):
        for j in range(i + 1, G):
            ab += np.abs(group_level_q[j] * group_level_x[i] - group_level_q[i] * group_level_x[j])
    return ab


def calc_weight_abs(group_level_x, group_level_q):
    group_num = group_level_x.shape[0]
    if group_num > 7:  # checked in jupyter notebook and it seems faster for more than 7 groups
        ab = calc_weight_abs_big(group_level_x, group_level_q)
    else:
        ab = calc_weight_abs_small(group_level_x, group_level_q)
    return ab


def calc_objective_z(group_level_x, group_level_q, ab=0):
    group_num = group_level_x.shape[0]
    if ab == 0:
        ab = calc_weight_abs(group_level_x, group_level_q)
    z = group_level_x.sum() - ab
    return z


def calc_gini(group_level_x, group_level_q, ab=0):
    group_num = group_level_x.shape[0]
    if ab == 0:
        ab = calc_weight_abs(group_level_x, group_level_q)
    gini = ab / group_level_x.sum()
    return gini


def calc_efficiency(group_level_x):
    return group_level_x.sum()


def calc_objectives(group_level_x, group_level_q):
    ab = calc_weight_abs(group_level_x, group_level_q)
    eff = calc_efficiency(group_level_x)
    gini = calc_gini(group_level_x, group_level_q, ab=ab)
    z = calc_objective_z(group_level_x, group_level_q, ab=ab)
    return z, gini, eff


def calc_x_G(system_obj, solution, is_warmup):
    """The function calculates the number of served requests from each group given a dictionary solution format
    (can be converted from list to dictionary with list_to_dict_solution function)
    (*) routes dictionary: dict - key is vehicle id and value is a list of requests id,
                        also key = 0 for not served requests.
    :param system_obj: object of class system
    :param solution: dictionary or list as above.
    :param is_warmup: bool, if True does not count the requests in the warmup duration in x_G.
    :return 1d array, where the i cell contains the number served requests in group of requests G_i
    """
    x_Gs = np.zeros(system_obj.find_geo_attr('group_num'))  # np.zeros_like(system_obj.sys_r_G)
    warmup_num = system_obj.warmup if is_warmup else 0
    for v_id, route in solution.items():
        if v_id == 0:
            continue
        for req_id in route:
            if req_id > warmup_num:
                req_obj = system_obj.sys_reqs[req_id - 1]
                req_group = req_obj.find_request_type(system_obj)
                x_Gs[req_group] += 1
    # x_Gs = np.zeros(len(Counter(G_script.values()).keys()))
    # iterator = sum(dict((i, solution[i]) for i in solution if i != 0).values(), [])
    # for req in iterator:
    #     # (-1) is needed since the dictionary holds the ID of the requests which is different from its index on the
    #     # system object
    #     req_obj = system_obj.sys_reqs[req + system_obj.warmup - 1]
    #     group = G_script[req_obj.get_od_zones()]
    #     x_Gs[group] += 1
    return x_Gs


def calc_partial_measures_of_sol(sol_lst, system_obj,
                                 partial_cond=lambda req_obj_, sys_obj_: req_obj_.get_expiration_time() <= sys_obj_.sys_T):
    # Create groups
    groups_partial = group_definition_given_reqs_set(system_obj, partial_cond)
    groups_num = len(groups_partial)
    groups_mapping = {group: i for i, group in enumerate(groups_partial)}
    x_g_partial = np.zeros(groups_num)
    r_g_partial = np.zeros(groups_num)
    for req_ind, v_id in enumerate(sol_lst):
        req = system_obj.sys_reqs[req_ind]
        if partial_cond(req, system_obj):
            o, d = req.get_od_zones()
            o_group = system_obj.find_geo_attr('group', zone_id=o)
            d_group = system_obj.find_geo_attr('group', zone_id=d)
            group_type = system_obj.find_geo_attr('group_type')
            if group_type == 'couple_od':
                group_str = f'{o_group},{d_group}'
            elif group_type == 'single_o':
                group_str = f'{o_group}'
            else:
                group_str = f'{d_group}'
            req_group = groups_mapping[group_str]
            # add to the group counter r_Gi only if the request is arriving early enough
            r_g_partial[req_group] += 1
            if v_id not in [0, 'r']:
                # add to the group service counter x_Gi only if the request is also served
                x_g_partial[req_group] += 1
    q_g_partial = r_g_partial / r_g_partial.sum()
    z_value = calc_objective_z(x_g_partial, q_g_partial)
    eff_measure = x_g_partial.sum()
    if group_type != 'couple_od':
        _, order = zip(*sorted(groups_mapping.items(), key=lambda x: int(x[0])))
        order = list(order)
        return z_value, eff_measure, r_g_partial[order], x_g_partial[order]
    else:
        return z_value, eff_measure, r_g_partial, x_g_partial


def get_value_df_cell(cell):
    if not cell.empty:
        return int(cell.values[0].split(',')[-1])
    else:
        return -1


def list_to_dict_solution(solution_lst, warmup_num):
    """
    Change between format of solution: from list to dictionary.
    :param solution_lst: list of int: the elem in index i is the vehicle ID that serves request i + 1
    (or 0 if i + 1 is not served)
    :param warmup_num: the warmup number is the number of requests that are not count for the initialization of the
    system.
    :return: routes_dictionary: dict - key is vehicle id and value is a list of requests id,
                        also key = 0 for not served requests.
    """
    routes_dictionary = {}
    for ind, veh in enumerate(solution_lst):
        req_id = ind + 1
        if req_id > warmup_num:
            route_v = routes_dictionary.get(veh, [])
            route_v.append(req_id)
            routes_dictionary[veh] = route_v
    return routes_dictionary


def dict_to_list_solution(solution_dict, warmup_num, reqs_size):
    """
    Change between format of solution: from dictionary to list.
    :param solution_dict: dict, key is vehicle id and value is list of request ids on its route.
                            also key = 0 for not served requests
    :param warmup_num: the warmup number is the number of requests that are not count for the initialization of the
    system.
    :param reqs_size:
    :return: list of int: the elem in index i is the vehicle ID that serves request i + 1
    (or 0 if i + 1 is not served) .
    """
    requests_num = max(sum(len(v) for v in solution_dict.values()), reqs_size)
    sol_arr = np.zeros(requests_num)
    for v_id, route_v in solution_dict.items():
        if v_id == 0:
            continue
        for req_id in route_v:
            if req_id > warmup_num:
                sol_arr[req_id - 1] = int(v_id)
    return list(sol_arr)


def create_simu_sol_to_milp_sol(routes_dictionary, system_obj, pickup_times):
    """
    Takes a simulation solution (any rule) and creates the matching values for the milp formulation.
    :param routes_dictionary: dict - key is vehicle id and value is a list of requests id,
                                also key = 0 for not served requests.
    :param system_obj: System object - with final status updates (requests has their updated attributes).
    :param pickup_times: list - in the (req_id - 1) index there is the pickup time for req_id requests.
    The pickup time for not served requests is their arrival time (satisfies the constraints)
    :return: vars_values_dict: dict, key is the decision value name and value is the value that corresponds the given
    solution.
    """
    import bisect
    vars_values_dict = {}
    not_served_reqs = routes_dictionary.get(0, [])
    # The function only assigns the variables that are "on" (equals 1) or that are needed for the feasibility of the
    # solution (u_vk, z_vk'k) or continues (t_k).
    for vehicle in range(1, system_obj.sys_V + 1):
        route_v = routes_dictionary.get(vehicle, [])
        for route_ind, req in enumerate(route_v):
            req_obj = system_obj.sys_reqs[req - 1]
            if route_ind == 0:
                vars_update = {**{f'y_{vehicle},{req}': 1, f't_{vehicle}_{req}': pickup_times[req - 1],
                                  f'p_{vehicle}_{req}': 1,
                                  f'u_{vehicle}_{req}': 1},
                               **{f'z_{vehicle},{k_tilda}': 1 for k_tilda in range(1, req)}}
                vars_values_dict.update(vars_update)

            else:
                prev_req = route_v[route_ind - 1]
                # u_v_k assignment - if vacant equals 0 ; if busy equals 1
                vars_update = {**{f'x_{vehicle}_{prev_req},{req}': 1, f't_{vehicle}_{req}': pickup_times[req - 1],
                                  f'p_{vehicle}_{req}': 1,
                                  f'u_{vehicle}_{req}': int(req_obj.get_is_busy_vehicle())},
                               **{f'z_{vehicle}_{prev_req},{k_tilda}': 1 for k_tilda in range(prev_req + 1, req + 1)}}
                vars_values_dict.update(vars_update)

            if req == route_v[-1]:
                vars_values_dict.update({f'z_{vehicle}_{req},{k_tilda}': 1
                                         for k_tilda in
                                         range(req + 1, system_obj.sys_reqs_size + system_obj.warmup + 1)})

            vars_update = {**{f't_{vehicle_other}_{req}': req_obj.get_arrival_time()
                              for vehicle_other in range(1, system_obj.sys_V + 1) if vehicle_other != vehicle},
                           **{f'p_{vehicle_other}_{req}': 0
                              for vehicle_other in range(1, system_obj.sys_V + 1) if vehicle_other != vehicle},
                           **{f'u_{vehicle_other}_{req}': 0
                              for vehicle_other in range(1, system_obj.sys_V + 1) if vehicle_other != vehicle}}
            vars_values_dict.update(vars_update)

        for req_rej_id in not_served_reqs:
            req_rej_obj = system_obj.sys_reqs[req_rej_id - 1]
            # find the closest int that is smaller then the req id
            req_prev_i = bisect.bisect_left(route_v, req_rej_id)
            # assign the z variable
            if req_prev_i == 0:
                vars_values_dict.update({f'z_{vehicle},{req_rej_id}': 1})
            else:
                req_prev_id = route_v[req_prev_i - 1]
                req_prev_obj = system_obj.sys_reqs[req_prev_id - 1]
                vars_values_dict.update({f'z_{vehicle}_{req_prev_id},{req_rej_id}': 1})
                # if pickup_times[req_prev_id - 1] + \
                #         system_obj.get_travel_time(*req_prev_obj.get_od_zones()) + \
                #         system_obj.get_travel_time(req_prev_obj.get_od_zones()[1], req_rej_obj.get_od_zones()[0]) > \
                #         req_rej_obj.get_expiration_time():
                if pickup_times[req_prev_id - 1] + system_obj.get_travel_time(*req_prev_obj.get_od_zones()) > \
                        req_rej_obj.get_arrival_time():
                    vars_values_dict.update({f'u_{vehicle}_{req_rej_id}': 1})
            # assign the pickup time to be a_k
            vars_values_dict.update({f't_{vehicle}_{req_rej_id}': req_rej_obj.get_arrival_time()})

    # assign the values for the absolute number variables:
    group_num = system_obj.sys_r_G.shape[0]
    x_group_level = calc_x_G(system_obj, routes_dictionary, is_warmup=False)
    vars_values_dict.update({f'W_G{i},G{j}': np.abs(system_obj.sys_q_G[j] * x_group_level[i] -
                                                    system_obj.sys_q_G[i] * x_group_level[j])
                             for i in range(group_num) for j in range(i + 1, group_num)})
    return vars_values_dict


def calc_solution_measures(routes_dictionary, system_obj):
    from script.SolutionsApproch.Simulation.simulation import run_simulation_route_measures

    # Objective calculation
    x_G = calc_x_G(system_obj, routes_dictionary, is_warmup=False)
    z, gini, f = calc_objectives(x_G, system_obj.sys_q_G)

    # Waiting time of vehicles
    tot_wait = 0
    for v_id, v_route in routes_dictionary.items():
        if v_id == 0:
            continue
        v_wait_time = run_simulation_route_measures(system_obj, v_id, v_route)
        tot_wait += v_wait_time

    # F, gini, Z, waiting time
    return z, gini, f, tot_wait


def calc_requests_waiting_times_measures(system_updated,
                                         partial_cond=lambda req_obj_, sys_obj_: req_obj_.get_expiration_time() <= sys_obj_.sys_T):
    elem_lst = system_updated.sys_reqs
    total_waiting_time = 0
    total_serve_count = 0
    total_enter_count = 0
    for req in elem_lst:
        if partial_cond(req, system_updated):
            total_enter_count += 1
            if req.get_is_served():
                total_waiting_time += req.get_waiting_time()
                total_serve_count += 1
    mean_waiting_time = total_waiting_time / total_serve_count
    percent_rejection = (total_enter_count - total_serve_count) / total_enter_count
    return total_waiting_time, mean_waiting_time, percent_rejection


def copy_objects_lst(objs_lst, system_obj):
    objs_lst_copy = []
    for obj in objs_lst:
        if obj.check_instance(obj_name='Request'):
            o, d = obj.get_od_zones()
            obj_copy = rq.Request(k=obj.get_id(), origin_zone=o, destination_zone=d, arrival_time=obj.get_arrival_time(),
                                  expiration_time=obj.get_expiration_time())
        else:
            t, z = obj.get_state()
            obj_copy = vc.Vehicle(vehicle_id=obj.get_id(), vehicle_time=t, vehicle_zone=z, sys=system_obj)
        objs_lst_copy.append(obj_copy)
    return objs_lst_copy


def assert_system_settings(system_obj, settings_dict, assert_name, with_T=True):
    if with_T:
        assert settings_dict['sys_T'] == system_obj.sys_T, f'{assert_name} sys_T'
    assert settings_dict['sys_V'] == system_obj.sys_V, f'{assert_name} sys_V'
    assert settings_dict['sys_zones_size'] == system_obj.sys_zones_size, f'{assert_name} sys_zones_size'
    assert settings_dict['sys_center_inx'] == system_obj.sys_center_inx, f'{assert_name} sys_center_inx'
    assert np.all(settings_dict['sys_t_i_j'] == system_obj.sys_t_i_j), f'{assert_name} sys_t_i_j'
    assert np.all(settings_dict['sys_reqs_arr_prob'] == system_obj.sys_reqs_arr_prob), f'{assert_name} sys_reqs_arr_prob'
    assert np.all(settings_dict['sys_exp_method'] == system_obj.sys_exp_method), f'{assert_name} sys_exp_method'
    assert np.all(settings_dict['warmup'] == system_obj.warmup), f'{assert_name} warm-up'


if __name__ == '__main__':
    # Params:
    seed = 44
    T = 30
    V = 2
    t_ij = np.array([[1, 4, 5, 13, 10, 8],
                     [4, 1, 3, 13, 14, 12],
                     [5, 3, 1, 10, 12, 11],
                     [13, 13, 10, 1, 20, 18],
                     [10, 14, 12, 20, 1, 3],
                     [8, 12, 11, 18, 3, 1]])
    t_ij = np.floor((t_ij + 1) / 2)
    reqs_arr_p = 0.2
    reqs_od_p_mat = np.array([[0.02222222, 0.02222222, 0.02222222, 0.03888889, 0.03888889, 0.03888889],
                              [0.02222222, 0.02222222, 0.02222222, 0.03888889, 0.03888889, 0.03888889],
                              [0.02222222, 0.02222222, 0.02222222, 0.03888889, 0.03888889, 0.03888889],
                              [0.03888889, 0.03888889, 0.03888889, 0.01111111, 0.01111111, 0.01111111],
                              [0.03888889, 0.03888889, 0.03888889, 0.01111111, 0.01111111, 0.01111111],
                              [0.03888889, 0.03888889, 0.03888889, 0.01111111, 0.01111111, 0.01111111]])
    # Payment Params:
    A = np.max(t_ij)
    a = 5
    t_min = A / a
    pay_func = lambda r, travel: A if travel[r.k_orig, r.k_dest] <= t_min \
                                    else A + a*(travel[r.k_orig, r.k_dest] - t_min)
    # System instance
    # sys1 = s.System(seed, T, V, t_ij, reqs_arr_p, reqs_od_p_mat, pay_func)

    # Request Mapping
    # r_G, q_G = calc_r_q_G(sys1.sys_reqs, {** {(i, j): 0 for i in range(3) for j in range(3)},
    #                                       ** {(i, j): 1 for i in range(3) for j in range(3, 6)},
    #                                       ** {(i, j): 2 for i in range(3, 6) for j in range(3)},
    #                                       ** {(i, j): 3 for i in range(3, 6) for j in range(3, 6)}}
    #                       )
    # print(r_G, q_G)
    # print(sys1.sys_reqs)
