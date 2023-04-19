import numpy as np
import copy

from script.utilis.utilis import calc_objective_z, group_definition_given_reqs_set, calc_r_q_G, \
    calc_partial_measures_of_sol
from script.utilis.utilis_simulation import write_stats_state_information
from script.SolutionsApproch.Simulation.dispatch import time_earliest_arriving, random_available
from script.ProblemObjects.vehicles import Vehicle


def run_simulation_rule(system_obj, rule_func, warm_up_func=None, warm_up_time=600,
                        condition=lambda req_, sys_: req_.get_expiration_time() <= sys_.sys_T, **kwargs):
    """
    the function will run the simulation according to the rule obtained as input.
    :param condition: boolean function to calculate the partial measures with
    :param system_obj: the initialized system with all the attributes and methods.
    :param rule_func: function, a rule to use in order to assign a request to a vehicle.
                    Input is a list of vehicles objects that are available and the request object.
                    Output is the chosen vehicle object.
    :param warm_up_func: dispatching function to assign in warm-up phase which is 600 seconds
    :param kwargs: (1) area_vehs_select_func -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: z value of the full solution and the solution as a list of v_ids.
    """
    group_num = system_obj.find_geo_attr('group_num')  # system_obj.sys_r_G.shape[0]
    warm_up_func = warm_up_func if warm_up_func else rule_func
    sol = []
    t_k = []
    x_g = np.zeros(group_num)          # measure of the full request set
    for req in system_obj.sys_reqs:
        v_k_set = system_obj.find_vehicles_k(req)
        if len(v_k_set) == 0:
            if req.k_id > system_obj.warmup:
                sol.append(0)
                t_k.append(req.get_arrival_time())
                if kwargs.get('keep_state_information', False):
                    write_stats_state_information(system_obj=system_obj, req_obj=req,
                                                  decision=(-1), avail_vehs=v_k_set)
            continue
        elif len(v_k_set) == 1:
            chosen_veh = v_k_set[0]
        else:
            if req.get_arrival_time() < warm_up_time:
                chosen_veh = warm_up_func(v_k_set, req, system_obj, **kwargs)
            else:
                chosen_veh = rule_func(v_k_set, req, system_obj, **kwargs)
        if kwargs.get('keep_state_information', False):
            decision = system_obj.find_geo_attr('neigh', zone_id=chosen_veh.get_zone()) if isinstance(chosen_veh, Vehicle) else ('r')
            write_stats_state_information(system_obj=system_obj, req_obj=req, decision=decision, avail_vehs=v_k_set)
        if not isinstance(chosen_veh, Vehicle):
            if req.k_id > system_obj.warmup:
                sol.append('r')
                t_k.append(req.get_arrival_time())
            continue
        # Update State
        veh_info, req_info = system_obj.update_match_measures(chosen_veh, req)
        # Request:
        req.update_match(req.k_id > system_obj.warmup, chosen_veh.v_id, *req_info)
        # Vehicle:
        chosen_veh.update_match(req.k_id > system_obj.warmup, *veh_info, req)
        # Solution:
        if req.k_id > system_obj.warmup:
            sol.append(chosen_veh.v_id)
            g_i = req.find_request_type(system_obj)
            x_g[g_i] += 1
            t_k.append(req.get_arrival_time() + req_info[0])

    # Measures Calculations
    z_val = calc_objective_z(x_g, system_obj.sys_q_G)
    z_val_partial, f_val_partial, r_g_partial, x_g_partial = \
        calc_partial_measures_of_sol(sol, system_obj,
                                     partial_cond=condition)
    return (z_val, z_val_partial), (x_g.sum(), f_val_partial), (system_obj.sys_r_G, r_g_partial), sol, t_k


def run_simulation_route_validation(system_obj, v_id, route, tk_df, mask_cond):
    """
    Takes a route as the solution of one vehicle from the off problem and checks:
    (1) that the t_k values determined by the MILP model works with the simulation travel time (trip time, cruise time
        and arrival time).
    (2) that the t_k value not violating the time window of any request.
    (3) whether the order of serving the requests is as their arrival order (no knowledge before arrival)
    (?) whether the rejected requests are rejected because of unavailability or by choice... ???
    :param system_obj: system object of the problem checked.
    :param v_id: int, the vehicle id that serves the route (starts with 1 so need to take minus 1).
    :param route: lst, a list of request indices (starts with 1 so need to take minus 1).
    :param tk_lst: lst, a list of floats that represent the pickup time of the request as determined by the MILP model.
    :param not_served_ids: lst, a list of integers which represent the requests ids of the requests that are rejected by
            the solution.
    :return:
    """
    veh = copy.deepcopy(system_obj.sys_vehs[v_id - 1])
    mask_vehicle = f'{mask_cond}{v_id}' if mask_cond == '_' else mask_cond
    res = {'t_k': {}, 'time_window': {}}
    epsilon = 10**(-10)
    for req_id in route:
        t_k = tk_df.loc[tk_df.varNames == f't{mask_vehicle}_{req_id}'].varValues.values[0]
        req = copy.deepcopy(system_obj.sys_reqs[req_id - 1])
        a_k = req.get_arrival_time()
        e_k = req.get_expiration_time()
        available_time = max(veh.get_time(), a_k)
        cruise_time = system_obj.get_travel_time(veh.get_zone(), req.get_od_zones()[0])
        trip_time = system_obj.get_travel_time(*req.get_od_zones())
        earliest_pick_up = available_time + cruise_time

        if abs(t_k - earliest_pick_up) > epsilon:
            examples = res.get('t_k', {})
            examples[req_id] = (earliest_pick_up, t_k)
            res['t_k'] = examples
        if (t_k - epsilon < a_k) or (t_k > e_k + epsilon):
            examples = res.get('time_window', {})
            examples[req_id] = (a_k, e_k, t_k)
            res['time_window'] = examples

        veh_info, req_info = system_obj.update_match_measures(veh, req)
        veh.update_match(True, *veh_info, req)
    res['order'] = sorted(route) == route
    return res


def run_simulation_route_measures(system_obj, v_id, route):
    """
    Take a route of a vehicle and a system object to calculate the wait time of the vehicle in the system. Can be easily
    expanded to give the cruising time and trip time of the vehicle.
    :param system_obj: System class object
    :param v_id: int, id of the vehicle (starting to count from 1)
    :param route: list of int, each element is a request id (starting to count from 1)
    :return:
    """
    veh = copy.deepcopy(system_obj.sys_vehs[v_id - 1])
    for req_id in route:
        req = copy.deepcopy(system_obj.sys_reqs[req_id - 1])
        veh_info, req_info = system_obj.update_match_measures(veh, req)
        veh.update_match(True, *veh_info, req)
    cruise_time, wait_time, gain = veh.write_row_one_veh()
    return wait_time


if __name__ == '__main__':
    from script.ProblemObjects import system as s
    import PreliminaryExperiments.pocML.solutionCollection.exp8.system_param_81 as sp
    from script.SolutionsApproch.Simulation.dispatch import time_earliest_arriving,  random_available, \
        time_nearest_available, balance_crowded_zone, balance_balanced_zone
    from script.SolutionsApproch.ML.voting_schemes import uniform_vote, inverse_distance_vote, percentage_vote, \
        inverse_rank_vote, minmax_weighted_distance_vote, dual_rank_minmax_vote
    from script.SolutionsApproch.ML.data_driven_rules import data_driven_rule_wo_rej
    from script.SolutionsApproch.ML.states_distance_functions import state_dist_avail_areas_euc
    from script.SolutionsApproch.ML.preprocess import create_data_sets

    import numpy as np

    rules = [
        time_earliest_arriving, random_available,
        time_nearest_available, balance_crowded_zone, balance_balanced_zone
    ]

    seeds = np.random.randint(999999, size=5)
    for seed_ind, seed_sys in enumerate(seeds):
        for ind, rule in enumerate(rules):
            system = s.System(seed_sys, sp.T, sp.V, sp.t_ij, sp.reqs_arr_p, sp.reqs_od_p_mat, sp.pay_func,
                              sp.requests_group.copy(), sp.center_zones_inxs, sp.warmup_reqs_num,
                              expiration_method=sp.expiration_method,
                              fixed_exp_c=sp.expiration_dur_c, fixed_exp_s=sp.expiration_dur_s)
            z_vals, f_vals, rgs, sol_lst, pickup_time = run_simulation_rule(system, rule,
                                                                            warm_up_func=time_earliest_arriving,
                                                                            warm_up_time=600,
                                                                            area_vehs_select_func=time_earliest_arriving)
            rule_name = str(rule).split(' ')[1]
            print(f'Rule {rule_name}: Z = {z_vals} | F = {f_vals}')
        print(f'Finished seed {seed_ind}!\n')
