# Functions to implement different Dispatching Rules to be used in the function run_simulation_rule.
from script.utilis.utilis_simulation import create_vehs_distribution, calc_arrival_time, calc_demand_neigh_est, \
    create_neigh_vehs_mapping, create_vehs_distribution_norm, extend_avail_vehs_with_vacant

import numpy as np
from collections import defaultdict


def random_available(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (5) Random available vehicle - returns a random vehicle object from the available vehicles.
    The rule uses the system object random state.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    veh_chosen = sys_obj.sys_seed.choice(vehs_lst)
    return veh_chosen


def select_veh(vehs_lst, req_obj, sys_obj, tie_func=random_available):
    """
    Randomly select a vehicle if the list is longer than 1, otherwise select one vehicle on random (so to remove the
    bias towards lower indexes vehicles or zones).
    :param vehs_lst: list of vehs objects
    :param req_obj: Request object
    :param sys_obj: System object
    :param tie_func: function to determine how to select between same status vehicles (that have the same score)
    :return: one veh object
    """
    veh_chosen = tie_func(vehs_lst, req_obj, sys_obj)
    return veh_chosen


def select_zone_max(avail_vehs, system_obj, measures_array):
    """
    Given a measure array that has a measure for each area, the function returns a list of the available
    vehicles that are in the area that matches the maximum measure value.
    Measures for example are the supply-demand ratio or the number of vehicles in each zone.
    :param avail_vehs: list of vehicles objects
    :param system_obj: System object
    :param measures_array: 1d array with the measure value for each neighborhood
    :return: list of vehicles objects in the same neighborhood that has the highest measure
    """
    zones_available = create_neigh_vehs_mapping(system_obj, vehs_lst=avail_vehs)
    inds_zones_available = sorted(zones_available.keys())
    if len(inds_zones_available) > 1:
        # Neighborhood with highest measure
        measure_max = measures_array[inds_zones_available].max()
        inds_zones_max = np.where(measures_array[inds_zones_available] == measure_max)[0]
        # Choose random zone if there are more than one zone with the same maximum measure
        ind_zone_max = system_obj.sys_seed.choice(inds_zones_max)
        ind_zone_max = inds_zones_available[ind_zone_max]
        vehs_in_zone = zones_available[ind_zone_max]
    else:
        # There is only one neighborhood where are all the available vehicles are
        vehs_in_zone = zones_available[inds_zones_available[0]]
    return vehs_in_zone


def time_vacant(vehs_lst, req_obj, sys_obj, desc_ord):
    func = min if not desc_ord else max
    time_val = func([veh.get_time() for veh in vehs_lst])
    vehs_with_val = [veh for veh in vehs_lst if veh.get_time() == time_val]
    return select_veh(vehs_with_val, req_obj, sys_obj, tie_func=random_available)


def time_arriving(vehs_lst, req_obj, sys_obj, desc_ord):
    func = min if not desc_ord else max
    time_val = func([calc_arrival_time(veh, req_obj, sys_obj) for veh in vehs_lst])
    vehs_with_val = [veh for veh in vehs_lst if calc_arrival_time(veh, req_obj, sys_obj) == time_val]
    return select_veh(vehs_with_val, req_obj, sys_obj, tie_func=random_available)


def time_cruising(vehs_lst, req_obj, sys_obj, desc_ord):
    func = min if not desc_ord else max
    time_val = func([sys_obj.get_travel_time(veh.get_zone(), req_obj.get_od_zones()[0]) for veh in vehs_lst])
    vehs_with_val = \
        [veh for veh in vehs_lst if sys_obj.get_travel_time(veh.get_zone(), req_obj.get_od_zones()[0]) == time_val]
    return select_veh(vehs_with_val, req_obj, sys_obj, tie_func=random_available)


def time_earliest_vacant(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (1.1) Earliest vacant vehicle - return the object of the available vehicle that is vacant first among Vk,
    the available vehicles.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    is_desc = False
    veh_earliest_vac = time_vacant(vehs_lst, req_obj, sys_obj, is_desc)
    return veh_earliest_vac


def time_latest_vacant(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (1.2) Latest vacant vehicle - return the object of the available vehicle that is vacant last among Vk,
    the available vehicles.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    is_desc = True
    veh_latest_vac = time_vacant(vehs_lst, req_obj, sys_obj, is_desc)
    return veh_latest_vac


def time_earliest_arriving(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (2.1) Earliest arriving vehicle - returns the object of the available vehicle that manage to arrive first to
    the origin of the request among Vk, the available vehicles. The rule accounts to the issue that the service time
    can only start after the vehicle is vacant or the request has arrived, the latest time between the two.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    is_desc = False
    veh_earliest_arr = time_arriving(vehs_lst, req_obj, sys_obj, is_desc)
    return veh_earliest_arr


def time_latest_arriving(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (2.2) Latest arriving vehicle - returns the object of the available vehicle that manage to arrive last to
    the origin of the request among Vk, the available vehicles. The rule accounts to the issue that the service time
    can only start after the vehicle is vacant or the request has arrived, the latest time between the two.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    is_desc = True
    veh_earliest_arr = time_arriving(vehs_lst, req_obj, sys_obj, is_desc)
    return veh_earliest_arr


def time_nearest_available(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (3.1) Nearest available vehicle - returns the object of the available vehicle with the minimum cruising time from
    the vacant zone z_v to the request origin zone ok.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    is_desc = False
    veh_nearest = time_cruising(vehs_lst, req_obj, sys_obj, is_desc)
    return veh_nearest


def time_farthest_available(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (3.2) Farthest available vehicle - returns the object of the available vehicle with the maximum cruising time from
    the vacant zone z_v to the request origin zone ok.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    is_desc = True
    veh_farthest = time_cruising(vehs_lst, req_obj, sys_obj, is_desc)
    return veh_farthest


def time_nearest_vacant(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (4) Nearest vacant vehicle - returns the object of the vehicle with the minimum cruising time from
    the vacant zone z_v to the request origin zone ok. This rule only considers the vehicles that are vacant at the time
    the request has arrived, that is t_v <= t_k.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :return: vehicle object
    """
    vehs_lst_vacant = [veh for veh in vehs_lst if veh.get_time() <= req_obj.get_arrival_time()]
    if len(vehs_lst_vacant) > 0:
        return time_nearest_available(vehs_lst_vacant, req_obj, sys_obj)
    else:  # There is no vacant vehicle among the available vehicles
        return []


def balance_crowded(vehs_k_avail, vehs_supply, req_obj, sys_obj, **kwargs):
    """
    Basic function to perform the different version of Most Crowded Area rules
    :param vehs_k_avail: list of available vehicles concerning req_obj
    :param vehs_supply: list of vehicles to calculate with the area-vehicle vector of supply
    :param req_obj: Request object
    :param sys_obj: System object, updated one
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return:
    """
    if 'area_vehs_select_func' in kwargs:
        area_vehs_select_func = kwargs['area_vehs_select_func']
    else:
        area_vehs_select_func = random_available
    # Create the area-vehicle vector of supply
    count_vehs_area = create_vehs_distribution(sys_obj, vehs_lst=vehs_supply)
    # Available vehicles in the highest number of vehicles
    vehs_in_area = select_zone_max(vehs_k_avail, sys_obj, count_vehs_area)
    return select_veh(vehs_in_area, req_obj, sys_obj, tie_func=area_vehs_select_func)


def balance_crowded_normalized(vehs_k_avail, vehs_supply, req_obj, sys_obj, **kwargs):
    """
    Basic function to perform the different version of Most Crowded Normalized Area rules.
    The demand is the estimated number of requests that are about to origin from that zone from tk to ek + t_ok_dk time
    period.
    :param vehs_k_avail: list of available vehicles concerning req_onj
    :param vehs_supply: list of vehicles to calculate with the area-vehicle vector of supply
    :param req_obj: Request object
    :param sys_obj: System object, updated one
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return:
    """
    if 'area_vehs_select_func' in kwargs:
        area_vehs_select_func = kwargs['area_vehs_select_func']
    else:
        area_vehs_select_func = random_available
    # Create the area-vehicle vector of supply
    count_vehs_area = create_vehs_distribution(sys_obj, vehs_lst=vehs_supply)
    # Time duration to estimate the demand
    dur_time = (req_obj.get_expiration_time() + sys_obj.get_travel_time(*req_obj.get_od_zones())) - \
               req_obj.get_arrival_time()
    # Num of requests estimated to arrive from each area
    reqs_demand = calc_demand_neigh_est(sys_obj, dur_time)
    # Supply-Demand ratio
    ratio = count_vehs_area / reqs_demand
    # Supply-Demand ratio
    sup_demnd_ratio = create_vehs_distribution_norm(sys_obj, req_obj, vehs_lst=vehs_supply)
    assert np.all(ratio == sup_demnd_ratio), \
        'Different arrays between the new function output and the previous function!'
    vehs_in_zone = select_zone_max(vehs_k_avail, sys_obj, ratio)
    return select_veh(vehs_in_zone, req_obj, sys_obj, tie_func=area_vehs_select_func)


def balance_MC1(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (6) Most crowded area vehicle sends an available vehicle that is currently in the most crowded zone,
        when the supply of vehicles is the whole fleet.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    vehs_supply_lst = sys_obj.sys_vehs
    chosen_veh = balance_crowded(vehs_lst, vehs_supply_lst, req_obj, sys_obj, **kwargs)
    return chosen_veh


def balance_MC_Norm1(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (7) Most crowded normalized area vehicle sends an available vehicle that is currently in the area with the highest
        supply-demand ratio, when the supply of vehicles is the whole fleet.
    :param vehs_lst: lst of vehicles object, each element is an available vehicle
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    vehs_supply_lst = sys_obj.sys_vehs
    chosen_veh = balance_crowded_normalized(vehs_lst, vehs_supply_lst, req_obj, sys_obj, **kwargs)
    return chosen_veh


def balance_MC2(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (8) Most crowded area vehicle sends an available vehicle that is currently in the most crowded zone,
        when the supply of vehicles is only the available vehicles.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    vehs_supply_lst = vehs_lst
    chosen_veh = balance_crowded(vehs_lst, vehs_supply_lst, req_obj, sys_obj, **kwargs)
    return chosen_veh


def balance_MC_Norm2(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (9) Most crowded normalized area vehicle sends an available vehicle that is currently in the area with the highest
        supply-demand ratio, when the supply of vehicles is only the available vehicles.
    :param vehs_lst: lst of vehicles object, each element is an available vehicle
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    vehs_supply_lst = vehs_lst
    chosen_veh = balance_crowded_normalized(vehs_lst, vehs_supply_lst, req_obj, sys_obj, **kwargs)
    return chosen_veh


def balance_MC3(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (10) Most crowded area vehicle sends an available vehicle that is currently in the most crowded zone,
        when the supply of vehicles is the available vehicles + vacant vehicles from the entire fleet.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    # Num vehicles in each area - distribution of the available vehicles + current vacant vehicles
    vehs_lst_extend = vehs_lst[:]
    vehs_set_indx = [v.get_id() for v in vehs_lst_extend]
    for v in sys_obj.sys_vehs:
        # v is vacant vehicle which is not already in V script k, the available vehicles
        if v.get_time() <= req_obj.get_arrival_time() and v.get_id() not in vehs_set_indx:
            vehs_lst_extend.append(v)
    vehs_avail_vacant = extend_avail_vehs_with_vacant(vehs_lst, sys_obj, req_obj)
    assert sorted(v.get_id() for v in vehs_avail_vacant) == sorted(v.get_id() for v in vehs_lst_extend), \
        'Different list of available + vacant vehicles'
    vehs_supply_lst = vehs_lst_extend
    chosen_veh = balance_crowded(vehs_lst, vehs_supply_lst, req_obj, sys_obj, **kwargs)
    return chosen_veh


def balance_MC_Norm3(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (11) Most crowded normalized area vehicle sends an available vehicle that is currently in the area with the highest
        supply-demand ratio,
        when the supply of vehicles is the available vehicles + vacant vehicles from the entire fleet.
    :param vehs_lst: lst of vehicles object, each element is an available vehicle
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    vehs_avail_vacant = extend_avail_vehs_with_vacant(vehs_lst, sys_obj, req_obj)
    vehs_supply_lst = vehs_avail_vacant
    chosen_veh = balance_crowded_normalized(vehs_lst, vehs_supply_lst, req_obj, sys_obj, **kwargs)
    return chosen_veh


def balance_with_rejection(vehs_lst, req_obj, sys_obj, balanced_rule, **kwargs):
    """

    :param vehs_lst:
    :param req_obj:
    :param sys_obj:
    :param balanced_rule: function, one of the variation of the balanced rule to chose the vehicle
    :param kwargs:
    :return:  (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
              (2) 'rej_th' -
                        the threshold number in which the request is rejected

    """
    chosen_veh = balanced_rule(vehs_lst, req_obj, sys_obj, **kwargs)
    chosen_area = sys_obj.find_geo_attr('neigh', zone_id=chosen_veh.get_zone())
    vehs_areas_cnt = create_vehs_distribution(sys_obj, vehs_lst=sys_obj.sys_vehs)
    if vehs_areas_cnt[chosen_area] > 1:
        return chosen_veh
    else:
        dest_zone = req_obj.get_od_zones()[1]
        dest_area = sys_obj.find_geo_attr('neigh', zone_id=dest_zone)
        vehs_areas_ratio = create_vehs_distribution_norm(sys_obj, req_obj, vehs_lst=sys_obj.sys_vehs)
        # print(f'{vehs_areas_ratio[dest_area]}')
        if vehs_areas_ratio[dest_area] > kwargs['rej_th']:
            return []
        else:
            return chosen_veh


def balance_MC1_with_rejection(vehs_lst, req_obj, sys_obj, **kwargs):
    """

    :param vehs_lst:
    :param req_obj:
    :param sys_obj:
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
                   (2) 'rej_th' -
                        the threshold number in which the request is rejected
    :return:
    """
    func_rule = balance_MC1
    chosen_deci = balance_with_rejection(vehs_lst, req_obj, sys_obj, func_rule, **kwargs)
    return chosen_deci


def balance_MC_Norm1_with_rejection(vehs_lst, req_obj, sys_obj, **kwargs):
    """

    :param vehs_lst:
    :param req_obj:
    :param sys_obj:
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
                   (2) 'rej_th' -
                        the threshold number in which the request is rejected
    :return:
    """
    func_rule = balance_MC_Norm1
    chosen_deci = balance_with_rejection(vehs_lst, req_obj, sys_obj, func_rule, **kwargs)
    return chosen_deci


def balance_crowded_zone(vehs_lst, req_obj, sys_obj, **kwargs):
    return balance_MC1(vehs_lst, req_obj, sys_obj, **kwargs)


def balance_balanced_zone(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (7) Most balanced zone vehicle - returns an available vehicle that is currently in the zone that has the highest
    supply to demand ratio. The supply is the number of vehicles of vehicles in the zone, and the demand is the
    estimated number of requests that are about to origin from that zone from tk to ek + t_ok_dk time period. The ratio
    is only calculated for zones where there are available
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    if 'area_vehs_select_func' in kwargs:
        area_vehs_select_func = kwargs['area_vehs_select_func']
    else:
        area_vehs_select_func = random_available
    # Num vehicles in each neighborhood - distribution of all the fleet (vehs_lst contains only the available ones)
    vehs_supply = create_vehs_distribution(sys_obj)
    # Time duration to estimate the demand
    dur_time = (req_obj.get_expiration_time() + sys_obj.get_travel_time(*req_obj.get_od_zones())) - \
               req_obj.get_arrival_time()
    # Num of requests estimated to arrive from each neighborhood
    reqs_demand = calc_demand_neigh_est(sys_obj, dur_time)
    # Supply-Demand ratio
    ratio = vehs_supply / reqs_demand
    vehs_in_zone = select_zone_max(vehs_lst, sys_obj, ratio)
    return select_veh(vehs_in_zone, req_obj, sys_obj, tie_func=area_vehs_select_func)


def balance_crowded_zone_2(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (6) Most crowded zone vehicle 2 - returns an available vehicle that is currently in the most crowded zone
    when the vehicle set is only the available vehicles.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    if 'area_vehs_select_func' in kwargs:
        area_vehs_select_func = kwargs['area_vehs_select_func']
    else:
        area_vehs_select_func = random_available
    # Num vehicles in each neighborhood - distribution of the available vehicles
    vehs_distribution = create_vehs_distribution(sys_obj, vehs_lst=vehs_lst)
    # Available vehicles in the highest number of vehicles
    vehs_in_zone = select_zone_max(vehs_lst, sys_obj, vehs_distribution)
    return select_veh(vehs_in_zone, req_obj, sys_obj, tie_func=area_vehs_select_func)


def balance_balanced_zone_2(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (7) Most balanced zone vehicle 2 - returns an available vehicle that is currently in the zone that has the highest
    supply to demand ratio.
    The supply is the number of available vehicles in the area
    The demand is the estimated number of requests that are about to origin from that zone from tk to ek + t_ok_dk time
    period.
    The ratio is only calculated for zones where there are available (???)
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    if 'area_vehs_select_func' in kwargs:
        area_vehs_select_func = kwargs['area_vehs_select_func']
    else:
        area_vehs_select_func = random_available
    # Num vehicles in each neighborhood - distribution of the available vehicles
    vehs_supply = create_vehs_distribution(sys_obj, vehs_lst=vehs_lst)
    # Time duration to estimate the demand
    dur_time = (req_obj.get_expiration_time() + sys_obj.get_travel_time(*req_obj.get_od_zones())) - \
               req_obj.get_arrival_time()
    # Num of requests estimated to arrive from each neighborhood
    reqs_demand = calc_demand_neigh_est(sys_obj, dur_time)
    # Supply-Demand ratio
    ratio = vehs_supply / reqs_demand
    vehs_in_zone = select_zone_max(vehs_lst, sys_obj, ratio)
    return select_veh(vehs_in_zone, req_obj, sys_obj, tie_func=area_vehs_select_func)


def balance_crowded_zone_3(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (6) Most crowded zone vehicle 2 - returns an available vehicle that is currently in the most crowded zone
    when the vehicle set is the available vehicles + vacant vehicles from the entire fleet.
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    if 'area_vehs_select_func' in kwargs:
        area_vehs_select_func = kwargs['area_vehs_select_func']
    else:
        area_vehs_select_func = random_available
    # Num vehicles in each neighborhood - distribution of the available vehicles + current vacant vehicles
    vehs_lst_extend = vehs_lst[:]
    vehs_set_indx = [v.get_id() for v in vehs_lst_extend]
    for v in sys_obj.sys_vehs:
        # v is vacant vehicle which is not already in V script k, the available vehicles
        if v.get_time() <= req_obj.get_arrival_time() and v.get_id() not in vehs_set_indx:
            vehs_lst_extend.append(v)
    vehs_distribution = create_vehs_distribution(sys_obj, vehs_lst=vehs_lst_extend)
    # Available vehicles in the highest number of vehicles
    vehs_in_zone = select_zone_max(vehs_lst, sys_obj, vehs_distribution)
    return select_veh(vehs_in_zone, req_obj, sys_obj, tie_func=area_vehs_select_func)


def balance_balanced_zone_3(vehs_lst, req_obj, sys_obj, **kwargs):
    """
    (7) Most balanced zone vehicle 2 - returns an available vehicle that is currently in the zone that has the highest
    supply to demand ratio.
    The supply is the number of available vehicles in the area + vacant vehicles from the
    entire fleet.
    The demand is the estimated number of requests that are about to origin from that zone from tk to ek + t_ok_dk time
    period.
    The ratio is only calculated for zones where there are available (????)
    :param vehs_lst: lst of vehicles object
    :param req_obj: request object
    :param sys_obj: system object
    :param kwargs: (1) 'area_vehs_select_func' -
                        dispatching func to select which vehicle to assign after deciding from which area to send an
                        available vehicle.
    :return: vehicle object
    """
    if 'area_vehs_select_func' in kwargs:
        area_vehs_select_func = kwargs['area_vehs_select_func']
    else:
        area_vehs_select_func = random_available
    # Num vehicles in each neighborhood - distribution of the available vehicles + current vacant vehicles
    vehs_lst_extend = vehs_lst[:]
    vehs_set_indx = [v.get_id() for v in vehs_lst_extend]
    for v in sys_obj.sys_vehs:
        # v is vacant vehicle which is not already in V script k, the available vehicles
        if v.get_time() <= req_obj.get_arrival_time() and v.get_id() not in vehs_set_indx:
            vehs_lst_extend.append(v)
    vehs_supply = create_vehs_distribution(sys_obj, vehs_lst=vehs_lst_extend)
    # Time duration to estimate the demand
    dur_time = (req_obj.get_expiration_time() + sys_obj.get_travel_time(*req_obj.get_od_zones())) - \
               req_obj.get_arrival_time()
    # Num of requests estimated to arrive from each neighborhood
    reqs_demand = calc_demand_neigh_est(sys_obj, dur_time)
    # Supply-Demand ratio
    ratio = vehs_supply / reqs_demand
    vehs_in_zone = select_zone_max(vehs_lst, sys_obj, ratio)
    return select_veh(vehs_in_zone, req_obj, sys_obj, tie_func=area_vehs_select_func)


