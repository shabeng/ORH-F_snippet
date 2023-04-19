# Some functions repeating in the operation of the online simulation
import numpy as np
import pandas as pd
from collections import defaultdict


def create_vehs_distribution(system_obj, vehs_lst=None):
    """
    Using the G_script attribute of the system object to create the distribution of the vehicles in the geographical
    parts of the city. For doing so, the functions uses the following attributes of the system object -
    (1) G script dictionary, both the neigh_num and the mapping from zone_id to the neigh num.
    (2) sys_vehs list of vehicles objects to get their current zone.
    (3) find_geo_attr method to find the neigh num of each z_v.
    :param system_obj: System object that its objects (vehicles especially) are updated.
    :param vehs_lst
    :return: 1-D np array in the dimension of neigh_num
    """
    # Num vehicles in each zone
    geo_dist_num = system_obj.find_geo_attr('neigh_num')
    vehs_dist = np.zeros(geo_dist_num)
    # List to run on
    lst = system_obj.sys_vehs if vehs_lst is None else vehs_lst
    # Distribution of all the fleet (vehs_lst contains only the available ones)
    for veh in lst:
        z_v = veh.get_zone()
        neigh_num = system_obj.find_geo_attr('neigh', zone_id=z_v)
        vehs_dist[neigh_num] += 1
    return vehs_dist


def create_vehs_distribution_norm(system_obj, req_obj, vehs_lst=None):
    """
    Using the G_script attribute of the system object to create the supply-demand ratio in the geographical
    parts of the city. For doing so, the functions uses the following attributes of the system object -
    (1) G script dictionary, both the neigh_num and the mapping from zone_id to the neigh num.
    (2) sys_vehs list of vehicles objects to get their current zone.
    (3) find_geo_attr method to find the neigh num of each z_v.
    :param system_obj: System object that its objects (vehicles especially) are updated.
    :param req_obj: a Request object that the simulation considering how to serve
    :param vehs_lst: to use as the vehicles supply basis
    :return: 1-D np array in the dimension of neigh_num
    """
    # Vehicles distribution according to the supply wanted
    vehs_dist = create_vehs_distribution(system_obj, vehs_lst)
    # Time duration to estimate the demand
    dur_time = (req_obj.get_expiration_time() + system_obj.get_travel_time(*req_obj.get_od_zones())) - \
               req_obj.get_arrival_time()
    # Num of requests estimated to arrive from each area
    reqs_demand = calc_demand_neigh_est(system_obj, dur_time)
    # Supply-Demand ratio
    sup_demand_ratio = vehs_dist / reqs_demand
    return sup_demand_ratio


def create_neigh_vehs_mapping(system_obj, vehs_lst=None):
    """
    Given a vehicle list the function return a mapping between a neighborhood (int) to a list of vehicles objects that
    their vacancy zone is in that neighborhood. If vehs_lst is None, then the function uses the entire fleet as in the
    system object (so it must be one that is updated in a simulation), otherwise the given list is used.
    :param system_obj: System object with updated vehicles objects.
    :param vehs_lst: None or a list of vehicles objects, if one dont want to use the entire fleet (just available?)
    :return: dict, key: integer, value: list of vehicles objects.
    """
    neigh_vehs_map = defaultdict(list)
    lst = system_obj.sys_vehs if vehs_lst is None else vehs_lst
    for veh in lst:
        neigh_v = system_obj.find_geo_attr('neigh', zone_id=veh.get_zone())
        neigh_vehs_map[neigh_v].append(veh)
    return neigh_vehs_map


def get_vehs_neighborhoods(system_obj, vehs_lst=None):
    """
    Given a list of vehicles instances the function returns a list of neighborhoods indices of where these vehicles are.
    Only the keys of the mapping that is the output of the above function create_neigh_vehs_mapping.
    :param system_obj: System object
    :param vehs_lst: list of updated vehicle objects
    :return: list of ints
    """
    lst = system_obj.sys_vehs if vehs_lst is None else vehs_lst
    neigh_available = [system_obj.find_geo_attr('neigh', zone_id=veh.get_zone()) for veh in lst]
    inds_zones_available = sorted(set(neigh_available))
    return inds_zones_available


def get_vehs_zones(system_obj, vehs_lst=None):
    """
    Given a list of vehicles instances the function returns a list of neighborhoods indices of where these vehicles are.
    Only the keys of the mapping that is the output of the above function create_neigh_vehs_mapping.
    :param system_obj: System object
    :param vehs_lst: list of updated vehicle objects
    :return: list of ints
    """
    lst = system_obj.sys_vehs if vehs_lst is None else vehs_lst
    zones_vacant = [veh.get_zone() for veh in lst]
    inds_zones_available = sorted(set(zones_vacant))
    return inds_zones_available


def calc_arrival_time(veh_obj, req_obj, sys_obj):
    ok, _ = req_obj.get_od_zones()
    time_arrival = max(veh_obj.get_time(), req_obj.get_arrival_time()) + sys_obj.get_travel_time(veh_obj.get_zone(), ok)
    return time_arrival


def calc_demand_neigh_est(sys_obj, time_duration):
    """
    Calculate the number of requests that about to origin in each neighborhood in the given time duration as the
    expectation. For the probability there is another function that is being calculated once after the requests set is
    sampled.
    :param sys_obj: System object
    :param time_duration: int, the number of time units to estimate the demand arriving from each zone
    :return: np array, each elem is the number of requests that we estimate are about to arrive from each zone
    """
    # Num requests in each zone
    # geo_dist_num = sys_obj.find_geo_attr('neigh_num')
    # reqs_demand_prob = np.zeros(geo_dist_num)
    # # Probability of request to arrive from each zone
    # for z in range(sys_obj.sys_zones_size):
    #     ind_zone = sys_obj.find_geo_attr('neigh', zone_id=z)
    #     prob_orig_zone = sys_obj.sys_reqs_od_probs[z, :].sum()
    #     reqs_demand_prob[ind_zone] += prob_orig_zone
    reqs_demand_prob = sys_obj.find_geo_attr('neigh_prob')  # calc_prob_neigh_est(sys_obj)
    reqs_demand = reqs_demand_prob * sys_obj.sys_reqs_arr_prob * time_duration
    return reqs_demand


def calc_prob_neigh_est(sys_obj):
    # Num requests in each zone
    geo_dist_num = sys_obj.find_geo_attr('neigh_num')
    reqs_demand_prob = np.zeros(geo_dist_num)
    # Probability of request to arrive from each zone
    for z in range(sys_obj.sys_zones_size):
        ind_zone = sys_obj.find_geo_attr('neigh', zone_id=z)
        prob_orig_zone = sys_obj.sys_reqs_od_probs[z, :].sum()
        reqs_demand_prob[ind_zone] += prob_orig_zone
    return reqs_demand_prob


def extend_avail_vehs_with_vacant(vehs_avail_lst, system_obj, req_obj):
    """
    Given a request object, the function adds to the available vehicles list also the vehicles that are now vacant.
    :param vehs_avail_lst: list, each element is a Vehicle object of an available vehicle
    :param system_obj: a System object, updated onr
    :param req_obj: a Request object
    :return:
    """
    # Num vehicles in each area - distribution of the available vehicles + current vacant vehicles
    vehs_lst_extend = vehs_avail_lst[:]
    vehs_set_indx = [v.get_id() for v in vehs_lst_extend]
    for veh in system_obj.sys_vehs:
        # v is vacant vehicle which is not already in V script k, the available vehicles
        if veh.get_time() <= req_obj.get_arrival_time() and veh.get_id() not in vehs_set_indx:
            vehs_lst_extend.append(veh)
    return vehs_lst_extend


def write_stats_state_information(system_obj, req_obj, decision, avail_vehs):
    """
    The function stores in the system object attribute sys_G_script information regarding the states of the system at
    the decision time.
    For the data rules but also for the dispatching rules.
    :param system_obj:
    :param req_obj:
    :param decision:
    :param avail_vehs
    :return: Update the values of the dictionary with the keys:
            [geo_states, decisions, vk_size, avail_zones, avail_areas, req_attrs]
    """
    # Get record of previous requests
    k = req_obj.get_id()
    prev_geo_states = system_obj.sys_G_script.get('geo_states', [])
    prev_decisions = system_obj.sys_G_script.get('decisions', [])
    prev_vk_size = system_obj.sys_G_script.get('vk_size', [])
    prev_avail_zones = system_obj.sys_G_script.get('avail_zones', {})
    prev_avail_areas = system_obj.sys_G_script.get('avail_areas', {})
    prev_avail_zones_size = system_obj.sys_G_script.get('avail_zones_size', [])
    prev_avail_areas_size = system_obj.sys_G_script.get('avail_areas_size', [])
    prev_req_attrs = system_obj.sys_G_script.get('req_attrs', [])
    # Update according to the current state and request
    new_geo_states = prev_geo_states + [create_vehs_distribution(system_obj)]
    new_vk_size = prev_vk_size + [len(avail_vehs)]
    new_decisions = prev_decisions + [decision]
    prev_avail_areas[k] = get_vehs_neighborhoods(system_obj, vehs_lst=avail_vehs)
    new_avail_areas = prev_avail_areas.copy()
    prev_avail_zones[k] = get_vehs_zones(system_obj, vehs_lst=avail_vehs)
    new_avail_zones = prev_avail_zones.copy()
    new_avail_zones_size = prev_avail_zones_size + [len(prev_avail_zones[k])]
    new_avail_areas_size = prev_avail_areas_size + [len(prev_avail_areas[k])]
    new_req_attrs = prev_req_attrs + [[req_obj.get_arrival_time(), *req_obj.get_od_zones()]]
    # Save in the dictionary
    system_obj.sys_G_script['geo_states'] = new_geo_states
    system_obj.sys_G_script['decisions'] = new_decisions
    system_obj.sys_G_script['vk_size'] = new_vk_size
    system_obj.sys_G_script['avail_zones'] = new_avail_zones
    system_obj.sys_G_script['avail_areas'] = new_avail_areas
    system_obj.sys_G_script['avail_zones_size'] = new_avail_zones_size
    system_obj.sys_G_script['avail_areas_size'] = new_avail_areas_size
    system_obj.sys_G_script['req_attrs'] = new_req_attrs
    return


def save_geo_states_df(system_obj, warm_up_time=600):
    """
    Create the df of a simulation solution for the states distance statistic
    :param system_obj: a System object, updated one with a keep_state_information=True argument
    :param warm_up_time: the earliest time a request enters to be included in the measures
    :return: Data Frame with the shape (N, M), where N = number of requests entered after warm_up_time,
                                                and M = number of areas of each state (=12).
    """
    # Save the data from simulation
    geo_states = system_obj.find_geo_attr('geo_states')
    area_num = system_obj.find_geo_attr('neigh_num')
    df_geo_states = pd.DataFrame(geo_states, columns=[f'Area{i}' for i in range(area_num)])
    reqs_attrs = system_obj.find_geo_attr('req_attrs')
    df_req_attrs = pd.DataFrame(reqs_attrs, columns=['ArrivalTime', 'Origin', 'Destination'])
    # Create a Data Frame
    df = pd.concat([df_req_attrs, df_geo_states], axis=1)
    # Slice out all the requests that have entered in the warm up time
    df_filter = df.loc[df.ArrivalTime >= warm_up_time][[f'Area{i}' for i in range(area_num)]]
    return df_filter


def create_states_distance_statistic(dfs_dict, seed):
    """
    Returns the statistics the measure the difference of the geographical states of the dispatching rules
    :param dfs_dict: a dictionary with key = str of the rule name,
                                and value = the states df, output of the function save_geo_states_df
    :param seed: int, the seed to conduct the system instance
    :return:
    """
    comparison_dict_mean = {'Seed': seed}
    comparison_dict_sum = {'Seed': seed}
    rules = list(dfs_dict.keys())
    # Not to include the bad dispatching rules
    rules.remove('time_nearest_available')
    rules.remove('random_available')
    rules = sorted(rules)
    # Go through all the pairs of rules
    for ind in range(len(rules)):
        for jnd in range(ind + 1, len(rules)):
            rule1 = rules[ind]
            df1 = dfs_dict[rule1]
            rule2 = rules[jnd]
            df2 = dfs_dict[rule2]
            # Calc the mean and sum of the absolute number of the difference of the geographical state
            dist_mean = np.mean(np.sum(np.abs(df1 - df2), axis=1) / 2)
            dist_sum = np.sum(np.sum(np.abs(df1 - df2), axis=1) / 2)
            comparison_title = f'{rule1}_{rule2}'
            # Save in dictionary
            comparison_dict_mean[comparison_title] = dist_mean
            comparison_dict_sum[comparison_title] = dist_sum
    return comparison_dict_mean, comparison_dict_sum


def save_simulation_characteristic(system_obj, seed, rule_name, warm_up_time=600, level=1, level_dur=30*60):
    """
    Save some simulation characteristics. It can be modified to create more columns for additional characteristics.
    Saves as default the ones for level = 1 which are a 1-hour duration simulation after ten minutes of warmup.
    :param system_obj:
    :param seed:
    :param rule_name:
    :param warm_up_time:
    :param level:
    :param level_dur:
    :return: Dictionary for the final DataFrame
    """
    characteristic_dict = {'Seed': seed, 'Rule': rule_name}
    # Create DataFrame of Whole Simulation Characteristics
    df_raw = pd.concat([pd.DataFrame(system_obj.find_geo_attr('req_attrs')),
                        pd.DataFrame(system_obj.find_geo_attr('vk_size')),
                        pd.DataFrame(system_obj.find_geo_attr('avail_areas_size')),
                        pd.DataFrame(system_obj.find_geo_attr('avail_zones_size')),
                        pd.DataFrame(system_obj.find_geo_attr('decisions'))],
                       axis=1
                       )
    df_raw.columns = ['ArrivalTime', 'Origin', 'Destination', 'SizeVk', 'SizeAk', 'SizeZk', 'Decision']

    # Add new columns
    cs_dict = {**{z_cent: 'C' for z_cent in system_obj.sys_center_inx},
               **{z_sub: 'S' for z_sub in range(system_obj.sys_zones_size) if z_sub not in system_obj.sys_center_inx}}
    df_raw['OriginAreaType'] = df_raw['Origin'].map(cs_dict)
    df_raw['OriginArea'] = df_raw['Origin'].apply(lambda ok: system_obj.find_geo_attr('neigh', zone_id=ok))

    # Filter Decisions on 1-hour scenario
    df_level_1 = df_raw.copy().loc[(df_raw.ArrivalTime <= warm_up_time + (1 + level) * level_dur) &
                                   (df_raw.ArrivalTime >= warm_up_time)]

    # Save Instance Size
    characteristic_dict['reqSize'] = df_level_1.shape[0]

    # Calculate the Number of Decisions
    if rule_name in ['time_earliest_arriving', 'time_nearest_available', 'random_available']:
        df_level_1_decision_is_made = df_level_1.copy().loc[df_level_1.SizeVk >= 2]
    else:
        df_level_1_decision_is_made = df_level_1.copy().loc[df_level_1.SizeAk >= 2]
    characteristic_dict['Decision_Num'] = df_level_1_decision_is_made.shape[0]

    # Save Decision Number for each Area Type
    characteristic_dict['Decision_Mean_Origin_Area'] = df_level_1_decision_is_made['OriginArea'].mean()
    for area_type in df_level_1_decision_is_made['OriginAreaType'].unique():
        characteristic_dict[f'Decision_Num_Origin_{area_type}'] = \
            df_level_1_decision_is_made['OriginAreaType'].value_counts()[area_type]

    # Calculate F for the Ratio Number of Decisions
    if df_level_1.dtypes.Decision == 'int':
        F = np.sum(df_level_1.Decision != (-1))
    else:
        assert df_level_1.dtypes.Decision == 'O', \
            f'Unexpected type of array: {df_level_1.dtypes.Decision}. Only int/int64 and O are allowed'
        F = np.sum(~df_level_1.Decision.isin(['r', -1]))
    # Save Ratio Number of Decisions to F
    characteristic_dict['Decision_Ratio'] = characteristic_dict['Decision_Num'] / F

    # Save the Mean size of V script k and A script k (when there is a decision!):
    characteristic_dict['vk_size_mean'.title()] = df_level_1_decision_is_made['SizeVk'].mean()
    characteristic_dict['vk_size_min'.title()] = df_level_1_decision_is_made['SizeVk'].min()
    characteristic_dict['vk_size_max'.title()] = df_level_1_decision_is_made['SizeVk'].max()
    characteristic_dict['avail_areas_size_mean'.title()] = df_level_1_decision_is_made['SizeAk'].mean()
    characteristic_dict['avail_areas_size_min'.title()] = df_level_1_decision_is_made['SizeAk'].min()
    characteristic_dict['avail_areas_size_max'.title()] = df_level_1_decision_is_made['SizeAk'].max()
    return characteristic_dict

