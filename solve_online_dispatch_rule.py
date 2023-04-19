# Python Packages
import pandas as pd
import numpy as np
import pickle
import sys
from os import listdir
from os.path import isfile, join
import time
from collections import defaultdict

# My Repo
import script.ProblemObjects.system as s
from script.SolutionsApproch.Simulation.simulation import run_simulation_rule
from script.SolutionsApproch.Simulation.dispatch import time_earliest_vacant, time_earliest_arriving, \
    time_nearest_available, random_available, \
    balance_crowded_zone, balance_balanced_zone
from script.utilis.utilis import calc_partial_measures_of_sol
from script.utilis.utilis_data_driven_rule import create_seeds_lst_for_refer
from script.utilis.utilis_simulation import save_geo_states_df, create_states_distance_statistic, \
    save_simulation_characteristic

# System parameters
from experiments import system_param as sp

# Create an instance of the system to solve the online problem with the basic dispatching rules
seed = 999221
rules = [
    time_earliest_arriving, time_nearest_available, random_available,
    balance_crowded_zone, balance_balanced_zone,
]
n_levels = 20
level_dur_time = 30 * 60

measures = ['full', f'level_{n_levels}']

Z_measures_dict_seed = defaultdict(dict)
F_measures_dict_seed = defaultdict(dict)

print(f'Running {len(rules)} Rules - Good Luck! :)')
warm_up_time = 600

time_start_rule = time.time()
for m in measures:
    Z_measures_dict_seed[m]['Seed'] = seed
    F_measures_dict_seed[m]['Seed'] = seed

for ind, rule in enumerate(rules):
    rule_name = str(rule).split(' ')[1]
    system_rule = s.System(seed, sp.T + n_levels*level_dur_time, sp.V, sp.t_ij, sp.reqs_arr_p, sp.reqs_od_p_mat,
                           sp.pay_func, sp.requests_group.copy(), sp.center_zones_inxs, sp.warmup_reqs_num,
                           expiration_method=sp.expiration_method,
                           fixed_exp_c=sp.expiration_dur_c, fixed_exp_s=sp.expiration_dur_s)

    if ind in [0, 1, 2]:
        # Vehicle level rules
        z_vals, f_vals, r_Gs, solution_lst, pickup_times = \
            run_simulation_rule(system_obj=system_rule,
                                rule_func=rule,
                                warm_up_func=time_earliest_arriving,
                                warm_up_time=warm_up_time,
                                condition=lambda req_, sys_: req_.get_arrival_time() >= warm_up_time,
                                keep_state_information=False)
    else:
        func_tie = time_earliest_arriving
        rule_name = rule_name + '_EA'
        # Area level rules
        z_vals, f_vals, r_Gs, solution_lst, pickup_times = \
            run_simulation_rule(system_obj=system_rule,
                                rule_func=rule,
                                warm_up_func=time_earliest_arriving,
                                warm_up_time=warm_up_time,
                                area_vehs_select_func=func_tie,
                                condition=lambda req_, sys_: req_.get_arrival_time() >= warm_up_time,
                                keep_state_information=False)

    for m in measures:
        if m == 'full':
            if ind == 0:
                assert system_rule.sys_reqs_size == r_Gs[0].sum(), 'r_G and |K| are not equal for full, line 182'
                Z_measures_dict_seed[m]['reqNum'] = system_rule.sys_reqs_size
                F_measures_dict_seed[m]['reqNum'] = system_rule.sys_reqs_size
            Z_measures_dict_seed[m][rule_name] = z_vals[0]
            F_measures_dict_seed[m][rule_name] = f_vals[0]
        elif m == f'level_{n_levels}':
            if ind == 0:
                Z_measures_dict_seed[m]['reqNum'] = r_Gs[1].sum()
                F_measures_dict_seed[m]['reqNum'] = r_Gs[1].sum()
            Z_measures_dict_seed[m][rule_name] = z_vals[1]
            F_measures_dict_seed[m][rule_name] = f_vals[1]
            # Fill rate by group:
            NL = int(n_levels)
            z_level_all, f_level_all, r_g_level_all, x_g_level_all = \
                calc_partial_measures_of_sol(sol_lst=solution_lst, system_obj=system_rule,
                                             partial_cond=lambda req_, sys_: req_.get_arrival_time() >= warm_up_time)
            assert z_level_all == z_vals[1], f'Different Z values! seed = {seed} | rule = {rule_name} | level = 3'
            assert f_level_all == f_vals[1], f'Different F values! seed = {seed} | rule = {rule_name} | level = 3'


