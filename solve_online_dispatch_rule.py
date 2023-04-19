# Python Packages
import pandas as pd

import time
from collections import defaultdict

# My Repo
import script.ProblemObjects.system as s
from script.SolutionsApproch.Simulation.simulation import run_simulation_rule
from script.SolutionsApproch.Simulation.dispatch import time_earliest_arriving, \
    time_nearest_available, random_available, \
    balance_crowded_zone, balance_balanced_zone
from script.utilis.utilis import calc_partial_measures_of_sol
from script.utilis.utilis_plot import plot_bars_results

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

Z_measures_dict_seed = defaultdict(dict)
F_measures_dict_seed = defaultdict(dict)

print(f'Running {len(rules)} Rules - Good Luck! :)')
warm_up_time = 600

time_start_rule = time.time()

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

    if ind == 0:
        Z_measures_dict_seed['reqNum'] = r_Gs[1].sum()
        F_measures_dict_seed['reqNum'] = r_Gs[1].sum()
    Z_measures_dict_seed[rule_name] = z_vals[1]
    F_measures_dict_seed[rule_name] = f_vals[1]
    # Fill rate by group:
    NL = int(n_levels)
    z_level_all, f_level_all, r_g_level_all, x_g_level_all = \
        calc_partial_measures_of_sol(sol_lst=solution_lst, system_obj=system_rule,
                                     partial_cond=lambda req_, sys_: req_.get_arrival_time() >= warm_up_time)
    assert z_level_all == z_vals[1], f'Different Z values! seed = {seed} | rule = {rule_name} | level = 3'
    assert f_level_all == f_vals[1], f'Different F values! seed = {seed} | rule = {rule_name} | level = 3'


# Show the results:
measures = ['Z', 'F', 'GI']
df = pd.DataFrame()
df['Z'] = Z_measures_dict_seed.values()
df.index = Z_measures_dict_seed.keys()
df['F'] = F_measures_dict_seed.values()
df['GI'] = df.Z / df.F
df.drop('reqNum', inplace=True)

df_norm = df.copy()
for measure in measures:
    df_norm[measure] = ((df_norm[measure] / df_norm[measure].time_earliest_arriving) - 1) * 100

plot_bars_results(3, [df_norm.index[1:]] * 3, [df_norm[measure][1:] for measure in measures],
                  x_ticks=['CV', 'R', 'MC', 'MC-N'], y_label='Increase [%]',
                  data_rules_num=0, color_ours=(40 / 250, 90 / 250, 224 / 250),
                  plus_label=0.13, minus_label=0.33, l_lim=-4, h_lim=2.5, bar_width=0.77,
                  sub_titles_lst=['Z', 'F', 'GI'],
                  main_title=f'Increase [%] Relative to EA - Seed = {seed}', dpi=500, fig_size=(4, 3),
                  bar_fontsize=6.5, yticks_label_fontsize=5,
                  save_fig=True, path_save='experiments', graph_name=f'online_seed_{seed}')
