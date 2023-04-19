import numpy as np
import pickle
from os import listdir
from os.path import isfile, join

from script.utilis.utilis import assert_system_settings
import script.ProblemObjects.system as s


def create_seeds_lst_for_refer(path_to_sol, system_param, gap_th):
    # Get fixed system settings
    with open(f'{path_to_sol}/sys_dict_prev.pickle', 'rb') as handle:
        sys_settings = pickle.load(handle)
    # Get seeds of solutions
    files = [f for f in listdir(path_to_sol) if isfile(join(path_to_sol, f))]
    seeds_opt = [int(f[13:-7]) for f in files if f[0:3] == 'sol']
    # Add to the reference only solutions with gap lower than threshold
    seeds_opt_filter = []
    for seed_ind, seed in enumerate(seeds_opt):
        with open(f'{path_to_sol}/solution_seed{seed}.pickle', 'rb') as handle:
            sol_dict = pickle.load(handle)
            if sol_dict['gap'] <= gap_th:
                system_opt_seed = s.System(seed, system_param.T, system_param.V, system_param.t_ij,
                                           system_param.reqs_arr_p, system_param.reqs_od_p_mat, system_param.pay_func,
                                           system_param.requests_group.copy(), system_param.center_zones_inxs,
                                           system_param.warmup_reqs_num,
                                           expiration_method=system_param.expiration_method,
                                           fixed_exp_c=system_param.expiration_dur_c,
                                           fixed_exp_s=system_param.expiration_dur_s)
                assert_system_settings(system_opt_seed, sys_settings, f'System Opt Sol | Seed = {seed}')
                seeds_opt_filter.append(seed)
    return seeds_opt_filter, sys_settings
