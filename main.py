"""Running the program"""
# Python Packages
import numpy as np
import time
import pickle
import sys

# My repo
from script.ProblemObjects import system as s
from script.SolutionsApproch.Simulation.simulation import run_simulation_rule
from script.SolutionsApproch.Simulation.dispatch import time_earliest_arriving
from script.utilis.utilis import create_simu_sol_to_milp_sol, list_to_dict_solution, copy_objects_lst, \
    assert_system_settings
from script.SolutionsApproch.MILP.offlineMILP import OfflineProblem

# System parameters
from experiments import system_param as sp

# Create an instance of the system
seed = 123456789
system_instance = s.System(seed_num=seed, num_time_points=sp.T, num_vehicles=sp.V, travel_time_mat=sp.t_ij,
                           reqs_arr_prob=sp.reqs_arr_p, reqs_od_probs=sp.reqs_od_p_mat, payment_func=sp.pay_func,
                           G_script=sp.requests_group.copy(), city_center_zones=sp.center_zones_inxs,
                           vehs_center_only=False, warm_up_reqs=sp.warmup_reqs_num,
                           expiration_method=sp.expiration_method,
                           fixed_exp_c=sp.expiration_dur_c, fixed_exp_s=sp.expiration_dur_s)

# Solve its offline variation of the problem with the MILP model using CPLEX api for python
objective_name = 'Z'

# Solve the online problem with the benchmark dispatching rules

# Show results with the bar plot

