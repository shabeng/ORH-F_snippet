"""Running the program"""
# Python Packages
import numpy as np
import time
import pickle
import sys

# My repo
from script.ProblemObjects import system as s
from script.SolutionsApproch.Simulation.simulation import run_simulation_rule
from script.SolutionsApproch.Simulation.dispatch import time_earliest_arriving, 
from script.utilis.utilis import create_simu_sol_to_milp_sol, list_to_dict_solution, copy_objects_lst, \
    assert_system_settings
from script.SolutionsApproch.MILP.offlineMILP import OfflineProblem

# System parameters
from experiments import system_param as sp

# Create an instance of the system for the MILP
seed = 10020
system_instance = s.System(seed_num=seed, num_time_points=sp.T, num_vehicles=sp.V, travel_time_mat=sp.t_ij,
                           reqs_arr_prob=sp.reqs_arr_p, reqs_od_probs=sp.reqs_od_p_mat, payment_func=sp.pay_func,
                           G_script=sp.requests_group.copy(), city_center_zones=sp.center_zones_inxs,
                           vehs_center_only=False, warm_up_reqs=sp.warmup_reqs_num,
                           expiration_method=sp.expiration_method,
                           fixed_exp_c=sp.expiration_dur_c, fixed_exp_s=sp.expiration_dur_s)

# Solve its offline variation of the problem with the MILP model using CPLEX api for python
objective_name = 'Z'

# MILP Model
start_time = time.time()
off_prob = OfflineProblem(system_instance, objective=objective_name, time_limit=60 * 60 * 12)
print(f'Start Adding Variables')
off_prob.add_decision_variables()
var_time = time.time()
print(f'Added Variables: {var_time - start_time}')
print(f'Start Adding Constraints')
off_prob.add_constraints()
cons_time = time.time()
print(f'Added Constraints: {cons_time - var_time}')

# Solve model
solve_time = time.time()
path_str = f'experiments/cplex_log_seed{seed}.log'
with open(path_str, 'w') as cplex_log:
    off_prob.prob.set_results_stream(cplex_log)
    off_prob.prob.set_warning_stream(cplex_log)
    off_prob.prob.set_error_stream(cplex_log)
    off_prob.prob.set_log_stream(cplex_log)
    off_prob.solve_model()
end_time = time.time()
off_prob.validate_sol()
print(f'MILP Optimal solution value: Z = {off_prob.solution.objective_value}')

# Solve the online problem with the benchmark dispatching rules
# Create an instance of the system to solve the online variation with the basic decision rules

# Show results with the bar plot

