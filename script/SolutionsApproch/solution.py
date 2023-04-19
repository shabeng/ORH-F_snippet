# Python Packages
import pandas as pd
from itertools import chain
import pickle

# My Repo
from script.utilis.utilis import calc_r_q_G, get_value_df_cell, calc_objective_z, calc_x_G, calc_objectives


class Solution:
    def __init__(self, off_problem):
        # self.problem = off_problem
        self.objective_name = off_problem.objective
        self.gap = off_problem.prob.solution.MIP.get_mip_relative_gap()
        self.solution_full = pd.DataFrame()
        self.solution_full['varNames'] = off_problem.prob.variables.get_names()
        self.solution_full['varValues'] = off_problem.prob.solution.get_values()
        self.solution_active = self.solution_full.loc[self.solution_full.varValues > 0 + 10**(-5)]
        self.objective_value = self.set_real_objective_value(off_problem)
        self.routes = self.build_routes(off_problem)
        self.reqs_not_served = self.find_reqs_not_served(off_problem)
        self.validation = {}
        self.x_G = self.calc_x_G(off_problem, warmup_bol=False)  # without the warm-up phase
        self.r_G, self.q_G = calc_r_q_G(off_problem.system)  # w.o. warm-up
        self.objective_value_wu = self.eval_objective_warmup(off_problem)
        self.objective_value_other, self.objective_value_other_wu = \
            self.eval_other_objective_value(off_problem)

    def set_real_objective_value(self, prob_obj):
        if self.objective_name in ['Profit', 'Z']:
            val = prob_obj.prob.solution.get_objective_value()
        else:
            # self.objective_name == 'RequstsWaiting'
            t_k = self.solution_full.loc[self.solution_full.varNames.str.contains('t_')].varValues.values
            a_k = prob_obj.timewindow_k[:, 0]
            val = t_k.sum() - a_k.sum()
        return val

    def build_routes(self, off_problem_obj, mask_cond=''):
        routes = {}
        for veh in off_problem_obj.system.sys_vehs:
            v_id = veh.v_id
            mask = lambda var, num: self.solution_active.varNames.str.contains(f'{var}_{num},')
            req_next = get_value_df_cell(self.solution_active.loc[mask('y', v_id)].varNames)
            if req_next == -1:
                continue
            route = [req_next]
            end_route = False
            while not end_route:
                mask_vehicle = f'{mask_cond}{v_id}' if mask_cond == '_' else mask_cond
                req_next = get_value_df_cell(self.solution_active.loc[mask('x'+mask_vehicle, req_next)].varNames)
                if req_next == -1:
                    end_route = True
                    continue
                route.append(int(req_next))
            routes[v_id] = route
        return routes

    def find_reqs_not_served(self, problem_obj):
        reqs_served_ids = list(chain.from_iterable(self.routes.values()))
        reqs_not_served_ids = [req.k_id for req in problem_obj.system.sys_reqs if req.k_id not in reqs_served_ids]
        return reqs_not_served_ids

    def calc_x_G(self, problem_obj, warmup_bol=False):
        system_obj = problem_obj.system
        x_G = calc_x_G(system_obj, self.routes, warmup_bol)
        return x_G

    def calc_z(self):
        return calc_objective_z(self.x_G, self.q_G)

    def calc_profit(self, off_problem_obj, warmup_bol=False):
        # we need to evaluate the profit value
        warm_up_req = off_problem_obj.system.warmup if warmup_bol else 0
        profit = 0
        for v_id, route in self.routes.items():
            for lst_inx, req_id in enumerate(route):
                if req_id > warm_up_req:
                    if lst_inx == 0:
                        profit += off_problem_obj.profits_v[v_id - 1, req_id - 1]
                    else:
                        profit += off_problem_obj.profits_k[route[lst_inx - 1] - 1, req_id - 1]
        return profit

    def update_validation(self, route_valid_dict):
        for key in route_valid_dict.keys():
            if key == 'order':
                prev_val = self.validation.get(key, True)
                updated = prev_val and route_valid_dict[key]
            else:
                prev_dict = self.validation.get(key, {})
                updated = {**prev_dict, **route_valid_dict[key]}
            self.validation[key] = updated

    def eval_objective_warmup(self, off_problem_obj):
        if self.objective_name == 'Profit':
            cum_sum = 0
            for v_id, route in self.routes.items():
                for lst_inx, req_id in enumerate(route):
                    if req_id <= off_problem_obj.system.warmup:
                        if lst_inx == 0:
                            cum_sum += off_problem_obj.profits_v[v_id - 1, req_id - 1]
                        else:
                            cum_sum += off_problem_obj.profits_k[route[lst_inx - 1] - 1, req_id - 1]
            profit_warmup = self.objective_value - cum_sum
            return profit_warmup
        else:
            x_G_warmup = self.calc_x_G(off_problem_obj, warmup_bol=True)
            q_G_warmup = off_problem_obj.system.sys_q_G
            Z_warmup = calc_objective_z(x_G_warmup, q_G_warmup)
            return Z_warmup

    def eval_other_objective_value(self, off_problem_obj):
        if self.objective_name != 'Z':
            # we need to evaluate the z value
            x_G_wu = self.calc_x_G(off_problem_obj, warmup_bol=True)
            q_G_wu = off_problem_obj.system.sys_q_G
            Z_wu = calc_objective_z(x_G_wu, q_G_wu)
            Z = self.calc_z()
            return Z, Z_wu

        else:
            # we need to evaluate the profit value
            profit_wu = self.calc_profit(off_problem_obj, warmup_bol=True)
            profit = self.calc_profit(off_problem_obj, warmup_bol=False)
            return profit, profit_wu

    def save_solution(self, path, seed, init_vehs_lst, init_reqs_lst, time_mat, compute_time, init_sol_val,
                      is_reject, reject_reqs_k_ids):
        """
        Save a pickle of the solution
        :param path: str, specify the directory to save the picke file
        :param seed: int, correspond the seed to create the system instance (other system parameters are given in
        separate .py file).
        :param init_vehs_lst
        :param init_reqs_lst
        :param time_mat
        :param compute_time
        :param init_sol_val
        :param is_reject
        :param reject_reqs_k_ids
        :return:
        """
        sol_dict = {'seed': seed, 'gap': self.gap, 'routes': self.routes, 'milp_obj_val': self.objective_value,
                    'vehs': init_vehs_lst, 'reqs': init_reqs_lst, 't_ij': time_mat, 'comp_time': compute_time,
                    'initial_sol_val': init_sol_val, 'is_reject': is_reject, 'rejected_kds': reject_reqs_k_ids}
        z, gini, eff = calc_objectives(self.x_G, self.q_G)
        sol_dict['Z'] = z
        sol_dict['gini'] = gini
        sol_dict['F'] = eff
        with open(path, 'wb') as handle:
            pickle.dump(sol_dict, handle)


class OnlineSolution(Solution):
    def __init__(self, online_problem):
        self.objective_name = online_problem.objective
        self.objective_value = online_problem.prob.solution.get_objective_value()
        self.gap = online_problem.prob.solution.MIP.get_mip_relative_gap()
        self.solution_full = pd.DataFrame()
        self.solution_full['varNames'] = online_problem.prob.variables.get_names()
        self.solution_full['varValues'] = online_problem.prob.solution.get_values()
        self.solution_active = self.solution_full.loc[self.solution_full.varValues > 0 + 10 ** -6]
        self.routes = self.build_routes(online_problem, '_')
        self.reqs_not_served = self.find_reqs_not_served(online_problem)
        self.validation = {}
        self.x_G = self.calc_x_G(online_problem, warmup_bol=False)  # without the warm-up phase
        self.r_G, self.q_G = calc_r_q_G(online_problem.system)  # w.o. warm-up
        self.objective_value_wu = self.eval_objective_warmup(online_problem)
        self.objective_value_other, self.objective_value_other_wu = \
            self.eval_other_objective_value(online_problem)

    def save_solution(self, path, seed, init_vehs_lst, init_reqs_lst, time_mat, compute_time, init_sol_val,
                      is_reject=False, reject_reqs_k_ids=()):
        super().save_solution(path=path, seed=seed, init_vehs_lst=init_vehs_lst, init_reqs_lst=init_reqs_lst,
                              time_mat=time_mat, compute_time=compute_time, init_sol_val=init_sol_val,
                              is_reject=is_reject, reject_reqs_k_ids=list(reject_reqs_k_ids))
