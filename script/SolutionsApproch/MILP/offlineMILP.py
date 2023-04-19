import cplex
import numpy as np
import bisect

from script.utilis.utilis import calc_r_q_G
from script.SolutionsApproch.Simulation.simulation import run_simulation_route_validation
from script.SolutionsApproch.solution import Solution


class OfflineProblem:
    def __init__(self, system_object, objective='Profit', time_limit=60 * 60):
        """
        This model is model 3 with the green correction in respect to Bertsimas et al. suggested model.
        Thus it is online but can reject request that there is an availabe vehicle to serve it.
        contain all the input of the model which is created from the system object attributes.
        :param system_object:
        :param objective: string, 'Profit' or 'Z'
        """
        # System Object
        self.system = system_object
        self.objective = objective
        self.n = len(self.system.sys_reqs)  # The number of requests without the warmup
        _, self.q_G = calc_r_q_G(self.system)
        self.group_n = self.system.sys_r_G.shape[0]
        self.request_groups = {i: [req.k_id for req in self.system.sys_reqs
                                   if (req.find_request_type(self.system) == i)]
                               for i in range(self.group_n)}
        # in if and k.k_id > self.system.warmup

        # Request to Request Arcs
        self.traveltime_k = np.zeros((self.n, self.n))  # |K|*|K|, travel time = t_ok'_dk' + t_dk'_ok
        self.cruising_k = np.zeros((self.n, self.n))  # |K|*|K|, cruising time = t_dk'_ok
        self.profits_k = np.zeros((self.n, self.n))  # |K|*|K|, profit = p_k - (t_dk'_ok + t_ok_dk)

        # Vehicle to Request Arcs
        self.traveltime_v = np.zeros((self.system.sys_V, self.n))  # |V|*|K|, travel time = t_zv_ok
        self.profits_v = np.zeros((self.system.sys_V, self.n))  # |V|*|K|, profit = p_k - (t_zv_ok + t_ok_dk)

        # Request Time Windows
        self.timewindow_k = np.zeros((self.n, 2))  # |K|*2

        # Model Object
        self.prob = cplex.Cplex()
        self.prob.parameters.timelimit.set(time_limit)
        # avoid from printing the results:
        # self.prob.set_log_stream(None)
        # self.prob.set_error_stream(None)
        # self.prob.set_warning_stream(None)
        # self.prob.set_results_stream(None)

        # Solution
        self.solution = None

        # Graph Arcs - list and dictionary
        self.arcs_k = []  # lst of (k', k) requests ids (original start from 1) of existing arcs
        self.out_arcs_k = {}  # dict - key: k' request id, value: list of indexes of the arcs that matches requests ids (k) that can be followed
        self.in_arcs_k = {}  # dict - key: k request id, value: list of indexes of the arcs that matches requests ids (k') that can be before it

        self.arcs_v = []  # lst of (v, k) vehicle-request ids (original start from 1) of existing arcs
        self.out_arcs_v = {}  # dict - key: v vehicle id, value: list of indexes of the arcs that matches all requests ids (k) that can be followed
        self.in_arcs_v = {}  # dict - key: k request id, value: list of indexes of the arcs that matches all vehicle ids (v) that can be before it

        self.create_parameters()

    def create_parameters(self):
        """
        Creates the full matrices of the model. Not all of this arcs actually exists in the graph.
        :return:
        """
        # Time window of the requests
        for req in self.system.sys_reqs:
            self.timewindow_k[req.k_id - 1] = req.k_arrival_time, req.k_expiration_time

        # Travel Time and Profit of Requests nodes
        for k_ in self.system.sys_reqs:
            for k in self.system.sys_reqs:
                if k_.k_id >= k.k_id:
                    continue
                trip_time_k_ = self.system.get_travel_time(*k_.get_od_zones())
                trip_time_k = self.system.get_travel_time(*k.get_od_zones())
                cruise_time = self.system.get_travel_time(k_.get_od_zones()[1], k.get_od_zones()[0])

                # travel time between requests nodes = t_ok'_dk' + t_dk'_ok
                self.traveltime_k[k_.k_id - 1, k.k_id - 1] = trip_time_k_ + cruise_time

                # travel time between k_ destination to k origin = t_dk'_ok
                self.cruising_k[k_.k_id - 1, k.k_id - 1] = cruise_time

                # profit between requests nodes = p_k - (t_dk'_ok + t_ok_dk)
                self.profits_k[k_.k_id - 1, k.k_id - 1] = self.system.calc_payment(k) - (cruise_time + trip_time_k)

                # Does the arc exists?
                arrival_k_ = self.timewindow_k[k_.k_id - 1][0]
                expiration_k = self.timewindow_k[k.k_id - 1][1]
                if arrival_k_ + self.traveltime_k[k_.k_id - 1, k.k_id - 1] <= expiration_k:  # and (k_.k_id < k.k_id):
                    self.arcs_k.append((k_.k_id, k.k_id))

                    prev_lst_out = self.out_arcs_k.get(k_.k_id, [])
                    self.out_arcs_k[k_.k_id] = prev_lst_out + [k.k_id]

                    prev_lst_in = self.in_arcs_k.get(k.k_id, [])
                    self.in_arcs_k[k.k_id] = prev_lst_in + [k_.k_id]

        # Travel Time and Profit of Vehicle-Request nodes
        for v in self.system.sys_vehs:
            for k in self.system.sys_reqs:
                trip_time_k = self.system.get_travel_time(*k.get_od_zones())
                cruise_time = self.system.get_travel_time(v.get_zone(), k.get_od_zones()[0])

                # travel time between vehicle-request nodes = t_v (initial vacancy time) + t_zv_ok
                self.traveltime_v[v.v_id - 1, k.k_id - 1] = v.get_time() + cruise_time

                # profit between vehicle-request nodes = p_k - (t_zv_ok + t_ok_dk)
                self.profits_v[v.v_id - 1, k.k_id - 1] = self.system.calc_payment(k) - (cruise_time + trip_time_k)

                # Does the arc exists?
                expiration_k = self.timewindow_k[k.k_id - 1][1]
                if self.traveltime_v[v.v_id - 1, k.k_id - 1] <= expiration_k:
                    self.arcs_v.append((v.v_id, k.k_id))

                    prev_lst = self.out_arcs_v.get(v.v_id, [])
                    self.out_arcs_v[v.v_id] = prev_lst + [k.k_id]

                    prev_lst = self.in_arcs_v.get(k.k_id, [])
                    self.in_arcs_v[k.k_id] = prev_lst + [v.v_id]

    def create_vars_binary_arc(self, var_prefix):
        """
        Given the variable name, the function return the neccesary list for the cplex to create the variable group that
        has a variable for each of the arcs defined in the graph.
        It first find the relevant parameters for the variable group and then creates the lists which include
        the objective function coefficients, the variable names and its types.
        :param var_prefix: string x or y, but also can be extend to include x_v_ for the OnlineModel
        :return: obj_coeffs (which considers the objective), vars_names, types
        """
        arcs = self.arcs_k if var_prefix[0] == 'x' else self.arcs_v
        coeffs = self.profits_k if var_prefix[0] == 'x' else self.profits_v
        vars_names = [f'{var_prefix}_{i},{j}' for i, j in arcs]
        types = 'B' * len(arcs)
        if self.objective == 'Profit' and var_prefix != 'z':
            obj_coeffs = [coeffs[i - 1][j - 1] for i, j in arcs]
        else:
            obj_coeffs = [0] * len(arcs)
        return obj_coeffs, vars_names, types

    def create_vars_reqs(self, var_prefix, var_type):
        """
        Given the variable name, the function return the neccesary list for the cplex to create the variable group that
        has a variable for each of the requests group K script.
        It uses the system object requests group K script and creates the lists which include the objective function
        coefficients, the variable names and its types.
        :param var_prefix: string 'p' or 't', but also can be extend to include p_v_, t_v_, or u_v_ for the OnlineModel
        :param var_type: string 'B' or 'C' respectively for the variable Binary or Continuous
        :return: obj_coeffs (which considers the objective), vars_names, types
        """
        vars_names = [f'{var_prefix}_{k.k_id}' for k in self.system.sys_reqs]
        types = var_type * self.n
        if self.objective == 'Profit' or var_prefix[0] == 'u':
            obj_coeffs = [0] * self.n
        else:
            if self.objective == 'Z':
                coeff = 1 if var_prefix[0] == 'p' else 0
                # Z objective that consider the warmup phase
                # obj_coeffs = [0] * self.system.warmup + [1] * self.system.sys_reqs_size + [0] * self.n
            elif self.objective == 'RequestsWaiting':
                coeff = (-1)*self.system.sys_T if var_prefix[0] == 'p' else 1
            else:
                coeff = 0
            obj_coeffs = [coeff] * self.n
        return obj_coeffs, vars_names, types

    def create_vars_abs(self):
        """
        Create the necessary list for linearizing the absolute number operator for Z objective function.
        It creates a continuous variable for each couple of requests groups.
        The number of couples is determined as the sum of arithmetic series sum
        :return: obj_coeffs, vars_names, types
        """
        var_num = sum(range(self.group_n))  # arithmetic series sum
        obj_coeffs = [-1] * var_num
        vars_names = [f'W_G{i},G{j}' for i in range(self.group_n) for j in range(i + 1, self.group_n)]
        types = 'C' * var_num
        return obj_coeffs, vars_names, types

    def add_decision_variables(self):
        """
        Create the model by rows - first create all the variables and then add the corresponding constraints.
        :return:
        """
        # Add x_k'k and y_vk - binary variables for existing arcs between requests or vehicle-request
        x_obj_coeffs,  x_vars_names, x_types = self.create_vars_binary_arc('x')
        y_obj_coeffs, y_vars_names, y_types = self.create_vars_binary_arc('y')

        # Add p_k and t_k - binary variables and continues variables for any request
        p_obj_coeffs, p_vars_names, p_types = self.create_vars_reqs('p', 'B')
        t_obj_coeffs, t_vars_names, t_types = self.create_vars_reqs('t', 'C')

        vars_obj_coeffs = x_obj_coeffs + y_obj_coeffs + p_obj_coeffs + t_obj_coeffs
        vars_names = x_vars_names + y_vars_names + p_vars_names + t_vars_names
        vars_types = x_types + y_types + p_types + t_types

        # Add W_GiGj - continuous variable for any couple of request groups
        if self.objective == 'Z':
            w_obj_coeffs, w_var_names, w_types = self.create_vars_abs()
            vars_obj_coeffs += w_obj_coeffs
            vars_names += w_var_names
            vars_types += w_types

        self.prob.variables.add(obj=vars_obj_coeffs, names=vars_names, types=vars_types)

    def create_const_num1(self, var_prefix=''):
        """
        Create the arguments needed for constrain number (1) in the corrected Bertsimas et al. model (Model 3).
        It assign whether an request k is served or not.
        :param var_prefix: string '' (empty) or '<_v>' where v is int of a vehicle id that specify if it is for
        a vehicle v (as in Model 3.1).
        :return: constrain_expr (nested list, a list for each constrain that has 2 lists - 1 for var names,
        2 for var coefficients), senses, rhs
        """
        constrain_expr = []
        for k in self.system.sys_reqs:
            constrain_var = [f'p{var_prefix}_{k.k_id}']
            if var_prefix == '':
                constrain_var += [f'y_{v},{k.k_id}' for v in self.in_arcs_v.get(k.k_id, [])]
            else:
                v_id = int(var_prefix.split('_')[1])
                if v_id in self.in_arcs_v.get(k.k_id, []):
                    constrain_var += [f'y{var_prefix},{k.k_id}']
            constrain_var += [f'x{var_prefix}_{k_},{k.k_id}' for k_ in self.in_arcs_k.get(k.k_id, []) if k_ < k.k_id]
            constrain_coeff = [-1] + [1] * (len(constrain_var) - 1)
            constrain_expr.append([constrain_var, constrain_coeff])
        senses = 'E' * len(constrain_expr)
        rhs = [0] * len(constrain_expr)
        return constrain_expr, senses, rhs

    def create_const_num2(self, var_prefix=''):
        """
        Create the arguments needed for constrain number (2) in the corrected Bertsimas et al. model (Model 3).
        It assign the next request k if k_ is served to be at most 1, or 0 if k_ is not served.
        :param var_prefix: string '' (empty) or '<_v>' where v is int of a vehicle id that specify if it is for
        a vehicle v (as in Model 3.1).
        :return: constrain_expr, senses, rhs
        """
        constrain_expr = []
        for k_ in self.system.sys_reqs:
            constrain_var = [f'p{var_prefix}_{k_.k_id}'] + \
                            [f'x{var_prefix}_{k_.k_id},{k}' for k in self.out_arcs_k.get(k_.k_id, []) if k > k_.k_id]
            constrain_coeff = [-1] + [1] * (len(constrain_var) - 1)
            constrain_expr.append([constrain_var, constrain_coeff])
        senses = 'L' * len(constrain_expr)
        rhs = [0] * len(constrain_expr)
        return constrain_expr, senses, rhs

    def create_const_num3(self, var_prefix=''):
        """
        Create the arguments needed for constrain number (3) in the corrected Bertsimas et al. model (Model 3).
        It assign the first request k served by vehicle v (if it serves any).
        :param var_prefix: string '' (empty) or '<_v>' where v is int of a vehicle id that specify if it is for
        a vehicle v (as in Model 3.1).
        :return: constrain_expr, senses, rhs
        """
        constrain_expr = []
        for v in self.system.sys_vehs:
            constrain_var = [f'y_{v.v_id},{k}' for k in self.out_arcs_v.get(v.v_id, [])]
            constrain_coeff = [1] * len(constrain_var)
            constrain_expr.append([constrain_var, constrain_coeff])
        senses = 'L' * len(constrain_expr)
        rhs = [1] * len(constrain_expr)
        return constrain_expr, senses, rhs

    def create_const_num4(self, var_prefix=''):
        """
        Create the arguments needed for constrain number (4) in the corrected Bertsimas et al. model (Model 3).
        It assign the time to serve request k considering the time window [ak,ek] and cruising time from previous request
        destination.
        :param var_prefix: string '' (empty) or '<_v>' where v is int of a vehicle id that specify if it is for
        a vehicle v (as in Model 3.1).
        :return: constrain_expr, senses, rhs
        """
        constrain_expr, senses, rhs = [], '', []
        for k in self.system.sys_reqs:
            # Constrain Lower Bound
            constrain_var = [f't{var_prefix}_{k.k_id}'] + \
                            [f'x{var_prefix}_{k_},{k.k_id}' for k_ in self.in_arcs_k.get(k.k_id, [])]
            constrain_coeff = [1] + \
                              [(-1) * self.cruising_k[k_ - 1][k.k_id - 1] for k_ in self.in_arcs_k.get(k.k_id, [])]
            # Removed for both the offline and online problem the y_vk variables cruising time since it is included in
            # other constrain in the model: constrain 6 in offline, constrain 8 in online.
            constrain_expr.append([constrain_var, constrain_coeff])
            # Constrain Upper Bound
            constrain_expr += [[[f't{var_prefix}_{k.k_id}'], [1]]]
            senses = senses + 'GL'
            rhs += list(self.timewindow_k[k.k_id - 1])
            # Correction for the online model - another constrain Upper Bound
            if var_prefix != '':
                # This is the online model
                big_M = self.system.sys_T + self.system.sys_tau_max
                # u_v_k
                constrain_var = constrain_var + [f'u{var_prefix}_{k.k_id}']
                constrain_coeff = constrain_coeff + [(-1) * big_M]
                constrain_expr.append([constrain_var, constrain_coeff])
                # u_v_k',k
                # constrain_var.append(f'u{var_prefix}_{k.k_id}')
                senses = senses + 'L'
                rhs.append(self.timewindow_k[k.k_id - 1][0])
        return constrain_expr, senses, rhs

    def create_const_num5(self, var_prefix=''):
        """
        Create the arguments needed for constrain number (5) in the corrected Bertsimas et al. model (Model 3).
        It assign the time to serve request k considering the time window cruising time after serving the prev request.
        :param var_prefix: string '' (empty) or '<_v>' where v is int of a vehicle id that specify if it is for
        a vehicle v (as in Model 3.1).
        :return: constrain_expr, senses, rhs
        """
        constrain_expr, rhs = [], []
        for k_, k in self.arcs_k:
            constrain_var = [f't{var_prefix}_{k}', f't{var_prefix}_{k_}', f'x{var_prefix}_{k_},{k}']
            constrain_coeff = [1, -1, -(self.traveltime_k[k_ - 1][k - 1] -
                                        (self.timewindow_k[k - 1][0] - self.timewindow_k[k_ - 1][1]))]
            constrain_expr.append([constrain_var, constrain_coeff])
            rhs.append(self.timewindow_k[k - 1][0] - self.timewindow_k[k_ - 1][1])
            # Correction for the online model - another constrain Upper Bound
            if var_prefix != '':
                # This is the online model
                big_M = self.system.sys_T + self.system.sys_tau_max
                # u_v_k
                constrain_var = constrain_var + [f'u{var_prefix}_{k}']
                # u_v_k'k
                # constrain_var = constrain_var + [f'u{var_prefix}_{k_},{k}']
                constrain_coeff = [1, -1, -(self.traveltime_k[k_ - 1][k - 1] - big_M), big_M]
                constrain_expr.append([constrain_var, constrain_coeff])
                rhs += [big_M + big_M]
        senses = 'G' * len(constrain_expr) if var_prefix == '' else 'GL' * int(len(constrain_expr) / 2)
        return constrain_expr, senses, rhs

    def create_const_num6(self, var_prefix=''):
        """
        Create the arguments needed for constrain number (6) in the corrected Bertsimas et al. model (Model 3).
        It assign the time to serve request k considering the time window cruising time when k is served first.
        :param var_prefix: string '' (empty) or '_' that specify if t decision variable is for a vehicle v
        (as in Model 3.1).
        :return: constrain_expr, senses, rhs
        """
        constrain_expr, rhs = [], []
        for v, k in self.arcs_v:
            veh = self.system.sys_vehs[v - 1]
            t_ind = var_prefix if var_prefix == '' else f'_{veh.v_id}'
            constrain_var = [f't{t_ind}_{k}', f'y_{veh.v_id},{k}']
            constrain_coeff = [1, -self.traveltime_v[veh.v_id - 1][k - 1]]
            constrain_expr.append([constrain_var, constrain_coeff])
            rhs.append(self.timewindow_k[k - 1][0])
            if var_prefix != '':
                # This is the online model
                big_M = self.system.sys_T + self.system.sys_tau_max  # TODO: DO WE NEED LARGER BIG_M AS IN CONSTRAIN (10)???
                u_ind = f'_{veh.v_id}'
                constrain_var = constrain_var + [f'u{u_ind}_{k}']
                constrain_coeff = [1, big_M - self.traveltime_v[veh.v_id - 1][k - 1], big_M]
                constrain_expr.append([constrain_var, constrain_coeff])
                rhs.append(self.timewindow_k[k - 1][0] + 2 * big_M)
        senses = 'GL' * len(self.arcs_v) if var_prefix != '' else 'G' * len(self.arcs_v)
        return constrain_expr, senses, rhs

    def create_const_abs(self):
        constrain_expr = []
        for i in range(self.group_n):
            for j in range(i + 1, self.group_n):
                constrain_var = [f'W_G{i},G{j}'] + \
                                [f'p_{k}' for k in self.request_groups[i]] + \
                                [f'p_{k}' for k in self.request_groups[j]]
                # Group j is greater
                constrain_coeff = [1] + \
                                  [(-1) * self.q_G[j]] * len(self.request_groups[i]) + \
                                  [self.q_G[i]] * len(self.request_groups[j])
                constrain_expr.append([constrain_var, constrain_coeff])
                # Group i is greater
                constrain_coeff = [1] + \
                                  [self.q_G[j]] * len(self.request_groups[i]) + \
                                  [(-1) * self.q_G[i]] * len(self.request_groups[j])
                constrain_expr.append([constrain_var, constrain_coeff])
        senses = 'G' * len(constrain_expr)
        rhs = [0] * len(constrain_expr)
        return constrain_expr, senses, rhs

    def add_constraints(self):
        # Constrain 1 (Bertsimas constrain 6):
        constrain_expr_1, senses_1, rhs_1 = self.create_const_num1()
        # Constrain 2 (Bertsimas constrain 7):
        constrain_expr_2, senses_2, rhs_2 = self.create_const_num2()
        # Constrain 3 (Bertsimas constrain 8):
        constrain_expr_3, senses_3, rhs_3 = self.create_const_num3()
        # Constrain 4 (Bertsimas constrain 12):
        constrain_expr_4, senses_4, rhs_4 = self.create_const_num4()
        # Constrain 5 (constrain 13):
        constrain_expr_5, senses_5, rhs_5 = self.create_const_num5()
        # Constrain 6 (constrain 14):
        constrain_expr_6, senses_6, rhs_6 = self.create_const_num6()
        # Absolute number constraints:
        constrain_expr_abs, senses_abs, rhs_abs = [], '', []
        if self.objective == 'Z':
            constrain_expr_abs, senses_abs, rhs_abs = self.create_const_abs()

        # Collect all the lists to add as a batch to the cplex model
        constrain_expr_all = constrain_expr_1 + constrain_expr_2 + constrain_expr_3 + constrain_expr_4 + \
                             constrain_expr_5 + constrain_expr_6 + constrain_expr_abs
        senses_all = senses_1 + senses_2 + senses_3 + senses_4 + senses_5 + senses_6 + senses_abs
        rhs_all = rhs_1 + rhs_2 + rhs_3 + rhs_4 + rhs_5 + rhs_6 + rhs_abs
        self.prob.linear_constraints.add(lin_expr=constrain_expr_all, senses=senses_all, rhs=rhs_all)

    def solve_model(self):
        if self.objective in ['Profit', 'Z']:
            self.prob.objective.set_sense(self.prob.objective.sense.maximize)
        else:
            assert self.objective in ['RequestsWaiting'], f'Different Objective Name :( {self.objective}'
            self.prob.objective.set_sense(self.prob.objective.sense.minimize)
        self.prob.solve()
        # Keep solution in the model attributes
        self.solution = Solution(self)

    def validate_sol(self):
        if self.objective == 'Z':
            if abs(self.solution.objective_value - self.solution.calc_z()) > 10 ** -8:
                print('MILP Z vs. Solution Z: ', abs(self.solution.objective_value - self.solution.calc_z()) < 10 ** -8)
                print(f'MILP Z = {self.solution.objective_value} | Solution Z = {self.solution.calc_z()}')

        elif self.objective == 'Profit':
            if abs(self.solution.objective_value - self.solution.calc_profit(self)) > 10 ** -8:
                print('MILP Profit vs. Solution Profit: ',
                      abs(self.solution.objective_value - self.solution.calc_profit(self)) < 10 ** -8)
                print(f'MILP Profit = {self.solution.objective_value} | '
                      f'Solution Profit = {self.solution.calc_profit(self)}')
        else:
            served_reqs = [k for k in range(self.n) if (k + 1) not in self.solution.reqs_not_served]
            t_k = self.solution.solution_full.loc[self.solution.solution_full.varNames.str.contains('t_')].varValues.values
            a_k = self.timewindow_k[:, 0]
            option1 = np.sum(t_k - a_k)
            option2 = np.sum(t_k[served_reqs] - a_k[served_reqs])
            if abs(self.solution.objective_value - option1) > 10 ** -8 or \
                abs(self.solution.objective_value - option2) > 10 ** -8:
                print(f'Different Waiting times! '
                      f'option1 diff: {abs(self.solution.objective_value - option1)} | '
                      f'option2 diff: {abs(self.solution.objective_value - option2)}')
                print(f'MILP = {self.solution.objective_value} | '
                      f'Option 1 = {option1} | Option 2 = {option2}')

        self.check_times()
        self.check_choose_reject()

    def check_times(self, mask_cond=''):
        results = []
        for v_id, route in self.solution.routes.items():
            mask_vehicle = f'{mask_cond}{v_id}' if mask_cond == '_' else mask_cond
            mask = self.solution.solution_active.varNames.isin([f't{mask_vehicle}_{k}' for k in route])
            t_k_df = self.solution.solution_active.loc[mask].sort_values(by='varValues')  # .varValues.values
            validation_route = run_simulation_route_validation(self.system, v_id, route, t_k_df, mask_cond)
            self.solution.update_validation(validation_route)

    def check_choose_reject(self, mask_cond=''):
        # find all the indices of the requests the solution have rejected
        reqs_not_served_ids = self.solution.reqs_not_served
        # check whether it could be fit in one of the routes
        for req_rej_id in reqs_not_served_ids:
            req_rej = self.system.sys_reqs[req_rej_id - 1]
            for v_id, route in self.solution.routes.items():
                # find the closest int that is smaller then the req id
                req_prev_i = bisect.bisect_left(route, req_rej_id)
                if req_prev_i == 0:
                    # if the rejected request could be served as first in route
                    veh = self.system.sys_vehs[v_id - 1]
                    serve_time = max(req_rej.get_arrival_time(), 0) + \
                                 self.system.get_travel_time(veh.get_zone(), req_rej.get_od_zones()[0])
                    if serve_time <= req_rej.get_expiration_time():
                        # if True, update the validation solution dictionary
                        examples = self.solution.validation.get('rej_choose', {})
                        preceding_lst = examples.get(req_rej_id, [])
                        examples[req_rej_id] = preceding_lst + [f'y_{v_id},{req_rej_id}']
                        self.solution.validation['rej_choose'] = examples
                    continue

                req_prev_id = route[req_prev_i - 1]
                req_prev = self.system.sys_reqs[req_prev_id - 1]
                # calc the optional service time with the earliest t_k (from validation or t_k in solution)
                mask_vehicle = f'{mask_cond}{v_id}' if mask_cond == '_' else mask_cond
                sol_pickup = float(self.solution.solution_active.loc[self.solution.solution_active.varNames ==
                                                                     f't{mask_vehicle}_{req_prev_id}'].varValues)
                earliest_pickup = self.solution.validation['t_k'].get(req_prev_id, sol_pickup)
                earliest_pickup_prev = earliest_pickup[0] if isinstance(earliest_pickup, tuple) else earliest_pickup
                trip_time_prev = self.system.get_travel_time(*req_prev.get_od_zones())
                cruising_time = self.system.get_travel_time(req_prev.get_od_zones()[1], req_rej.get_od_zones()[0])
                serve_time = max(earliest_pickup_prev + trip_time_prev, req_rej.get_arrival_time()) + cruising_time
                # check whether it is possible to serve it
                if serve_time <= req_rej.get_expiration_time():
                    # if True, update the validation solution dictionary
                    examples = self.solution.validation.get('rej_choose', {})
                    preceding_lst = examples.get(req_rej_id, [])
                    examples[req_rej_id] = preceding_lst + [(req_prev_id, req_rej.get_expiration_time() - serve_time)]
                    self.solution.validation['rej_choose'] = examples
        if 'rej_choose' not in self.solution.validation:
            self.solution.validation['rej_choose'] = False
