# from collections import Counter
import numpy as np
import pandas as pd
import pickle
from os import listdir
from os.path import isfile, join
from collections import defaultdict

from script.ProblemObjects import vehicles as vc
from script.ProblemObjects import requests as rq
from script.utilis.utilis import calc_r_q_G, group_definition_given_reqs_set
from script.utilis.utilis_simulation import calc_prob_neigh_est


class System:
    def __init__(self, seed_num, num_time_points, num_vehicles, travel_time_mat, reqs_arr_prob, reqs_od_probs,
                 payment_func, G_script, city_center_zones, vehs_center_only=True, warm_up_reqs=0,
                 expiration_method='fixed', **kwargs):
        """
        create new instance of the system that can serve several experiments with the same requests set and vehicles
        initials state.
        :param seed_num: to initiate the seed for the creation of the vehicle and request sets
        :param num_time_points: int, T
        :param num_vehicles: int, |V|
        :param travel_time_mat: numpy array |Z|*|Z|, travel time between zones
        :param reqs_arr_prob: 0 < float < 1, probability that request arrive at t
        :param reqs_od_probs: numpy array |Z|*|Z|, probability of request from o to d
        :param payment_func: lambda func, get a request as input and returns the payment of request
        :param G_script: dictionary, mapping from request geo attributes to the group request;
                         key: tuple, group criteria  value: int, group index G
        :param warm_up_reqs: int, the number of requests to ignore in the measure calculations.
        :param expiration_method: str, fixed, zone_dependent or fixed_orig
        :param city_center_zones: list of int, the indices of the city center zones
        :param kwargs: if 'fixed' -> fixed_exp_c: int, the fixed duration till expiration from city center reqs.
                                     fixed_exp_s: int, the fixed duration till expiration from outskirts center reqs.
                       if 'zone_dependent' -> exp_func: function of orig and dest to calculate the duration to sample
                       till expiration.
        """
        self.sys_T = num_time_points
        self.sys_V = num_vehicles
        self.sys_zones_size = travel_time_mat.shape[0]
        self.sys_center_inx = city_center_zones
        self.sys_t_i_j = travel_time_mat
        self.sys_reqs_arr_prob = reqs_arr_prob
        self.sys_reqs_od_probs = reqs_od_probs
        self.sys_payment_func = payment_func
        self.sys_exp_method = expiration_method
        self.sys_tau_max = max(kwargs['fixed_exp_c'], kwargs['fixed_exp_s']) if self.sys_exp_method == 'fixed' \
            else self.sys_T
        self.sys_seed = np.random.RandomState(seed_num)
        self.sys_reqs = self.generate_request_set(**kwargs)
        self.warmup = warm_up_reqs
        self.sys_reqs_size = len(self.sys_reqs) - self.warmup
        self.sys_vehs = self.generate_vehicle_set(vehs_center_only)
        self.sys_G_script = G_script
        self.update_fairness_groups()
        self.sys_r_G, self.sys_q_G = calc_r_q_G(self)

    def generate_request_set(self, **kwargs):
        """
        create a set of request to assign it to the system
        :param exp_method: str, fixed or zone dependent
        :return: list, requests objects
        """
        k = 0
        reqs_lst = []
        for t in range(1, self.sys_T + 1):
            p = self.sys_seed.uniform()
            if p <= self.sys_reqs_arr_prob:
                k += 1
                o, d, e = self.generate_req_attrs(**kwargs)
                req = rq.Request(k, o, d, t, t+e)  # k, origin_zone, destination_zone, arrival_time, expiration_time
                reqs_lst.append(req)
            else:
                continue
        return reqs_lst

    def generate_req_attrs(self, **kwargs):
        orig, dest = self.generate_origin_destination()
        expr = self.generate_expiration_dur(orig, dest, **kwargs)
        return orig, dest, expr

    def generate_origin_destination(self):
        lst_inx = self.sys_seed.choice(np.arange(self.sys_reqs_od_probs.size), p=self.sys_reqs_od_probs.ravel())
        arr_inx = list(np.ndindex(self.sys_reqs_od_probs.shape))[lst_inx]
        return arr_inx

    def generate_expiration_dur(self, orig, dest, **kwargs):
        if self.sys_exp_method == 'fixed':
            dur = kwargs['fixed_exp_c'] if orig in self.sys_center_inx else kwargs['fixed_exp_s']

        elif self.sys_exp_method == 'zone_dependent':
            # min_dur, max_dur = self.sys_t_i_j[kwargs['orig'], kwargs['orig']], 3 * self.sys_t_i_j[orig, dest]
            dur = kwargs['exp_func'](orig, dest, self.sys_seed)

        elif self.sys_exp_method == 'fixed_orig':
            dur = round(kwargs['fixed_exp_c'] * 1.5) if orig not in self.sys_center_inx else kwargs['fixed_exp_c']

        return dur

    def generate_vehicle_set(self, vehs_center_only):
        vehs_lst = []
        for v in range(1, self.sys_V+1):
            if vehs_center_only:
                zones_prob = self.sys_reqs_od_probs.sum(axis=1)[self.sys_center_inx]
                zones_prob = zones_prob / zones_prob.sum()
                z_v = self.sys_seed.choice(self.sys_center_inx, p=zones_prob)
            else:
                # zones_prob = self.sys_reqs_od_probs.sum(axis=1)
                zones_prob = np.ones(self.sys_zones_size)
                zones_prob[self.sys_center_inx] = 2
                zones_prob = zones_prob / zones_prob.sum()
                z_v = self.sys_seed.choice(self.sys_zones_size, p=zones_prob)
            t_v = 0
            veh = vc.Vehicle(v, t_v, z_v, sys=self)  # vehicle_id, vehicle_time, vehicle_zone
            vehs_lst.append(veh)
        return vehs_lst

    def calc_payment(self, req):
        """calc the payment of the request according to its travel time
        :param req: request object
        :return: float, what the passenger pays
        """
        pay = self.sys_payment_func(req, self.sys_t_i_j)
        return pay

    def get_travel_time(self, orig_zone, dest_zone):
        return self.sys_t_i_j[orig_zone, dest_zone]

    def find_vehicles_k(self, req, vehs_set=None):
        vehicles_k = []
        V = self.sys_vehs if vehs_set is None else vehs_set
        for v in V:
            if max(req.k_arrival_time, v.v_time) + self.sys_t_i_j[v.v_zone][req.k_orig] <= req.k_expiration_time:
                vehicles_k.append(v)
        return vehicles_k

    def find_profitable_vehicles_k(self, req, vehs_set, min_profit):
        vehicles_k_p = []
        for v_k in vehs_set:
            z_v = v_k.get_zone()
            if z_v in self.sys_center_inx:
                # Only vehicles in the city center rejects upon profit
                o_k, d_k = req.get_od_zones()
                passenger_time = self.get_travel_time(o_k, d_k)
                cruise_time = self.get_travel_time(z_v, o_k)
                # Calc the profit per time unit for this vehicle serving req
                profit = self.calc_payment(req) - (cruise_time + passenger_time)
                if np.any((profit / (cruise_time + passenger_time)) >= min_profit):
                    vehicles_k_p.append(v_k)
            else:
                # Vehicles in city outskirts always accept serving req if available
                vehicles_k_p.append(v_k)
        return vehicles_k_p

    def update_fairness_groups(self):
        """
        Add to the G_script dictionary the entries that are dependent with the sampled requests set, namely it includes:
        (1) group_num - from all the possible neighborhood couples, there is a group only if there at least one request
        with the matching attributes, omitting groups with zero requests (r_Gi == 0).
        (2) group_name - mapping from each group name to a corresponding integer that is the index of this group in the
        vectors (r_G or x_G vectors for example).
        :return: None, change the dictionary of the sys_G_script class attribute
        """
        # (1) Add the number of requests groups
        # groups = defaultdict(list)
        # for req in self.sys_reqs:
        #     o, d = req.get_od_zones()
        #     group_o = self.find_geo_attr('group', zone_id=o)
        #     group_d = self.find_geo_attr('group', zone_id=d)
        #     groups[f'{group_o},{group_d}'].append(req)
        groups = group_definition_given_reqs_set(self)
        self.sys_G_script['group_num'] = len(groups)

        # (2) Add the mapping from group name to its index
        increment_ind = 0
        for group in groups:
            self.sys_G_script[group] = increment_ind
            increment_ind += 1

        # (3) Calc the probability of request from arriving from each neighborhood
        # (this is because it is costly and constant for the system)
        self.sys_G_script['neigh_prob'] = calc_prob_neigh_est(self)

        # (4) Insert mapping of each zone to its nearby areas
        self.update_nearby_areas()

    def find_geo_attr(self, attr_key, zone_id=None):
        """
        Returns the value of the mention attr_key that the request objects has
        :param attr_key: str, the name of the attributes in the dictionary of the zone.
        :param zone_id: int, the id number of the zone, starting with 1
        :return: int
        """
        if zone_id is not None:
            attr_val = self.sys_G_script[zone_id][attr_key]
        else:
            attr_val = self.sys_G_script[attr_key]
        return attr_val

    def update_nearby_areas(self):
        for ok in range(self.sys_zones_size):
            set_ok = set()
            time_to_ok = self.sys_t_i_j[:, ok]
            near_ok = np.where(time_to_ok < self.sys_tau_max)[0]
            for z in near_ok:
                area = self.find_geo_attr('neigh', zone_id=z)
                set_ok.add(area)
            self.sys_G_script[ok]['nearby_areas'] = sorted(set_ok)

    def update_match_measures(self, veh, req):
        t_v, z_v = veh.get_state()
        o_k, d_k = req.get_od_zones()
        t_k = req.get_arrival_time()
        cruise_time = self.get_travel_time(z_v, o_k)
        passenger_time = self.get_travel_time(o_k, d_k)
        payment_k = self.calc_payment(req)
        z_v_new = d_k
        t_v_new = max(t_v, t_k) + cruise_time + passenger_time
        wait_time_veh = max(t_k - t_v, 0) + max(self.sys_T - t_v_new, 0) - \
                        max(self.sys_T - t_v, 0)
        wait_time_req = t_v_new - (passenger_time + t_k)
        profit = payment_k - (cruise_time + passenger_time)
        return (t_v_new, z_v_new, cruise_time, passenger_time, wait_time_veh, profit), \
               (wait_time_req, cruise_time)

    def calc_total_trip_time(self):
        cum_sum = 0
        for req in self.sys_reqs[self.warmup:]:
            cum_sum += self.get_travel_time(*req.get_od_zones())
        return cum_sum

    def calc_demand_supply_ratio(self):
        total_trip_time = self.calc_total_trip_time()
        supply_time = self.sys_V * (self.sys_T + self.sys_tau_max + self.sys_t_i_j.max())
        return total_trip_time / supply_time

    def calc_instance_measures(self):
        total_trip_time = self.calc_total_trip_time()
        ratio = self.calc_demand_supply_ratio()
        # |V|, T, p, |K|, tau, total reqs trip times, supply demand ratio
        return self.sys_V, self.sys_T, self.sys_reqs_arr_prob, self.sys_reqs_size + self.warmup, self.sys_tau_max, \
               total_trip_time, ratio


class SystemData(System):
    def __init__(self, seed_num, payment_func, warm_up_reqs=0, expiration_method='fixed', vehs_center_only=True,
                 path='/Users/shabeng/PycharmProjects/ORH-F/Data/sampleBubble_K60_V4', **kwargs):
        """create new instance of the system that can serve several experiments with the same requests set and vehicles
        initials state.
        :param seed_num: to initiate the seed for the creation of the vehicle and request sets
        :›param payment_func: lambda func, get a request as input and returns the payment of request
        :param warm_up_reqs: int, the number of requests to ignore in the measure calculations.
        :param expiration_method: str, fixed, zone_dependent or fixed_orig
        :param city_center_zones: list of int, the indices of the city center zones
        :param kwargs: if 'fixed' -> fixed_exp_c: int, the fixed duration till expiration from city center reqs.
                                     fixed_exp_s: int, the fixed duration till expiration from outskirts center reqs.
                       if 'zone_dependent' -> exp_func: function of orig and dest to calculate the duration to sample
                       till expiration."""
        self.sys_T = 0
        self.sys_V = 0
        self.sys_zones_size = 0
        self.sys_center_inx = []
        self.sys_t_i_j = np.zeros((1, 1))
        self.sys_reqs_arr_prob = np.zeros((1, 1))
        self.sys_reqs_od_probs = np.zeros((1, 1))
        self.sys_payment_func = payment_func
        self.sys_exp_method = expiration_method
        self.sys_tau_max = max(kwargs['fixed_exp_c'], kwargs['fixed_exp_s']) if self.sys_exp_method == 'fixed' \
            else self.sys_T
        self.sys_seed = np.random.RandomState(seed_num)
        self.sys_reqs = []
        self.warmup = warm_up_reqs
        self.sys_reqs_size = 0
        self.sys_vehs = []
        self.sys_G_script = {}
        self.sys_r_G, self.sys_q_G = np.zeros((1, 1)), np.zeros((1, 1))
        self.data_path = path
        # self.create_system_from_files(**kwargs)

    def create_reqs_from_file(self, file_name_orders, file_name_groups, **kwargs):
        with open(f'{self.data_path}/{file_name_groups}', 'rb') as handle:
            suberb_inx, self.sys_center_inx, self.sys_G_script = pickle.load(handle)
        self.sys_zones_size = self.sys_G_script['zone_num']  # TODO: change the dictionary in Jupyter exploration files.

        df = pd.read_csv(f'{self.data_path}/{file_name_orders}',
                         header=None,
                         names=['ordReqDate', 'ordReqTime', 'ordPUDate', 'ordPUTime', 'ordDODate', 'ordDOTime',
                                'ordOrigin', 'ordDestination'])
        df['ordPUTime'] = pd.to_datetime(df.ordPUTime)
        if self.sys_exp_method == 'fixed':
            first_pu = df.ordPUTime.iloc[0]  # df.ordPUTime.min()
            start_time = \
                first_pu - pd.Timedelta(f'00:{self.sys_tau_max if self.sys_tau_max >= 10 else "0%d" % self.sys_tau_max}:00')
            df['ordExpirationTime'] = (df['ordPUTime'] - start_time).astype('timedelta64[m]') + 1
            df['ordArriveTime'] = df.apply(
                lambda line: line.ordExpirationTime - kwargs['fixed_exp_c'] if line.ordOrigin in self.sys_center_inx
                else line.ordExpirationTime - kwargs['fixed_exp_s'], axis=1)

        elif self.sys_exp_method == 'zone_dependent':
            start_time = df['ordReqTime'].iloc[0]
            df['ordExpirationTime'] = (df['ordPUTime'] - start_time).astype('timedelta64[m]')
            df['ordArriveTime'] = (df['ordReqTime'] - start_time).astype('timedelta64[m]')

        self.sys_reqs_size = df.shape[0] - self.warmup
        self.sys_T = df['ordArriveTime'].iloc[-1]  # df['ordArriveTime'].max()

        for index, row in df.iterrows():
            k = index + 1
            o = row.ordOrigin
            d = row.ordDestination
            t = row.ordArriveTime
            e = row.ordExpirationTime
            req = rq.Request(k, o, d, t, e)  # k, origin_zone, destination_zone, arrival_time, expiration_time
            self.sys_reqs.append(req)
        self.sys_reqs_arr_prob = self.sys_reqs_size / self.sys_T

        self.update_fairness_groups()
        self.sys_r_G, self.sys_q_G = calc_r_q_G(self)
        return

    def create_vehs_from_file(self, file_name_vehs):
        df = pd.read_csv(f'{self.data_path}/{file_name_vehs}')
        self.sys_V = df.shape[0]
        for index, row in df.iterrows():
            z_v = self.sys_seed.choice(self.sys_center_inx)
            t_v = 0
            veh = vc.Vehicle(index + 1, t_v, z_v, sys=self)  # vehicle_id, vehicle_time, vehicle_zone
            self.sys_vehs.append(veh)
        return

    def generate_vehicle_set(self, vehs_center_only, **kwargs):
        vehs_lst = []
        self.sys_V = kwargs['V']
        zones_to_sample = self.sys_center_inx if vehs_center_only else range(self.sys_t_i_j.shape[0])
        for v in range(1, self.sys_V+1):
            z_v = self.sys_seed.choice(zones_to_sample)
            t_v = 0
            veh = vc.Vehicle(v, t_v, z_v, sys=self)  # vehicle_id, vehicle_time, vehicle_zone
            vehs_lst.append(veh)
        return vehs_lst

    def create_system_from_files(self, **kwargs):
        files = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f))]  # Read the folder in the path
        file_name_orders = [name for name in [file for file in files if 'orders' in file] if 'group' in name][0]
        file_name_groups = [file for file in files if 'GScript' in file][0]
        file_name_vehs = [file for file in files if 'cars' in file][0]
        file_name_time = [file for file in files if 'drive_dur' in file][0]

        exp_dur = self.sys_tau_max if self.sys_exp_method == 'fixed' else 'Need to solve this!'

        self.create_reqs_from_file(file_name_orders, file_name_groups, exp_dur)
        self.create_vehs_from_file(file_name_vehs)
        self.sys_t_i_j = \
            pd.read_csv(f'{self.data_path}/{file_name_time}').values[:self.sys_zones_size, :self.sys_zones_size]
        self.sys_t_i_j = (self.sys_t_i_j / 60).round()
        return

    # def func_to_edit(self):
    #     cs_arr = np.zeros(4)
    #     mapping = {'c2c': 0, 'c2s': 1, 's2c': 2, 's2s': 3}
    #     for index, row in df.iterrows():
    #         if row.ordOrigin in self.sys_center_inx:
    #             if row.ordDestination in self.sys_center_inx:
    #                 cs_arr[0] += 1
    #             else:
    #                 cs_arr[1] += 1
    #         else:
    #             if row.ordDestination in self.sys_center_inx:
    #                 cs_arr[2] += 1
    #             else:
    #                 cs_arr[3] += 1
    #
    #     od_prob = np.zeros((self.sys_G_script['zone_num'], self.sys_G_script['zone_num']))
    #     cs_prob = cs_arr / 60
    #     od_prob[self.sys_center_inx, self.sys_center_inx] = cs_prob[0] / (
    #                 len(self.sys_center_inx) * len(self.sys_center_inx))
    #     od_prob[self.sys_center_inx, suberb_inx] = cs_prob[1] / (len(self.sys_center_inx) * len(suberb_inx))


if __name__ == '__main__':
    # Params:
    seed = 44
    # T = 100
    # V = 50
    # t_ij = np.array([[1, 4, 5, 13, 10, 8],
    #                  [4, 1, 3, 13, 14, 12],
    #                  [5, 3, 1, 10, 12, 11],
    #                  [13, 13, 10, 1, 20, 18],
    #                  [10, 14, 12, 20, 1, 3],
    #                  [8, 12, 11, 18, 3, 1]])
    # t_ij = np.floor((t_ij + 1) / 2)
    # reqs_arr_p = 0.2
    # reqs_od_p_mat = np.array([[0.02222222, 0.02222222, 0.02222222, 0.03888889, 0.03888889, 0.03888889],
    #                           [0.02222222, 0.02222222, 0.02222222, 0.03888889, 0.03888889, 0.03888889],
    #                           [0.02222222, 0.02222222, 0.02222222, 0.03888889, 0.03888889, 0.03888889],
    #                           [0.03888889, 0.03888889, 0.03888889, 0.01111111, 0.01111111, 0.01111111],
    #                           [0.03888889, 0.03888889, 0.03888889, 0.01111111, 0.01111111, 0.01111111],
    #                           [0.03888889, 0.03888889, 0.03888889, 0.01111111, 0.01111111, 0.01111111]])
    # city_center_inxs = range(3)
    # Payment Params:
    A = np.max(19)
    a = 5
    t_min = A / a
    pay_func = lambda r, travel: A if travel[r.k_orig, r.k_dest] <= t_min \
                                    else A + a*(travel[r.k_orig, r.k_dest] - t_min)

    # G script - set of request groups
    # requests_group = {** {(i, j): 0 for i in range(3) for j in range(3)},
    #                   ** {(i, j): 1 for i in range(3) for j in range(3, 6)},
    #                   ** {(i, j): 2 for i in range(3, 6) for j in range(3)},
    #                   ** {(i, j): 3 for i in range(3, 6) for j in range(3, 6)}}
    # System instance
    sys1 = SystemData(seed, pay_func, fixed_exp_c=20, fixed_exp_s=20,
                      path='/Users/shabeng/PycharmProjects/ORH-F/Data/sampleBubble2/3pm/v01')
    sys1.create_reqs_from_file(file_name_orders='bubble_3pm_to_4pm_v01_orders.csv',
                               file_name_groups='processed/sZones_cZones_GScript.pickle',
                               fixed_exp_c=20, fixed_exp_s=20)
    # sys2 = System(seed, T, V, t_ij, reqs_arr_p, reqs_od_p_mat, pay_func, requests_group, city_center_inxs,
    #               fixed_exp_c=1, fixed_exp_s=100)

    # print(sys1.sys_reqs == sys2.sys_reqs)
    # print(sys1.sys_vehs == sys2.sys_vehs)

    # req1 = sys1.sys_reqs[0]
    # veh = sys1.sys_vehs[0]
    # cruise = sys1.get_travel_time(veh.get_zone(), req1.get_od_zones()[0])
    # passenger = sys1.get_travel_time(*req1.get_od_zones())
    # new_t = max(req1.get_arrival_time(), veh.get_time()) + cruise + passenger
    # new_z = req1.get_od_zones()[1]
    # pay = sys1.calc_payment(req1)
    # wait = max(req1.k_arrival_time-veh.v_time, 0) + max(sys1.sys_T-new_t, 0) - (sys1.sys_T-veh.v_time)
    # veh.update_match(req1, new_t, new_z, cruise, passenger, wait, pay)
    # # print('hello')
