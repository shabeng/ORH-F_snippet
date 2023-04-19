class Vehicle:
    def __init__(self, vehicle_id, vehicle_time, vehicle_zone,
                 served_requests_list=None, total_cruise_time=0, total_passenger_time=0, total_gain=0, **kwargs):
        self.v_id = vehicle_id
        self.v_time = vehicle_time
        self.v_zone = vehicle_zone
        self.v_tot_cruise_time = total_cruise_time
        self.v_tot_passenger_time = total_passenger_time
        self.v_tot_gain = total_gain
        self.v_tot_wait_time = kwargs['sys'].sys_T
        if served_requests_list is not None:
            self.v_served_reqs = served_requests_list
        else:
            self.v_served_reqs = []

    def __repr__(self):
        return f'Vehicle(v: {self.v_id}, t_v: {self.v_time}, z_v: {self.v_zone})'

    def __eq__(self, other):
        return self.get_id() == other.get_id() and self.get_state() == other.get_state()

    def get_id(self):
        return self.v_id

    def get_state(self):
        return self.v_time, self.v_zone

    def get_zone(self):
        return self.v_zone

    def get_time(self):
        return self.v_time

    def get_waiting_time(self):
        return self.v_tot_wait_time

    def get_all_attr(self):
        return self.v_id, self.v_time, self.v_zone, \
               self.v_served_reqs[:], self.v_tot_cruise_time, self.v_tot_passenger_time, self.v_tot_gain

    def check_instance(self, obj_name):
        if obj_name == 'Vehicle':
            return True
        return False

    def update_match(self, is_warmup_pass, new_time, new_zone, cruise_time_increment=0,
                     passenger_time_increment=0, waiting_time_increment=0, payment=0, req_obj=None):
        self.v_time = new_time
        self.v_zone = new_zone
        if is_warmup_pass:
            self.v_tot_cruise_time += cruise_time_increment
            self.v_tot_passenger_time += passenger_time_increment
            self.v_tot_wait_time += waiting_time_increment
            self.v_tot_gain += payment
            self.v_served_reqs.append(req_obj)

    def write_row_one_veh(self):
        return self.v_tot_cruise_time, self.v_tot_wait_time, self.v_tot_gain


if __name__ == '__main__':
    #     # system = {'T':20}
    # vec_1 = Vehicle(1,20,12, sys=system)
    # vec_2 = Vehicle(1,20,12, sys=system)
    # print(vec_1 == vec_2)
    # print(vec_1 != vec_2)
    pass
