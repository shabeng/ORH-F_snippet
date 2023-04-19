import numpy as np


class Request:
    def __init__(self, k, origin_zone, destination_zone, arrival_time, expiration_time):
        """
        create new instance of request
        :param k: int, request index
        :param origin_zone: int, zone index of the origin
        :param destination_zone: int, zone index of the destination
        :param arrival_time: int, time point index
        :param expiration_time: int, time point index
        """
        self.k_id = k
        self.k_orig = origin_zone
        self.k_dest = destination_zone
        self.k_arrival_time = arrival_time
        self.k_expiration_time = expiration_time
        self.k_served = False
        self.k_vehicle = None
        self.k_waiting_time = np.inf
        self.k_is_busy_vehicle = None
        self.k_cruise_time = np.inf

    def __repr__(self):
        """print a request
        """
        return f'Request(k: {self.k_id}, origin: {self.k_orig}, destination: {self.k_dest}, ' \
               f'arrival_t: {self.k_arrival_time}, ' \
               f'expiration_t: {self.k_expiration_time}, served?: {self.k_served})'

    def __eq__(self, other):
        return self.k_orig == other.k_orig and self.k_dest == other.k_dest and \
               self.k_arrival_time == other.k_arrival_time and self.k_expiration_time == other.k_expiration_time

    def get_id(self):
        return self.k_id

    def get_od_zones(self):
        return self.k_orig, self.k_dest

    def get_arrival_time(self):
        return self.k_arrival_time

    def get_expiration_time(self):
        return self.k_expiration_time

    def get_is_busy_vehicle(self):
        return self.k_is_busy_vehicle

    def get_waiting_time(self):
        if self.k_served:
            return self.k_waiting_time
        else:
            return None

    def get_is_served(self):
        return self.k_served

    def check_instance(self, obj_name):
        if obj_name == 'Request':
            return True
        return False

    def update_match(self, is_warmup_pass, v_id, waiting_time, cruise_time):
        """
        update data after an assignment of request to a vehicle
        :param v_id: int, vehicle index
        :param waiting_time: float, time till pickup from arrival (t_pickup - t_k)
        :param cruise_time:  float, time till pickup from serving starts (t_pickup - t_v)
        """
        self.k_is_busy_vehicle = waiting_time != cruise_time
        if is_warmup_pass:
            self.k_served = True
            self.k_vehicle = v_id
            self.k_waiting_time = waiting_time
            self.k_cruise_time = cruise_time

    def find_request_type(self, system_obj):
        """
        returns the identifier of the group G the request belongs to
        :param system_obj: dict, mapping from zone to geo attributes and strings to the group request;
                key: zone num  value: dict, which maps key words (strings) to the values.
        :return: identifier of the request group G
        """
        o, d = self.get_od_zones()
        o_cs = system_obj.find_geo_attr('group', zone_id=o)
        d_cs = system_obj.find_geo_attr('group', zone_id=d)
        group_type = system_obj.find_geo_attr('group_type')
        if group_type == 'couple_od':
            group_str = f'{o_cs},{d_cs}'
        elif group_type == 'single_o':
            group_str = f'{o_cs}'
        else:
            group_str = f'{d_cs}'
        request_group_ind = system_obj.find_geo_attr(group_str)
        return request_group_ind

    def calc_payment(self, func):
        """calc the payment of the request according to its travel time
        :param func: lambda function with two inputs that can be changed in accordance to the experiments
        :return: float, the payment the passenger pay for the ride
        """
        return func(self)

    def write_row_one_req(self):
        return self.k_waiting_time, self.k_cruise_time


if __name__ == '__main__':
    dic = {(20, 3): 2}
    r1 = Request(1, 20, 3, 15, 25)
    r2 = Request(2, 5, 4, 25, 30)
    print(r1)
    r1.update_match(15, 20, 5)
    print(r1)
    # print(r1.find_request_type(dic))
    print(r1.calc_payment(lambda req: req.k_orig * req.k_dest))
