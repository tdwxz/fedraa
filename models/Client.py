from sortedcontainers import SortedList


class Client:
    def __init__(self, client_id=0):
        self.current_time = 0
        self.model_idx = 0
        self.client_id = client_id
        self.region_parament = None

    def __lt__(self, other):
        if isinstance(other, Client):
            return self.current_time > other.current_time
        return NotImplemented

    def __str__(self):
        return "client_id: {}, current_time: {}".format(self.client_id, self.current_time)


class Clients:
    def __init__(self, client_num: int):
        self.relax_list = self.create_client(client_num)
        self.waiting_list = dict()
        self.update_list = SortedList()

    def create_client(self, client_num) -> SortedList:
        relax_list = SortedList()
        for i in range(client_num):
            relax_list.add(Client(i))
        return relax_list

    def set_client_initial_time(self, server_time):
        l = []
        while len(self.relax_list) > 0:
            client = self.relax_list.pop()
            client.current_time = server_time
            l.append(client)
        for client in l:
            self.relax_list.add(client)

