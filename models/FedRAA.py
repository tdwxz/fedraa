import copy
import random
import time

from models.Client import Clients, Client
from models.Update import LocalUpdate
from models.test import test_img
from utils.tool import MyLog, save_file_to_json
import torch
from torch import nn
import numpy as np
from models.Data import Data


def gen_client_cmp(client_num: int, heterogeneous_strategy: dict):
    client_cmp = []
    val_sum = 0
    max_key = -1
    client_sum = 0
    for key in heterogeneous_strategy.keys():
        max_key = max(max_key, key)
        val_sum += heterogeneous_strategy[key]
        heterogeneous_strategy[key] = int(heterogeneous_strategy[key] * client_num)
        client_sum += heterogeneous_strategy[key]

    heterogeneous_strategy[key] = client_num - (client_sum - heterogeneous_strategy[key])

    for key, val in heterogeneous_strategy.items():
        for i in range(val):
            client_cmp.append(key)
    return client_cmp


def get_sub_paras(w_glob, wmask: dict, target_neuron_idx):
    for name in wmask.keys():
        w_glob[name] *= wmask[name][target_neuron_idx]
    return w_glob


def get_ranks(net: nn.Module, layer_name: str):
    return torch.argsort(net.state_dict()[layer_name].view(-1))


def get_local_wmasks(net: nn.Module, range_list: list, mat_size: dict, device):
    local_masks_dict, opp_local_masks_dict = {}, {}
    for name in mat_size.keys():
        ranks = get_ranks(net, name)
        layer_mat_size = mat_size[name]
        local_masks = []
        opp_local_masks = []
        percent_range = [0]
        mask0 = torch.zeros_like(ranks)
        mask1 = torch.ones_like(ranks)
        if sum(range_list) != 100:
            raise Exception("range_list总和不为100")
        for i in range(len(range_list)):
            percent_range.append(range_list[i] + percent_range[-1])
        print("wmask percent_range: {}".format(percent_range))
        neuron_num = len(ranks)
        for idx in range(1, len(percent_range)):
            left = percent_range[idx - 1] / 100
            right = percent_range[idx] / 100
            x = copy.deepcopy(ranks)
            mask = torch.where(torch.logical_and(x >= neuron_num * left, x < neuron_num * right), mask1, mask0)
            local_masks.append(mask.view(layer_mat_size).to(device))  # 200 * 784
            mask = torch.where(torch.logical_and(x >= neuron_num * left, x < neuron_num * right), mask0, mask1)
            opp_local_masks.append(mask.view(layer_mat_size).to(device))
        local_masks_dict[name] = local_masks
        opp_local_masks_dict[name] = opp_local_masks
    return local_masks_dict, opp_local_masks_dict


class UpdateModel:
    def __init__(self):
        self.w = None
        self.loss = None
        self.new_update = None
        self.update_time = 0
        self.target_neuron_idx = 0
        self.train_client_id = 0
        self.start_train_time = 0

    def __lt__(self, other):
        if isinstance(other, UpdateModel):
            return self.update_time > other.update_time
        return NotImplemented


def get_staleness(update: UpdateModel):
    return 1 / (update.update_time / 100 - update.start_train_time / 100 + 1)


class FedRAA:
    def __init__(self,
                 client_num: int,
                 net_glob: nn.Module,
                 heterogeneous_strategy: dict,
                 part_array: list,
                 K: int,
                 args,
                 device,
                 data: Data,
                 mat_size,
                 open_client_local_test=False,
                 ):
        self.data = data
        self.client_num = client_num
        self.clients = Clients(client_num)
        self.net_glob = net_glob
        self.heterogeneous_strategy = heterogeneous_strategy
        self.client_cmp = gen_client_cmp(self.client_num, self.heterogeneous_strategy)
        self.part_array = part_array
        self.part_array_num = len(self.part_array)
        self.server_time = 0
        self.K = K
        self.args = args
        self.device = device
        self.open_client_local_test = open_client_local_test
        self.mat_size = mat_size
        self.wmasks_dict, self.opp_wmasks_dict = get_local_wmasks(net_glob, part_array, mat_size, device)

    def get_wranks(self):
        state = copy.deepcopy(self.net_glob.state_dict())
        return torch.argsort(state['layer_hidden.weight'].view(-1))

    def get_cmp_array(self):
        client_train_time = np.zeros((self.client_num, self.part_array_num))
        for i in range(self.client_num):
            for j in range(self.part_array_num):
                client_train_time[i, j] = (self.part_array[j] / 100) / self.client_cmp[i]
        return client_train_time

    def train_init(self):
        self.net_glob.train()
        self.net_glob.to(self.device)

    def train(self, epochs: int, log, data_file, select_submodel_type=0):
        if select_submodel_type == 4:
            train_info = self.train_avg(epochs, log, data_file)
            return train_info

        self.train_init()
        self.clients.set_client_initial_time(self.server_time)
        client_train_time = self.get_cmp_array()
        partition_percentage_use_num = np.zeros(self.part_array_num)
        train_info = {
            'client_train_num': [0 for _ in range(self.client_num)],
            'neural_train_num': [0 for _ in range(self.part_array_num)]
        }
        dic = {
            'client_num': self.client_num,
            'part_array': self.part_array,
            'cilent_cmp': self.client_cmp,
            'K': self.K,
        }
        save_file_to_json(dic, data_file)
        if self.open_client_local_test:
            train_info['client_train_acc'] = [[] for _ in range(self.client_num)]
            train_info['client_train_loss'] = [[] for _ in range(self.client_num)]
        neuron_allocated_time = [0 for _ in range(self.part_array_num)]
        for epoch in range(epochs):
            while self.clients.relax_list[-1].current_time <= self.server_time:
                client: Client = self.clients.relax_list.pop()

                train_info['client_train_num'][client.client_id] += 1

                client.current_time = self.server_time
                target_neuron_idx = self.get_target_neuron_idx(client.client_id,
                                                               client_train_time,
                                                               partition_percentage_use_num,
                                                               select_submodel_type,
                                                               self.server_time,
                                                               neuron_allocated_time)

                train_info['neural_train_num'][target_neuron_idx] += 1

                partition_percentage_use_num[target_neuron_idx] += 1
                client.model_idx = target_neuron_idx
                part_neural_region: nn.Module = copy.deepcopy(self.net_glob)
                mask_param = get_sub_paras(part_neural_region.state_dict(), self.wmasks_dict, target_neuron_idx)
                part_neural_region.load_state_dict(mask_param)
                client.region_parament = part_neural_region

                local = LocalUpdate(self.args, dataset=self.data.train_datas[client.client_id])
                start_train_time = time.time()
                w, loss = local.train(net=copy.deepcopy(part_neural_region).to(self.device), mask=self.wmasks_dict, target_neuron_idx=target_neuron_idx)
                train_time = time.time() - start_train_time
                real_train_time = self.get_real_train_time(train_time, target_neuron_idx, client.client_id)

                if self.open_client_local_test:
                    client_net = copy.deepcopy(self.net_glob)
                    client_net.load_state_dict(w)
                    acc, loss = test_img(client_net, self.data.test_dataset, self.args.local_bs, self.args.device)

                update_model = UpdateModel()
                update_model.start_train_time = client.current_time
                client.current_time = client.current_time + real_train_time

                neuron_allocated_time[target_neuron_idx] = max(neuron_allocated_time[target_neuron_idx], client.current_time)

                update_model.train_client_id = client.client_id
                update_model.target_neuron_idx = target_neuron_idx
                update_model.w, update_model.loss, update_model.update_time = copy.deepcopy(
                    w), loss, client.current_time
                with torch.no_grad():
                    for name, param in client.region_parament.state_dict().items():
                        w[name] -= param
                    update_model.new_update = w
                self.clients.update_list.add(update_model)
                self.clients.relax_list.add(client)
            lately_update = self.clients.update_list.pop()
            self.aggregate(lately_update)
            self.server_time = lately_update.update_time

            acc, loss = test_img(self.net_glob, self.data.test_dataset, self.args.local_bs, self.args.device)
            save_file_to_json({'server_time': self.server_time,
                               'acc': float(acc),
                               'loss': float(loss),
                               'epoch': epoch,
                               'client_train_num': train_info['client_train_num'],
                               'neural_train_num': train_info['neural_train_num']}, data_file)
            log.print(f"epoch: {epoch}, server_time: {self.server_time}, acc: {acc}, loss: {loss}")
        return train_info

    def train_avg(self, epochs: int, log: MyLog, data_file):
        self.train_init()
        client_train_time = self.get_cmp_array()
        partition_percentage_use_num = np.zeros(self.part_array_num)
        train_info = {
            'client_train_num': [0 for _ in range(self.client_num)],
            'neural_train_num': [0 for _ in range(self.part_array_num)]
        }
        dic = {
            'client_num': self.client_num,
            'part_array': self.part_array,
            'cilent_cmp': self.client_cmp,
            'K': self.K,
        }
        save_file_to_json(dic, data_file)
        if self.open_client_local_test:
            train_info['client_train_acc'] = [[] for _ in range(self.client_num)]
            train_info['client_train_loss'] = [[] for _ in range(self.client_num)]
        neuron_allocated_time = [0 for _ in range(self.part_array_num)]
        self.server_time = 0
        for epoch in range(epochs):
            longest_train_time = 0
            glob_update = None
            for idx in range(self.client_num):
                target_neuron_idx = self.get_target_neuron_idx(idx, client_train_time, partition_percentage_use_num, 0, self.server_time, neuron_allocated_time)
                train_info['neural_train_num'][target_neuron_idx] += 1
                partition_percentage_use_num[target_neuron_idx] += 1
                part_neural_region: nn.Module = copy.deepcopy(self.net_glob)
                mask_param = get_sub_paras(part_neural_region.state_dict(), self.wmasks_dict, target_neuron_idx)
                part_neural_region.load_state_dict(mask_param)

                local = LocalUpdate(self.args, dataset=self.data.train_datas[idx])

                start_train_time = time.time()
                w, loss = local.train(net=copy.deepcopy(part_neural_region).to(self.device), mask=self.wmasks_dict, target_neuron_idx=target_neuron_idx)
                train_time = time.time() - start_train_time
                real_train_time = self.get_real_train_time(train_time, target_neuron_idx, idx)
                longest_train_time = max(longest_train_time, real_train_time)

                update = part_neural_region.state_dict()
                weight = len(self.data.train_datas[idx]) / self.data.train_data_num
                with torch.no_grad():
                    for name, param in update.items():
                        update[name] = w[name] - update[name]
                        update[name] *= weight
                if glob_update is None:
                    glob_update = update
                with torch.no_grad():
                    for name, param in update.items():
                        glob_update[name] += update[name]
            server_param = self.net_glob.state_dict()
            for name, param in server_param.items():
                server_param[name] += glob_update[name]
            self.net_glob.load_state_dict(server_param)
            self.server_time += longest_train_time
            acc, loss = test_img(self.net_glob, self.data.test_dataset, self.args.local_bs, self.args.device)
            save_file_to_json({'server_time': self.server_time,
                               'acc': float(acc),
                               'loss': float(loss),
                               'epoch': epoch,
                               'client_train_num': train_info['client_train_num'],
                               'neural_train_num': train_info['neural_train_num']}, data_file)
            log.print(f"epoch: {epoch}, server_time: {self.server_time}, acc: {acc}, loss: {loss}")
        return train_info

    def aggregate(self, update: UpdateModel):
        update_param = update.new_update
        staleness = get_staleness(update)
        data_proportion = len(self.data.train_datas[update.train_client_id]) / self.data.train_data_num
        with torch.no_grad():
            server_param = self.net_glob.state_dict()
            for name, param in server_param.items():
                server_param[name] += data_proportion * staleness * update_param[name]
            self.net_glob.load_state_dict(server_param)

    def get_real_train_time(self, train_time: float, target_neuron_idx: int, client_id: int):
        return train_time * (self.part_array[target_neuron_idx] / 100) / self.client_cmp[client_id]

    def get_target_neuron_idx(self, client_id, client_train_time, partition_percentage_use_num, select_submodel_type, server_time, neuron_allocated_time):
        target_neuron_idx = 0
        if select_submodel_type == 0:
            for i in range(1, self.part_array_num):
                if client_train_time[client_id, i] <= self.K:
                    if partition_percentage_use_num[target_neuron_idx] > partition_percentage_use_num[i]:
                        target_neuron_idx = i
        elif select_submodel_type == 1:
            target_neuron_idx = random.randint(0, self.part_array_num - 1)
        elif select_submodel_type == 2:
            for i in range(1, self.part_array_num):
                if partition_percentage_use_num[target_neuron_idx] > partition_percentage_use_num[i]:
                    target_neuron_idx = i
        return target_neuron_idx


