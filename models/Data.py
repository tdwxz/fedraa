from torch.utils.data import Subset
from torchvision import datasets, transforms
import torch
import random


def mnist_iid(dataset, num_users):
    labels = dataset.targets
    dict_users = {}
    indices = []
    client_data_num = len(dataset) // num_users
    type_num = 10
    each_type_num = client_data_num // type_num
    for i in range(type_num):
        idx = torch.where(labels == i)[0]
        indices.append(idx.tolist())
    for i in range(num_users):
        choice_type = [i for i in range(0, type_num)]
        dict_user = []
        for type in choice_type:
            select_idx = random.sample(indices[type], each_type_num)
            dict_user += select_idx
        dict_users[i] = dict_user
    return dict_users


def my_mnist_noniid(dataset, alpha, num_users):
    labels = dataset.targets
    dict_users = {}
    indices = []
    client_data_num = len(dataset) // num_users
    choice_type_num = int(10 * alpha)
    type_num = 10
    each_type_num = client_data_num // choice_type_num
    for i in range(type_num):
        idx = torch.where(labels == i)[0]
        indices.append(idx.tolist())
    for i in range(num_users):
        choice_type = random.sample(range(0, type_num), choice_type_num)
        dict_user = []
        for type in choice_type:
            select_idx = random.sample(indices[type], each_type_num)
            dict_user += select_idx
        dict_users[i] = dict_user
    return dict_users


def get_mnist_data(client_num: int, is_iid: bool, alpha:float=0):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=trans_mnist)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=trans_mnist)
    if is_iid:
        dict_users = mnist_iid(train_dataset, client_num)
    else:
        dict_users = my_mnist_noniid(train_dataset, client_num)
    train_datas = []
    for i in range(client_num):
        dic = dict_users[i]
        subset_dataset = Subset(train_dataset, dic)
        train_datas.append(subset_dataset)
    return train_dataset, test_dataset, train_datas, len(train_dataset), len(test_dataset)


def cifar10_iid(dataset, num_users):
    labels = torch.tensor(dataset.targets)
    dict_users = {}
    indices = []
    client_data_num = len(dataset) // num_users
    type_num = 10
    each_type_num = client_data_num // type_num
    for i in range(type_num):
        idx = torch.where(labels == i)[0]
        indices.append(idx.tolist())
    for i in range(num_users):
        choice_type = [i for i in range(0, type_num)]
        dict_user = []
        for type in choice_type:
            select_idx = random.sample(indices[type], each_type_num)
            dict_user += select_idx
        dict_users[i] = dict_user
    return dict_users


def cifar10_noniid(dataset, alpha, num_users):
    labels = torch.tensor(dataset.targets)
    dict_users = {}
    indices = []
    client_data_num = len(dataset) // num_users
    type_num = 10
    choice_type_num = int(10 * alpha)
    each_type_num = client_data_num // choice_type_num
    for i in range(type_num):
        idx = torch.where(labels == i)[0]
        indices.append(idx.tolist())
    for i in range(num_users):
        choice_type = random.sample(range(0, type_num), choice_type_num)
        dict_user = []
        for type in choice_type:
            select_idx = random.sample(indices[type], each_type_num)
            dict_user += select_idx
        dict_users[i] = dict_user
    return dict_users


def get_cifar10_data(client_num: int, is_iid: bool, alpha: float=0):
    trans_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=trans_cifar10)
    dataset_test = datasets.CIFAR10('../data', train=False, download=True, transform=trans_cifar10)
    if is_iid:
        dict_users = cifar10_iid(dataset_train, client_num)
    else:
        dict_users = cifar10_noniid(dataset_train, alpha, client_num)
    train_datas = []
    for i in range(client_num):
        dic = dict_users[i]
        subset_data = Subset(dataset_train, dic)
        train_datas.append(subset_data)
    return dataset_train, dataset_test, train_datas, len(dataset_train), len(dataset_test)


def cifar100_iid(dataset, num_users):
    labels = torch.tensor(dataset.targets)
    dict_users = {}
    indices = []
    client_data_num = len(dataset) // num_users
    type_num = 100
    each_type_num = client_data_num // type_num
    for i in range(type_num):
        idx = torch.where(labels == i)[0]
        indices.append(idx.tolist())
    for i in range(num_users):
        choice_type = [i for i in range(0, type_num)]
        dict_user = []
        for type in choice_type:
            select_idx = random.sample(indices[type], each_type_num)
            dict_user += select_idx
        dict_users[i] = dict_user
    return dict_users


def cifar100_noniid(dataset, alpha, num_users):
    labels = torch.tensor(dataset.targets)
    dict_users = {}
    indices = []
    client_data_num = len(dataset) // num_users
    type_num = 100
    choice_type_num = int(100 * alpha)
    each_type_num = client_data_num // choice_type_num
    for i in range(type_num):
        idx = torch.where(labels == i)[0]
        indices.append(idx.tolist())
    for i in range(num_users):
        choice_type = random.sample(range(0, type_num), choice_type_num)
        dict_user = []
        for type in choice_type:
            select_idx = random.sample(indices[type], each_type_num)
            dict_user += select_idx
        dict_users[i] = dict_user
    return dict_users


def get_cifar100_data(client_num, is_iid, alpha: float = 0):
    trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR100('../data', train=True, download=True, transform=trans_cifar100)
    dataset_test = datasets.CIFAR100('../data', train=False, download=True, transform=trans_cifar100)
    dict_users = cifar100_iid(dataset_train, client_num) if is_iid else cifar100_noniid(dataset_train, alpha, client_num)
    train_datas = []
    for i in range(client_num):
        dic = dict_users[i]
        subset_data = Subset(dataset_train, dic)
        train_datas.append(subset_data)
    return dataset_train, dataset_test, train_datas, len(dataset_train), len(dataset_test)


class Data:
    def __init__(self, client_num, dataset_name, is_iid, alpha:float=0):
        self.dataset_name = dataset_name
        self.client_num = client_num
        self.is_iid = is_iid
        self.alpha = alpha
        self.train_data_func = {
            'mnist': get_mnist_data,
            'cifar10': get_cifar10_data,
            'cifar100': get_cifar100_data
        }
        self.train_dataset, self.test_dataset, self.train_datas, self.train_data_num, self.test_data_num = (
            self.train_data_func[self.dataset_name](client_num, is_iid, alpha))


if __name__ == '__main__':
    data = Data(client_num=10, dataset_name='cifar10', is_iid=True)
