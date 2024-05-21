from models.Nets import MLP
from torch import nn
import torch.nn.functional as F


mat_size = {
    'mnist': (28 * 28, 200, 10),
    'cifar10': (32 * 32 * 3, 2048, 10),
    'cifar100': (32 * 32 * 3, 4096, 100),
}

# 用于神经网络分割时的模型分割参数
MLP_MAT_SIZE = {
    'mnist': {'layer_hidden.weight': (10, 200)},
    'cifar10': {'layer_hidden.weight': (10, 2048)},
    'cifar100': {'layer_hidden.weight': (100, 4096)}
}

CNN_MAT_SIZE = {
    'mnist': {'conv2.weight': (20, 10, 5, 5), 'fc1.weight': (50, 320), 'fc2.weight': (10, 50)},
    'cifar10': {'conv2.weight': (16, 6, 5, 5), 'fc1.weight': (120, 16 * 5 * 5), 'fc2.weight': (84, 120)},
    'cifar100': {'conv2.weight': (16, 6, 5, 5), 'fc1.weight': (120, 16 * 5 * 5), 'fc2.weight': (84, 120)},
}


class ClientSide(nn.Module):
    def __init__(self, use_data, use_model):
        super(ClientSide, self).__init__()
        self.use_data = use_data
        self.use_model = use_model
        if use_model == 'mlp':
            self.layer_input = nn.Linear(mat_size[use_data][0], mat_size[use_data][1])
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()
        elif use_model == 'cnn' and use_data == 'mnist':
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
        elif use_model == 'cnn' and use_data in ('cifar10', 'cifar100'):
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
        elif use_model == 'cnn' and use_data == 'tiny-imagenet':
            pass
        else:
            raise Exception(f"无法创建ClientSide，未识别use_model：{use_model}和use_data：{use_data}")

    def forward(self, x):
        if self.use_model == 'mlp':
            x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
            x = self.layer_input(x)
            x = self.dropout(x)
            x = self.relu(x)
        elif self.use_model == 'cnn' and self.use_data == 'mnist':
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        elif self.use_model == 'cnn' and self.use_data in ('cifar10', 'cifar100'):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
        elif self.use_model == 'cnn' and self.use_data == 'tiny-imagenet':
            pass
        return x


# 定义服务端的神经网络模型
class ServerSide(nn.Module):
    def __init__(self, use_data, use_model):
        super(ServerSide, self).__init__()
        self.use_data = use_data
        self.use_model = use_model
        if use_model == 'mlp':
            self.layer_hidden = nn.Linear(mat_size[use_data][1], mat_size[use_data][2])
        elif use_model == 'cnn' and use_data == 'mnist':
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
        elif use_model == 'cnn' and use_data == 'cifar10':
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        elif use_model == 'cnn' and use_data == 'cifar100':
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 100)
        elif use_model == 'cnn' and use_data == 'tiny-imagenet':
            pass
        else:
            raise Exception(f"无法创建ClientSide，未识别use_model：{use_model}和use_data：{use_data}")

    def forward(self, x):
        if self.use_model == 'mlp':
            x = self.layer_hidden(x)
        elif self.use_model == 'cnn' and self.use_data == 'mnist':
            x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
        elif self.use_model == 'cnn' and self.use_data in ('cifar10', 'cifar100'):
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.use_model == 'cnn' and self.use_data == 'tiny-imagenet':
            pass
        return x




class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(data_name: str, model_name: str, is_splitfl: bool = False):
    if not is_splitfl:
        if data_name == 'mnist' and model_name == 'mlp':
            return MLP(28 * 28, 200, 10)
        elif data_name == 'cifar10' and model_name == 'mlp':
            return MLP(32 * 32 * 3, 2048, 10)
        elif data_name == 'cifar100' and model_name == 'mlp':
            return MLP(32 * 32 * 3, 4096, 100)
        elif data_name == 'tiny-imagenet' and model_name == 'mlp':
            return MLP(3 * 64 * 64, 4096, 200)
        elif data_name == 'mnist' and model_name == 'cnn':
            return CNNMnist()
        elif data_name == 'cifar10' and model_name == 'cnn':
            return CNNCifar(10)
        elif data_name == 'cifar100' and model_name == 'cnn':
            return CNNCifar(100)
        elif data_name == 'tiny-imagenet' and model_name == 'cnn':
            pass
        else:
            raise Exception('无法识别data_name或者model_name')
    else:
        return ClientSide(data_name, model_name), ServerSide(data_name, model_name)


if __name__ == '__main__':
    cnn_mnist = CNNMnist()
    cnn_cifar = CNNCifar(10)
    print(f"cnn_mnist: {cnn_mnist.state_dict().keys()}")
    print(f"cnn_cifar: {cnn_cifar.state_dict().keys()}")
