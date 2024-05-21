from models.FedRAA import FedRAA
from utils.options import args_parser
import torch
import configparser
from models.Data import Data
from utils.tool import MyLog, float_to_str
from models.Model import get_model, CNN_MAT_SIZE, MLP_MAT_SIZE
import pickle


torch.cuda.empty_cache()

args = args_parser()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

heterogeneous_strategy_dict = {
    1: {
        1.0: 0.4,
        2.0: 0.3,
        3.0: 0.3
    },
    2: {
        0.1: 0.2,
        0.5: 0.2,
        1.0: 0.3,
        5.0: 0.3
    },
    3: {
        1: 0.1,
        5: 0.9
    },
    4: {
        1: 0.3,
        5: 0.7
    },
    5: {
        1: 0.5,
        5: 0.5
    },
    6: {
        1: 0.7,
        5: 0.3
    },
    7: {
        1: 0.9,
        5: 0.1
    }
}

part_array = {
    2: [40, 60],
    3: [20, 30, 50],
    4: [10, 20, 30, 40],
    5: [5, 10, 20, 30, 35]
}


if __name__ == '__main__':
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    file_name = f'MyRamFed_{args.num_users}_{args.client_cmp}_{args.model}_{args.dataset}_{args.iid}_{float_to_str(args.alpha)}_{args.part_array}_{float_to_str(args.lr)}_{args.K}_{args.getK}_{args.select_submodel_type}'
    log_file = f'./log_file/{file_name}.out'
    data_file = f'./data_anls/{file_name}.jsonl'
    param_file = f'./param/{file_name}.pkl'

    if args.debug:
        log_file = 'debug.out'
        data_file = 'debug.jsonl'

    heterogeneous_strategy = heterogeneous_strategy_dict[args.client_cmp]

    data = Data(args.num_users, args.dataset, args.my_iid, args.my_alpha)
    img_size = data.train_dataset[0][0].shape
    net_glob = get_model(args.dataset, args.model)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    fed_raa = FedRAA(
        client_num=args.num_users,
        net_glob=net_glob,
        heterogeneous_strategy=heterogeneous_strategy,
        part_array=part_array[args.part_array],
        K=args.K,
        args=args,
        device=args.device,
        data=data,
        mat_size=CNN_MAT_SIZE[args.dataset] if args.model == 'cnn' else MLP_MAT_SIZE[args.dataset],
        open_client_local_test=True,
    )

    if args.getK:
        print(fed_raa.get_cmp_array())
    else:
        if args.K < 0:
            raise Exception("Abnormal K-value")
        train_info = fed_raa.train(args.epochs, MyLog(log_file), data_file, args.select_submodel_type)
        with open(param_file, 'wb') as f:
            pickle.dump(fed_raa, f)
