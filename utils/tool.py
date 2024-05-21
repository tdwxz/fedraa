import json
import zlib
import base64
import re

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
    }
}


def save_file_to_json(data: dict, file_path: str):
    """
    把字典存入文件，去除上面的预处理
    :param data:
    :param file_path:
    :return:
    """
    s = json.dumps(data)
    s = s.encode('utf-8')
    s = zlib.compress(s)
    s = base64.b64encode(s)
    with open(file_path, 'a+') as f:
        f.write(s.decode('utf8') + '\n')


def read_json_file(file_path):
    """
    读取jsonl文件
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as file:
        data = file.read().split('\n')
    res = []
    for d in data[:-1]:
        d_json = zlib.decompress(base64.b64decode(d))
        res.append(json.loads(d_json.decode('utf8')))
    return res


class MyLog:
    def __init__(self, filename):
        self.file = open(filename, 'a+', encoding='utf-8')
        self.file.write("create new log obj\n")
        self.file.flush()

    def print(self, content='', end='\n'):
        if content is not None:
            content = str(content)
        print(content)
        self.file.write(content + end)
        self.file.flush()

    def close(self):
        self.file.close()


def float_to_str(f: float):
    return str(f).replace('.', 'd')


def read_jsonl(file_path):
    res = []
    out: list[dict] = read_json_file(file_path)
    epochs, accs, server_times, losses = [], [], [], []
    glob_info = None
    for i in range(len(out)):
        if out[i].get('client_num') is not None:
            if glob_info is None:
                glob_info = out[i]
            else:
                res.append({
                    'glob_info': glob_info,
                    'epoch': epochs,
                    'acc': accs,
                    'loss': losses,
                    'server_time': server_times
                })
                glob_info = out[i]
                epochs.clear()
                accs.clear()
                losses.clear()
                server_times.clear()
        else:
            epochs.append(out[i]['epoch'])
            accs.append(out[i]['acc'])
            server_times.append(out[i]['server_time'])
            losses.append(out[i]['loss'])
    res.append({
        'glob_info': glob_info,
        'epoch': epochs,
        'acc': accs,
        'loss': losses,
        'server_time': server_times
    })
    return res


def trans_txt_to_json(file_path, save_file_path):
    dic = {
        'server': []
    }
    pattern = re.compile(r"epoch: (\d+), server_time: (\d+\.\d+), acc: (\d+\.\d+), loss: (\d+\.\d+)")
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                server_time = float(match.group(2))
                acc = float(match.group(3))
                loss = float(match.group(4))
                dic['server'].append((server_time, acc, loss))
                print(f'epoch: {epoch}')
    save_file_to_json(dic, save_file_path)
