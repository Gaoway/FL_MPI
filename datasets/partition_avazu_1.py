import csv 
import numpy as np
import logging
from collections import defaultdict
import copy
import json
import pandas as pd

# 设置一个阈值，将含有数据量大于这个阈值的device_id筛选出来，作为worker

dataset_path = '/data/lygao/dataset/avazu/train.csv'
train_data_partition_path = '/data/lygao/dataset/avazu/partition/train_data_partition.json'
test_data_partition_path = '/data/lygao/dataset/avazu/partition/test_data_partition.json'
test_data_idxes_path = '/data/lygao/dataset/avazu/partition/test_data_idxes.json'
active_data_path = '/data/lygao/dataset/avazu/partition/data_results.csv'
device_id_data_path = '/data/lygao/dataset/avazu/partition/device_id_data.json'
active_device_id_data_path = '/data/lygao/dataset/avazu/partition/active_device_id_data.json'
data_cache = False


data_num_per_worker_min = 100
data_num_per_worker_max = 1000
train_ratio = 0.8
test_ratio = 0.1

partition_type = 11  # device_id: 11, device_ip: 12

# print('取出数据集中的所有device_id, 保存为一个list........')
# # 取出数据集中所有的device_id
# with open(dataset_path, 'r') as f: 
#     reader = csv.reader(f)  
#     device_ids = [row[partition_type] for row in reader]
#     # device_ids = device_ids[: data_num + 1]
# del device_ids[0]
# device_ids = set(device_ids)
# device_ids = list(device_ids)
# print('数据集中的所有device_id数量为：', len(device_ids))
# print()


if data_cache:
    print('取出每一个device_id包含的数据id，保存为字典...........')
    print()
    # 得出每一个device_id包含哪些数据
    cnt = 0
    device_id_datas = defaultdict(list)
    with open(dataset_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            device_id = row[partition_type]
            if device_id != 'device_id':
                device_id_datas[device_id].append(cnt)
                cnt += 1
                # if cnt >= data_num:
                #     break
    f.close()
    # 保存每一个device_id包含哪些数据
    f = open(device_id_data_path, "w")
    json.dump(device_id_datas, f)
    f.close()

# 读取每一个device_id包含哪些数据
f = open(device_id_data_path, "r")
device_id_datas = json.load(f)

print('筛选满足数据数量的device_id以及包含的数据id，保存为字典.........')
active_device_id_datas = dict()
# 筛选满足数据数量要求的device_id
active_device_count = 0
count_data = 0
for k, v in device_id_datas.items():
    if data_num_per_worker_min <= len(v) <= data_num_per_worker_max:
        active_device_count += 1
        active_device_id_datas[k] = v
        count_data += len(v)
# 保存满足要求的device_id包含哪些数据
f = open(active_device_id_data_path, "w")
json.dump(active_device_id_datas, f)
f.close()
print('满足要求的device_id有', active_device_count, "个")
print('包含', count_data, '个数据')
print()

print('加载整个数据集数据...............')
active_data = []
client_data_idxes = [[] for _ in range(active_device_count)]
data_idx = 0
# 取出所有的数据
all_data = pd.read_csv(dataset_path)

print('将所有的客户端数据取出来...........')
print('将数据idx划分到每一个客户端.........')
# 整理出每一个客户端包含的data_idx，这是一个list，list中的每一个元素也是一个list，代表一个客户端中包含的数据id
# 将所有客户端的数据从整个数据集中取出
for client_idx, (k, v) in enumerate(active_device_id_datas.items()):
    for data_idx_global in v: 
        client_data_idxes[client_idx].append(data_idx)
        data_idx += 1
        active_data.append(all_data.loc[data_idx_global])
pd.DataFrame(active_data).to_csv(active_data_path, index=False)

print('从每个客户端的数据idx中划分训练集和测试集.........')
print('将每一个客户端的测试集进行汇总，得到总的测试集idx..........')
# 分别得出每一个客户端的训练data_idxes和测试data_idxes
train_data_partition = []
test_data_partition = []
test_idxes = []
for data_idxes in client_data_idxes:
    data_num = len(data_idxes)
    train_num = int(data_num * train_ratio)
    train_data_partition.append(copy.deepcopy(data_idxes[: train_num]))
    test_data_partition.append(copy.deepcopy(data_idxes[train_num: ]))
    test_idxes.extend(copy.deepcopy(data_idxes[train_num: ]))
f = open(train_data_partition_path, "w")
json.dump(train_data_partition, f)
f.close()
f = open(test_data_partition_path, "w")
json.dump(test_data_partition, f)
f.close()
f = open(test_data_idxes_path, "w")
json.dump(test_idxes, f)
f.close()

print('测试集数据：', len(test_idxes), '个')
train_data_num = 0
for x in train_data_partition:
    train_data_num += len(x)

print('训练集数据：', train_data_num, '个')
