import copy
import asyncio
from collections import OrderedDict
from typing import List

from torch import nn
import json

import csv

from comm_utils import send_data, get_data
from config import cfg
import os
import time
import random
from random import sample

import numpy as np
import torch
from client import *
import datasets.utils
from models import utils
from training_utils import test

from mpi4py import MPI

import logging

random.seed(cfg['client_selection_seed'])

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['server_cuda']
device = torch.device("cuda" if cfg['server_use_cuda'] and torch.cuda.is_available() else "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

RESULT_PATH = os.getcwd() + '/adp_server_log/'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'

fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

comm_tags = np.ones(200 + 1)

def quantization(params_dict, level=32):
    for name, param in params_dict.items():
        if param.dtype == torch.float32:
            if level == 16:#16bit
                params_dict[name] = param.to(torch.float16)
                
            elif level == 12:
                param_i = torch.round((param*2047))
                param_i = param_i.to(torch.int16)
                param_f = param_i/(2047.0)
                params_dict[name] = param_f
            
            elif level == 8:
                param_i = torch.round((param*127))
                param_i = param_i.to(torch.int16)
                param_f = param_i/(127.0)
                params_dict[name] = param_f
            else:
                break
             
    return params_dict 

def main():
    f = open(cfg['data_partition_path'], "r")
    train_data_partition = json.load(f)
    f = open(cfg['test_data_idxes_path'], "r")
    test_data_idxes = json.load(f)
    client_num = len(train_data_partition)
    # load the test dataset and test loader
    _, test_dataset = datasets.utils.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_batch_size = len(test_dataset)
    print(test_batch_size)
    test_loader = datasets.utils.create_dataloaders(test_dataset, batch_size=test_batch_size + 100, selected_idxs=test_data_idxes, shuffle=False)
    logger.info(f"Test quantization: {8}")
    logger.info("Total number of clients: {}".format(client_num))
    logger.info("\nModel type: {}".format(cfg["model_type"]))
    logger.info("Dataset: {}".format(cfg["dataset_type"]))

    # init the global model
    global_model = utils.create_model_instance(cfg['model_type'], test_dataset.field_dims)
    global_model.to(device)
    global_params_num = nn.utils.parameters_to_vector(global_model.parameters()).nelement()
    global_model_size = global_params_num * 4 / 1024 / 1024
    logger.info("Global params num: {}".format(global_params_num))
    logger.info("Global model Size: {} MB".format(global_model_size))

    # create clients
    all_clients: List[ClientConfig] = list()
    for client_idx in range(client_num):
        client = ClientConfig(client_idx)
        client.lr = cfg['lr'] / cfg['decay_rate']
        client.train_data_idxes = train_data_partition[client_idx]
        all_clients.append(client)

    best_epoch = 1
    best_auc = 0
    # begin each epoch
    down_total_bandwidth    = 0 # full model as 32, 16 quantization as 16, and 8 quantization as 8. 
    up_total_bandwidth  = 0
    
    for epoch_idx in range(1, 1 + cfg['epoch_num']):
        logger.info("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        print("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        
        # The client selection algorithm can be implemented
        selected_num = 30
        selected_client_idxes = sample(range(client_num), selected_num)
        logger.info("Selected client idxes: {}".format(selected_client_idxes))
        print("Selected client idxes: {}".format(selected_client_idxes))
        
        selected_clients = []
        # get the selected clients' train data length
        train_data_len = []
        for i in selected_client_idxes:
            train_data_len.append(train_data_partition[i])
        sorted_train_data_len = sorted(train_data_len)
        
        part_ratio1 = 0.6-0.4*(epoch_idx-1)/cfg['epoch_num'] # 60%~20%
        part_ratio2 = 0.8-0.4*(epoch_idx-1)/cfg['epoch_num'] # 80%~40%
        part_point1 = sorted_train_data_len[int(selected_num * part_ratio1)]
        part_point2 = sorted_train_data_len[int(selected_num * part_ratio2)]
        logger.info(f"8 bit point: {int(selected_num * part_ratio1)} ")
        logger.info(f"16 bit point: {int(selected_num * (part_ratio2-part_ratio1))} ")
        
        for client_idx in selected_client_idxes:
            if True and epoch_idx <= 6:
                all_clients[client_idx].q_level = 16
                up_total_bandwidth += 16
            elif train_data_partition[client_idx] <= part_point1:
                all_clients[client_idx].q_level = 8
                up_total_bandwidth += 8
            elif train_data_partition[client_idx] <= part_point2:
                all_clients[client_idx].q_level = 12
                up_total_bandwidth += 12
            else:
                all_clients[client_idx].q_level = 16
                up_total_bandwidth += 16
                
            all_clients[client_idx].epoch_idx = epoch_idx
            all_clients[client_idx].lr = max(cfg['decay_rate'] * all_clients[client_idx].lr, cfg['min_lr'])
            all_clients[client_idx].local_updates = min(int(len(all_clients[client_idx].train_data_idxes) / cfg['local_batch_size']), cfg['local_updates'])
            all_clients[client_idx].train_batch_size = cfg['local_batch_size']
            all_clients[client_idx].params_dict = copy.deepcopy(global_model.state_dict())
            down_total_bandwidth += 32
            selected_clients.append(all_clients[client_idx])
                        

        # 每一轮都需要将选中的客户端的配置（client.config）发送给相应的客户端
        communication_parallel(selected_clients, action="send_config")

        # 从选中的客户端那里接收配置，此时选中的客户端均已完成本地训练。配置包括训练好的本地模型，学习率等
        communication_parallel(selected_clients, action="get_config")

        # aggregate the clients' local model parameters
        aggregate_models(global_model, selected_clients)

        # test the global model
        test_loss, test_auc, test_acc = test(global_model, test_loader, device)

        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch = epoch_idx
            model_save_path = cfg['model_save_path'] + now + '/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path, exist_ok=True)
            torch.save(global_model.state_dict(), model_save_path + now + '.pth')

        logger.info(
            "Test_Loss: {:.4f}\n".format(test_loss) +
            "Test_ACC: {:.4f}\n".format(test_acc) + 
            "Test_AUC: {:.4f}\n".format(test_auc) +
            "Best_AUC: {:.4f}\n".format(best_auc) +
            "Best_Epoch: {:04d}\n".format(best_epoch)
        )
        
        logger.info(f"Downlink bandwidth: {down_total_bandwidth}")
        logger.info(f"Uplink bandwidth: {up_total_bandwidth}")

        for m in range(len(selected_clients)):
            comm_tags[m + 1] += 1


def aggregate_models(global_model, client_list):
    with torch.no_grad():
        params_dict = copy.deepcopy(global_model.state_dict())
        total_updates = 0
        for client in client_list:
            total_updates += client.local_updates
        
        for client in client_list:
            client.aggregate_weight = client.local_updates / total_updates
            for k, v in client.params_dict.items():
                params_dict[k] = params_dict[k].detach() + copy.deepcopy(
                    client.aggregate_weight * (v - global_model.state_dict()[k])
                )
    global_model.load_state_dict(params_dict)


async def send_config(client, client_rank, comm_tag):
    await send_data(comm, client, client_rank, comm_tag)


async def get_config(client, client_rank, comm_tag):
    config_received = await get_data(comm, client_rank, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(client, k, v)


def communication_parallel(client_list, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for m, client in enumerate(client_list):
        if action == "send_config":
            task = asyncio.ensure_future(send_config(client, m + 1, comm_tags[m+1]))
        elif action == "get_config":
            task = asyncio.ensure_future(get_config(client, m + 1, comm_tags[m+1]))
        else:
            raise ValueError('Not valid action')
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


if __name__ == "__main__":
    main()
