import copy
import asyncio
from collections import OrderedDict
from typing import List
import torch.optim as optim
import torch.nn.functional as F
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

RESULT_PATH = os.getcwd() + '/log/server_law/'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)
now = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))
filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

comm_tags = np.ones(200 + 1)


def main():
    f = open(cfg['data_partition_path'], "r")
    train_data_partition = json.load(f)
    f = open(cfg['test_data_idxes_path'], "r")
    test_data_idxes = json.load(f)
    client_num = len(train_data_partition)
    # load the test dataset and test loader
    train_dataset , test_dataset = datasets.utils.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_batch_size = len(test_dataset)
    
    logger.info(f"Test batch size: {test_batch_size},test idx: {len(test_data_idxes)}")
    
    test_loader = datasets.utils.create_dataloaders(test_dataset, batch_size=test_batch_size + 100, selected_idxs=test_data_idxes, shuffle=False)
    
    valid_idx = test_data_idxes[::3]
    valid_loader = datasets.utils.create_dataloaders(test_dataset, batch_size=128, selected_idxs=valid_idx, shuffle=True)
    
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
    for epoch_idx in range(1, 1 + cfg['epoch_num']):
        logger.info("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        print("_____****_____\nEpoch: {:04d}".format(epoch_idx))


        # The client selection algorithm can be implemented
        selected_num = 30
        selected_client_idxes = sample(range(client_num), selected_num)
        logger.info("Selected client idxes: {}".format(selected_client_idxes))
        print("Selected client idxes: {}".format(selected_client_idxes))
        selected_clients = []
        
        size_weights = []
        for i in selected_client_idxes:
            size_weights.append(len(train_data_partition[i]))
        logger.info(f"size_weights: {size_weights}")
        size_weights = [i/sum(size_weights) for i in size_weights]
        
        for client_idx in selected_client_idxes:
            all_clients[client_idx].epoch_idx = epoch_idx
            all_clients[client_idx].lr = max(cfg['decay_rate'] * all_clients[client_idx].lr, cfg['min_lr'])
            all_clients[client_idx].local_updates = min(int(len(all_clients[client_idx].train_data_idxes) / cfg['local_batch_size']), cfg['local_updates'])
            all_clients[client_idx].train_batch_size = cfg['local_batch_size']
            all_clients[client_idx].params_dict = copy.deepcopy(global_model.state_dict())
            selected_clients.append(all_clients[client_idx])
                        

        # 每一轮都需要将选中的客户端的配置（client.config）发送给相应的客户端
        communication_parallel(selected_clients, action="send_config")

        # 从选中的客户端那里接收配置，此时选中的客户端均已完成本地训练。配置包括训练好的本地模型，学习率等
        communication_parallel(selected_clients, action="get_config")
        
        #use FedLAW to better aggregate models
        parameters = [client.params_dict for client in selected_clients]
        gamma, optimized_weight = fedlaw_optimization(size_weights, parameters, global_model,  valid_loader)

        # aggregate the clients' local model parameters
        #aggregate_models(global_model, selected_clients)
        logger.info(f"gamma: {gamma}, optimized_weight: {optimized_weight}, size_weights: {size_weights}, sum: {sum(size_weights)}")

        global_model = fedlaw_generate_global_model(gamma, optimized_weight, selected_clients, global_model)
        
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


def fedlaw_optimization(size_weights, parameters, central_node, train_loader):
    cohort_size = len(parameters)
    server_lr = 0.01
    server_epochs = 10
    
    optimizees = torch.tensor([torch.log(torch.tensor(j)) for j in size_weights] + [0.0], device='cuda', requires_grad=True)
    optimizee_list = [optimizees]
    
    optimizer = optim.SGD(optimizee_list, lr=server_lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    for i in range(len(optimizee_list)):
        optimizee_list[i].grad = torch.zeros_like(optimizee_list[i])
        
    softmax = nn.Softmax(dim=0)
    
    new_node = copy.deepcopy(central_node)
    new_node.train()
    for epoch in range(server_epochs): 
        for itr, (data, target) in enumerate(train_loader.loader):
            for i in range(cohort_size):
                if i == 0:
                    #model_param = torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i]*parameters[i]
                    for param, new_param in zip(new_node.parameters(), parameters[i]):
                        print(softmax(optimizees[:-1]))  # 检查 softmax 输出
                        print(softmax(optimizees[:-1])[i])  
                        print(torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i])  
                        print(torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i]*new_param)  
                        param.copy_(torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i]*new_param)
                else:
                    #model_param = model_param.add(torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i]*parameters[i])
                    for param, new_param in zip(new_node.parameters(), parameters[i]):
                        param.add_(torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i]*new_param)
            
            #with torch.no_grad():
            #    for param, new_param in zip(new_node.parameters(), model_param):
            #        param.copy_(new_param)
                    
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = new_node(data)
            loss =  F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        optmized_weights = [j for j in softmax(optimizees[:-1]).detach().cpu().numpy()]
        learned_gamma = torch.exp(optimizees[-1])
    return learned_gamma, optmized_weights

def fedlaw_generate_global_model(gamma, optmized_weights, client_params, central_node):
    for i in range(len(client_params)):
        if i == 0:
            fedlaw_param = gamma*optmized_weights[i]*client_params[i]
        else:
            fedlaw_param = fedlaw_param.add(gamma*optmized_weights[i]*client_params[i])
    central_node.load_param(copy.deepcopy(fedlaw_param.detach()))

    return central_node     

if __name__ == "__main__":
    main()
