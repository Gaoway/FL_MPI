import asyncio
import os
import time

from config import cfg

import torch
import torch.optim as optim
from client import ClientConfig
from comm_utils import *
from training_utils import train, test
from models import utils
from mpi4py import MPI
import logging
import numpy as np

import copy

import datasets.utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

if cfg['client_cuda'] == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(rank) % 4 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['client_cuda']
device = torch.device("cuda" if cfg['client_use_cuda'] and torch.cuda.is_available() else "cpu")


now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/adp_clients_dblink_log/' + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

MASTER_RANK = 0


def main():
    client_config = ClientConfig(idx=0)

    # dataset and the test dataloader
    train_dataset, _ = datasets.utils.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    # test_loader = datasets.create_dataloaders(test_dataset, batch_size=cfg['client_test_batch_size'], shuffle=False)

    # begin each epoch
    comm_tag = 1
    while True:
        # receive the configuration from the server
        communicate_with_server(client_config, comm_tag, action='get_config')
        
        logger = init_logger(comm_tag, client_config)

        logger.info("_____****_____\nEpoch: {:04d}".format(client_config.epoch_idx))

        #torch.manual_seed(12345)  # 设置CPU和CUDA种子
        #if torch.cuda.is_available():
        #    torch.cuda.manual_seed_all(12345)  # 设置所有GPU的种子
            
        train_loader = datasets.utils.create_dataloaders(
            train_dataset, batch_size=client_config.train_batch_size, selected_idxs=client_config.train_data_idxes
        )
        
        # start local training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [asyncio.ensure_future(local_training(client_config, train_loader, None, logger))]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        # send the configuration to the server
        communicate_with_server(client_config, comm_tag, action='send_config')
        comm_tag += 1

        if client_config.epoch_idx >= cfg['epoch_num']:
            break


async def get_config(config, comm_tag):
    config_received = await get_data(comm, MASTER_RANK, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)


async def send_config(config, comm_tag):
    await send_data(comm, config, MASTER_RANK, comm_tag)


def communicate_with_server(config, comm_tag, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    if action == "send_config":
        task = asyncio.ensure_future(
            send_config(config, comm_tag)
        )
    elif action == "get_config":
        task = asyncio.ensure_future(
            get_config(config, comm_tag)
        )
    else:
        raise ValueError('Not valid action')
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def init_logger(comm_tag, client_config):
    logger = logging.getLogger(os.path.basename(__file__).split('.')[0] + str(comm_tag))
    logger.setLevel(logging.INFO)
    filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '_' + str(
        client_config.idx) + '.log'
    file_handler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


async def local_training(config, train_loader, test_loader, logger):
    local_model = utils.create_model_instance(cfg['model_type'], field_dims=train_loader.loader.dataset.data.field_dims)
    local_model.load_state_dict(config.params_dict)
    local_model.to(device)

    if cfg['momentum'] < 0:
        optimizer = optim.SGD(local_model.parameters(), lr=config.lr, weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.SGD(local_model.parameters(), momentum=cfg['momentum'], lr=config.lr, weight_decay=cfg['weight_decay'])

    train_loss, train_time = train(
        local_model,
        train_loader,
        optimizer,
        local_iters=config.local_updates,
        device=device
    )

    logger.info(
        "Train_Loss: {:.4f}\n".format(train_loss) +
        "Train_Time: {:.4f}\n".format(train_time)
    )

    params_dict = copy.deepcopy(local_model.state_dict())
    
    params_dict  = quantization(params_dict, level = config.q_level)
    logger.info(f"{config.idx}: Quantization level: {config.q_level}")

    config.params_dict  = params_dict
    config.train_time = train_time

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
                
def model_quantization(params_dict, level=0):
    for name, param in params_dict.items():
        if param.dtype == torch.float32:
            if level == 16:#16bit
                params_dict[name] = param.to(torch.float16)
                break
            elif level == 10:
                all_parts = 1023
                             
            elif level == 8: #8bit
                all_parts = 255

            elif level == 4:  #4bit
                all_parts = 15
            else:
                return params_dict   
             
            min_val = param.min()
            max_val = param.max()
            scale = all_parts / (max_val - min_val)
            zero_point = -min_val * scale
            param = (param * scale + zero_point+0.5).clamp(0, all_parts).byte()
            
            param_dq = (param.float() - zero_point) / scale
            params_dict[name]   = param_dq
       
    return params_dict

if __name__ == '__main__':
    main()
