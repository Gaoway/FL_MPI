import time
import math

from sklearn import metrics

import numpy as np
import torch
import torch.nn as nn

from config import cfg


def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu")):
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)

    train_loss = 0.0
    samples_num = 0
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCELoss()

    t_start = time.time()
    #for data, target in data_loader
    for iter_idx in range(local_iters):
        data, target = next(data_loader)
        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        loss = loss_func(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num

    t_end = time.time()
    
    return train_loss, t_end-t_start


def test(model, data_loader, device=torch.device("cpu")):
    model.eval()
    data_loader = data_loader.loader

    test_loss = 0.0
    correct = 0

    # # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCELoss()
    # with torch.no_grad():
    #     for data, target in data_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         # sum up batch loss
    #         test_loss += loss_func(output, target).item() * data.shape[0]
    #         # get the index of the max log-probability
    #         pred = output.argmax(1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss = test_loss / len(data_loader.dataset)
    # test_accuracy = correct / len(data_loader.dataset)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            print(f"output&tgt:{output},{target}")
            test_loss = loss_func(output, target).item()

            e, f, g = metrics.roc_curve(target.cpu(), output.cpu())
            test_auc = metrics.auc(e, f)

            for i in range(len(output)):
                if (output[i] >= 0.5 and target[i] == 1) or (output[i] < 0.5 and target[i] == 0):
                    correct += 1

    test_accuracy = correct / len(data_loader.dataset)

    # TODO: Record

    return test_loss, test_auc, test_accuracy
