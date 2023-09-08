import json
import random
import re
import numpy as np
from matplotlib import pyplot as plt


axis_label_font1 = {
    'color': 'black',
    'weight': 'normal',
    'size': 26
}

marker = ['s', 'p', 'o', '^', '*']
lines = ['-', '-', '-', '-', '-']

colors = plt.get_cmap('Set1')

labels = ['FedAvg', 'Ours', 'Ours_A', 'HeteroFL', 'Adaptive']


def CNN_CIFAR10(acc_file_paths, save_path):
    plt.figure(1, figsize=(6, 5.5))
    plt.ylim(0.5, 0.74)
    plt.xlim(-2, 202)
    algorithms = []
    for idx in range(len(acc_file_paths)):
        epoch_data = []
        file = open(acc_file_paths[idx], 'r')
        acc_data = []
        # search the line including accuracy
        count = 0
        for line in file:
            if re.search('Test_AUC', line):
                acc = re.search('[0]\.[0-9]+', line)  # 正则表达式
                if acc is not None:
                    count += 1
                    acc_data.append(float(acc.group()))  # 提取精度数字
                    epoch_data.append(count)
        if not len(epoch_data) == 0:
            plt.figure(1)
            algorithm, = plt.plot(
                epoch_data,
                acc_data,
                c=colors(idx + 1),
                marker=marker[idx],
                ms=7,
                markevery=(0.14, 0.1),
                linestyle=lines[idx],
                linewidth=1.5,
                label=labels[idx]
            )
            algorithms.append(algorithm)

            plt.xlabel('Epoch', fontdict=axis_label_font1)
            plt.ylabel('AUC', fontdict=axis_label_font1)

            plt.xticks(
                fontsize=24,
                ticks=np.linspace(0, 200, 9),
                labels=[0, None, 50, None, 100, None, 150, None, 200]
            )
            plt.yticks(
                fontsize=24,
                ticks=np.linspace(0.53, 0.73, 9),
                labels=['0.53', None, '0.58', None, '0.63', None, '0.68', None, '0.73']
            )

            plt.grid(True)

    for i in range(int(len(algorithms) / 2)):
        temp = algorithms[i]
        algorithms[i] = algorithms[len(algorithms) - i - 1]
        algorithms[len(algorithms) - i - 1] = temp

    print(labels)

    plt.figure(1)
    plt.legend(handles=algorithms, fontsize=25, loc='lower right')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(1)


acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_24_41_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_25_29_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_45_05_server_main.log']
CNN_CIFAR10(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_101_30.pdf')

acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_09_19_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_04_43_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_57_52_server_main.log']
CNN_CIFAR10(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_101_20.pdf')

acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-11_23_49_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-11_24_53_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-13_05_06_server_main.log']
CNN_CIFAR10(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_101_10.pdf')

labels = ['updates_10', 'updates_20', 'updates_30']
acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-09_46_52_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-10_35_27_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-10_35_45_server_main.log']
CNN_CIFAR10(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_local_updates.pdf')

labels = ['clients_10', 'clients_20', 'clients_30']
acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-10_35_27_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-11_05_34_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-11_06_10_server_main.log']
CNN_CIFAR10(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_clients_num.pdf')

labels = ['weight_0.1', 'weight_0.2', 'weight_0.3']
acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-16_30_18_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-16_38_11_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-20-16_39_01_server_main.log']
CNN_CIFAR10(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_weight.pdf')