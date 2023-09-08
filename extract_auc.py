import re
import pandas as pd

labels = ['FedAvg', 'Ours', 'Ours_A']

def extract_auc(auc_file_paths, save_path):
    datas = dict()
    datas['Epoch'] = list(range(1, 201))
    for idx in range(len(auc_file_paths)):
        file = open(auc_file_paths[idx], 'r')
        auc_data = []
        # search the line including accuracy
        count = 0
        for line in file:
            if re.search('Test_AUC', line):
                auc = re.search('[0]\.[0-9]+', line)  # 正则表达式
                if auc is not None:
                    count += 1
                    auc_data.append(float(auc.group()))  # 提取精度数字
        while count < 200:
            count += 1
            auc_data.append(None)
        datas[labels[idx]] = auc_data

    datas = pd.DataFrame(datas)
    datas.to_csv(save_path, index=False)


acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_24_41_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_25_29_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_45_05_server_main.log']
extract_auc(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_101_30.csv')

acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_09_19_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_04_43_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-12_57_52_server_main.log']
extract_auc(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_101_20.csv')

acc_file_paths = ['/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-11_23_49_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-11_24_53_server_main.log',
                  '/data/slwang/FL_MPI_CV_REC/server_log/2023-07-19-13_05_06_server_main.log']
extract_auc(acc_file_paths, '/data/slwang/FL_MPI_CV_REC/results/epoch_acc_101_10.csv')

