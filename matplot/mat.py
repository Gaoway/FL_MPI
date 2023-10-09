import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

log_file_path1 = "/data0/lygao/git/oppo/FL_MPI/adp_server_log/2023-09-21-11_49_57_adp_server_main_q.log"
log_file_path2 = "/data0/lygao/git/oppo/FL_MPI/adp_server_log/2023-09-21-14_45_57_adp_server_main_q.log"
log_file_path3 = "/data0/slwang/FL_MPI_CV_REC_Compress/server_log/2023-09-19-10_58_19_server_main.log"  #origin
log_file_path4 = "/data0/lygao/git/oppo/FL_MPI/adp_server_log/2023-09-21-15_58_48_adp_server_main_q.log"

with open(log_file_path1, "r") as file:
    log_content1 = file.read()

with open(log_file_path2, "r") as file:
    log_content2 = file.read()
    
with open(log_file_path3, "r") as file:
    log_content3 = file.read()
    
with open(log_file_path4, "r") as file:
    log_content4 = file.read()
    
def extract_data(log_content):
    epochs = []
    aucs = []

    for line in log_content.split("\n"):
        if "Epoch:" in line:
            epoch = int(re.search(r"Epoch: (\d+)", line).group(1))
            epochs.append(epoch)
        if "Test_AUC:" in line:
            auc = float(re.search(r"Test_AUC: (\d+\.\d+)", line).group(1))
            aucs.append(auc)
    
    return epochs, aucs

def extract_data1(log_content):
    pattern = r"__\nEpoch: (\d+)\s+.*?Test_AUC: (\d+\.\d+)"
    matches = re.findall(pattern, log_content, re.DOTALL)
    epochs = [int(match[0]) for match in matches]
    auc_scores = [float(match[1]) for match in matches]
    
    df = pd.DataFrame({'x': epochs, 'y': auc_scores})
    window_size = 10
    df['y_smooth'] = df['y'].rolling(window=window_size, min_periods=3).mean()
    
    return epochs, auc_scores, df
    
sigma = 0.6
epochs1, aucs1,df1 = extract_data1(log_content1)
s_aucs1 = gaussian_filter1d(aucs1, sigma=sigma)
epochs2, aucs2,df2 = extract_data1(log_content2)
s_aucs2 = gaussian_filter1d(aucs2, sigma=sigma)
epochs3, aucs3,df3 = extract_data1(log_content3)
s_aucs3 = gaussian_filter1d(aucs3, sigma=sigma)
epochs4, aucs4,df4 = extract_data1(log_content4)
s_aucs4 = gaussian_filter1d(aucs4, sigma=1)

plt.figure(figsize=(10, 6))
plt.plot([x * (16 / 1024 /2) for x in epochs1[:420]], s_aucs1[:420], 'k-', markersize=0, label="16bit") 
plt.plot([x * (16 / 1024*(3/8)) for x in epochs1[:420]], s_aucs2[:420],  'r-', markersize=0, label="adp-16-8bit") 
plt.plot([x * (16 / 1024) for x in epochs1[:400]], s_aucs3[:420],  'b-', markersize=0, label="origin") 
plt.plot([x * (16 / 1024/4) for x in epochs1[:420]], s_aucs4[:420],  'c-', markersize=0, label="8bit") 
plt.xlabel("GB")
plt.ylabel("Test AUC")
plt.title("Test AUC of Commnication Cost")
plt.legend()  # 显示图例
plt.grid(True)


plt.savefig("/data0/lygao/git/oppo/FL_MPI/matplot/165.png")