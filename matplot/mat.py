import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

log_file_path1 = "/data0/lygao/git/oppo/FL_MPI/adp_server_log/16bit_q_2023-09-21-11_49_57_adp_server_main_q.log" #16bit
log_file_path2 = "/data0/lygao/git/oppo/FL_MPI/adp_server_dblink_log/compose_2_adq16128_2023-10-11-01_13_02_adp_server_main_dblink_c.log" #adp-16~8bit-dwn*0.5
log_file_path3 = "/data0/lygao/git/oppo/FL_MPI/adp_server_dblink_log/adp16~12~8-dwn162023-10-10-11_21_57_adp_server_main_dblink.log"  #adp-16~8bit-dwn16bit
log_file_path4 = "/data0/lygao/git/oppo/FL_MPI/adp_server_dblink_log/16128~dwn-122023-10-10-14_44_28_adp_server_main_dblink.log" #adp-16~8bit-dwn12bit


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
    pattern = r"__\nEpoch: (\d+)\s+.*?Test_AUC: (\d+\.\d+)\s+.*?Downlink bandwidth: (\d+)\s+.*?Uplink bandwidth: (\d+)"
    matches = re.findall(pattern, log_content, re.DOTALL)
    epochs = [int(match[0]) for match in matches]
    auc_scores = [float(match[1]) for match in matches]
    dwn_bandwidth = [int(match[2]) for match in matches]
    up_bandwidth = [int(match[3]) for match in matches]
    bandwidth = [dwn_bandwidth[i] + up_bandwidth[i] for i in range(len(dwn_bandwidth))]
    
    df = pd.DataFrame({'x': epochs, 'y': auc_scores})
    window_size = 10
    df['y_smooth'] = df['y'].rolling(window=window_size, min_periods=3).mean()
    
    return epochs, auc_scores, bandwidth, df

def extract_data0(log_content):
    pattern = r"__\nEpoch: (\d+)\s+.*?Test_AUC: (\d+\.\d+)"
    matches = re.findall(pattern, log_content, re.DOTALL)
    epochs = [int(match[0]) for match in matches]
    auc_scores = [float(match[1]) for match in matches]
    bandwidth = [i*(30*48) for i in range(len(epochs))]
    
    df = pd.DataFrame({'x': epochs, 'y': auc_scores})
    window_size = 10
    df['y_smooth'] = df['y'].rolling(window=window_size, min_periods=3).mean()
    
    return epochs, auc_scores, bandwidth, df  

  
sigma = 1.0
epochs1, aucs1, bandwidth1, df1 = extract_data0(log_content1)
s_aucs1 = gaussian_filter1d(aucs1, sigma=sigma)
epochs2, aucs2, bandwidth2, df2 = extract_data1(log_content2)
s_aucs2 = gaussian_filter1d(aucs2, sigma=sigma)
epochs3, aucs3, bandwidth3, df3 = extract_data1(log_content3)
s_aucs3 = gaussian_filter1d(aucs3, sigma=sigma)
epochs4, aucs4, bandwidth4, df4 = extract_data1(log_content4)
s_aucs4 = gaussian_filter1d(aucs4, sigma=sigma)

plt.figure(figsize=(10, 6))
epoch = True
if epoch:
    x_target1 = epochs1
    x_target2 = epochs2
    x_target3 = epochs3
    x_target4 = epochs4
    label   = 'Epoch'
    title   = 'Test AUC of Training Epoch'
else:
    x_target1 = [x/64*1.58 for x in bandwidth1[:]]
    x_target2 = [x/64*1.58 for x in bandwidth2[:]]
    x_target3 = [x/64*1.58 for x in bandwidth3[:]]
    x_target4 = [x/64*1.58 for x in bandwidth4[:]]
    label   = 'Communication Cost (MB)'
    title   = 'Test AUC of Bandwidth'

plt.plot(x_target1, s_aucs1[:], 'k-', markersize=0, label="Q16bit") 
plt.plot(x_target2, s_aucs2[:],  'r-', markersize=0, label="adpQ-16~8-dwn*0.5") 
plt.plot(x_target3, s_aucs3[:],  'b-', markersize=0, label="adpQ16~8-dwn16bit") 
plt.plot(x_target4, s_aucs4[:],  'c-', markersize=0, label="adpQ16~8-dwn12bit") 
plt.xlabel(label)
plt.ylabel("Test AUC")
plt.title(title)
plt.legend()  # 显示图例
plt.grid(True)


plt.savefig("/data0/lygao/git/oppo/FL_MPI/matplot/pics/adp16128-dwn-epoch1.png")