import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

log_file_path1 = "/data0/lygao/git/oppo/FL_MPI/adp_server_log/16bit_q_2023-09-21-11_49_57_adp_server_main_q.log" #16bit
log_file_path2 = "/data0/lygao/git/oppo/FL_MPI/adp_server_dblink_log/adp16~12~82023-10-09-21_32_44_adp_server_main_dblink.log" #adp 16-12-8bit
log_file_path3 = "/data0/lygao/git/oppo/FL_MPI/adp_server_dblink_log/16128~dwn-122023-10-10-14_44_28_adp_server_main_dblink.log"  #dwn12bit
log_file_path4 = "/data0/lygao/git/oppo/FL_MPI/adp_server_dblink_log/adp16~12~8-dwn162023-10-10-11_21_57_adp_server_main_dblink.log" #adpc-16-12-8-d16bit

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
plt.plot([x  for x in epochs1[:]], s_aucs1[:], 'k-', markersize=0, label="16bit") 
plt.plot([x  for x in epochs2[:]], s_aucs2[:],  'r-', markersize=0, label="adp-16~12~8bit") 
plt.plot([x  for x in epochs3[:]], s_aucs3[:],  'b-', markersize=0, label="adpc-16~12~8-dwn12bit") 
plt.plot([x  for x in epochs4[:]], s_aucs4[:],  'c-', markersize=0, label="adpc-16~12~8-dwn16bit") 
plt.xlabel("epoch")
plt.ylabel("Test AUC")
plt.title("Test AUC of Commnication Cost")
plt.legend()  # 显示图例
plt.grid(True)


plt.savefig("/data0/lygao/git/oppo/FL_MPI/matplot/pics/adp16128-dwn16-12eph.png")