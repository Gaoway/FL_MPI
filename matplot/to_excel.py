import pandas as pd
import re
#/data0/slwang/FL_MPI_CV_REC_Compress/server_log/2023-09-19-10_58_19_server_main.log
#/data0/lygao/git/oppo/FL_MPI/adp_server_log/8bit_q_2023-09-21-15_58_48_adp_server_main_q.log
#/data0/lygao/git/oppo/FL_MPI/adp_server_log/16bit_q_2023-09-21-11_49_57_adp_server_main_q.log
#/data0/lygao/git/oppo/FL_MPI/adp_server_log/16-8bit-adp2023-09-21-14_45_57_adp_server_main_q.log

def extract_data(file_path):
    epochs = []
    aucs = []
    with open(file_path, "r") as file:
        log_content = file.read()
    
    pattern = r"__\nEpoch: (\d+)\s+.*?Test_AUC: (\d+\.\d+)"
    matches = re.findall(pattern, log_content, re.DOTALL)
    epochs = [int(match[0]) for match in matches]
    aucs = [float(match[1]) for match in matches]

    # Ensure that both lists have the same length
    min_len = min(len(epochs), len(aucs))
    epochs = epochs[:min_len]
    aucs = aucs[:min_len]
    
    return epochs, aucs

path1 = '/data0/lygao/git/oppo/FL_MPI/log/server_log/2023-11-30-01_38_server_main.log'#100
path3 = '/data0/lygao/git/oppo/FL_MPI/log/server_log/2023-11-30-02_25_server_main.log'#95
path2 = '/data0/lygao/git/oppo/FL_MPI/log/server_log/2023-11-30-11_10_server_main.log'#90
path4 = '/data0/lygao/git/oppo/FL_MPI/log/server_log/2023-11-30-11_12_server_main.log'#85
path5 = '/data0/lygao/git/oppo/FL_MPI/log/server_log/2023-11-30-17_07_server_main.log'# 99
path6 = '/data0/lygao/git/oppo/FL_MPI/log/server_log/2023-11-30-17_04_server_main.log'#97

epochs, aucs1 = extract_data(path1)
_, aucs2 = extract_data(path2)
_, aucs3 = extract_data(path3)
_, aucs4 = extract_data(path4)
_, aucs5 = extract_data(path5)
_, aucs6 = extract_data(path6)

maxlen = max(len(aucs1), len(aucs2), len(aucs3), len(aucs4), len(epochs), len(aucs5), len(aucs6))
for i in [epochs, aucs1, aucs2, aucs3, aucs4, aucs5, aucs6]:
    if len(i) < maxlen:
        i.extend([None] * (maxlen - len(i)))

df = pd.DataFrame({'Epoch': epochs, '1.0': aucs1, '0.95': aucs2, '0.9': aucs3, '0.85': aucs4, '0.99': aucs5, '0.97': aucs6})

    # Save the DataFrame to an Excel file
df.to_excel('/data0/lygao/git/oppo/FL_MPI/matplot/xlsx/re_weight_aggre.xlsx',)
