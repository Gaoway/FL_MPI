import pandas as pd
import re
#/data0/slwang/FL_MPI_CV_REC_Compress/server_log/2023-09-19-10_58_19_server_main.log
#/data0/lygao/git/oppo/FL_MPI/adp_server_log/8bit_q_2023-09-21-15_58_48_adp_server_main_q.log
#/data0/lygao/git/oppo/FL_MPI/adp_server_log/16bit_q_2023-09-21-11_49_57_adp_server_main_q.log
#/data0/lygao/git/oppo/FL_MPI/adp_server_log/16-8bit-adp2023-09-21-14_45_57_adp_server_main_q.log
log_file_path1 = "/data0/lygao/git/oppo/FL_MPI/adp_server_log/16-8bit-adp2023-09-21-14_45_57_adp_server_main_q.log"

with open(log_file_path1, "r") as file:
    log_content1 = file.read()

def extract_data(log_content):
    epochs = []
    aucs = []
    
    pattern = r"__\nEpoch: (\d+)\s+.*?Test_AUC: (\d+\.\d+)"
    matches = re.findall(pattern, log_content, re.DOTALL)
    epochs = [int(match[0]) for match in matches]
    aucs = [float(match[1]) for match in matches]

    # Ensure that both lists have the same length
    min_len = min(len(epochs), len(aucs))
    epochs = epochs[:min_len]
    aucs = aucs[:min_len]
    
    print(len(epochs),epochs)
    print(len(aucs),aucs)

    # Create a DataFrame with epochs and aucs as columns
    df = pd.DataFrame({'Epoch': epochs, 'Test_AUC': aucs})
    
    # Save the DataFrame to an Excel file
    df.to_excel('/data0/lygao/git/oppo/FL_MPI/matplot/xlsx/adp16-8bit.xlsx', index=False, engine='openpyxl')
    return epochs, aucs

extract_data(log_content1)
