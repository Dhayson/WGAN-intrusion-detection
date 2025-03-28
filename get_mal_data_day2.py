from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.random import RandomState
from main import DescartarDuplicatas

# OBS: o dataset grande não cabe no repositório, mas pode ser baixado em http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/
# Uso: python3 get_mal_data_day2.py dataset_big/Day2 [Diretório com arquivos]
def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"

    df_day_2_mal = GetDataset(sys.argv[1], rs, DATASET_FORMAT, "mal")
    df_day_2_mal = DescartarDuplicatas(df_day_2_mal, do_print=True)
    
    
    for kind in ["Syn", "DrDoS_UDP", "UDP-lag", "DrDoS_MSSQL", "DrDoS_NetBIOS", "DrDoS_LDAP", "UDP", "UDPLag", "MSSQL", "NetBIOS", "LDAP", "Portmap"]:
        num = 11342
        num = min(num, len(df_day_2_mal[df_day_2_mal["Label"] == kind]))
        df_label = df_day_2_mal[df_day_2_mal["Label"] == kind].head(n=num).copy(deep=True)
        df_day_2_mal = df_day_2_mal[df_day_2_mal["Label"] != kind]
        df_day_2_mal = pd.concat([df_day_2_mal, df_label])
    
    print(df_day_2_mal['Label'].value_counts())
    df_day_2_mal.to_csv("dataset_filtered2/Day2/day2_attack.csv", encoding='utf-8', index=False)
    print()

if __name__ == '__main__':
    main()