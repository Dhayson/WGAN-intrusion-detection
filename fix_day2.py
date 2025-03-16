from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.random import RandomState

# OBS: o dataset grande não cabe no repositório, mas pode ser baixado em http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/
# Esse arquivo é executado para ajustar os nomes dos ataques, de forma que fiquem iguais aos do dia 1
def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"
    
    df_day_2 = GetDataset("dataset_filtered2/Day2", rs, DATASET_FORMAT, filter=False)
    df_day_2["Label"] = df_day_2["Label"].map({
        'Syn' : 'Syn', 
        "DrDoS_UDP" : "UDP", 
        "UDP-lag" : "UDPLag", 
        "DrDoS_MSSQL" : "MSSQL", 
        "DrDoS_NetBIOS" : "NetBIOS", 
        "DrDoS_LDAP" : "LDAP",
        "BENIGN" : BENIGN,
        "Portmap" : "Portmap"
    })
    print(df_day_2['Label'].value_counts())
    df_day_2[df_day_2["Label"] == BENIGN].to_csv("dataset_filtered2/Day2/day2_benign.csv", encoding='utf-8', index=False)
    df_day_2[df_day_2["Label"] != BENIGN].to_csv("dataset_filtered2/Day2/day2_attack.csv", encoding='utf-8', index=False)
    print()
    print()


if __name__ == '__main__':
    main()