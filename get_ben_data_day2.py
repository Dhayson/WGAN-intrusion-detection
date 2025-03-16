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
# Uso: python3 get_ben_data_day2.py dataset_big/Day2 [Diretório com arquivos]
def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"
    
    # Nesse caso já está dividido entre treino e teste, isto é, entre o primeiro e o segundo dia
    df_day_2_ben = GetDataset(sys.argv[1], rs, DATASET_FORMAT, "ben")
    df_day_2_ben = DescartarDuplicatas(df_day_2_ben, do_print=True)
    df_day_2_ben.to_csv("dataset_filtered2/Day2/day2_benign.csv", encoding='utf-8', index=False)
    print()

if __name__ == '__main__':
    main()