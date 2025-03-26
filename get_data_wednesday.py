from src.import_dataset_alt import GetDataset2017
from src.dataset_split import SplitDataset
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.random import RandomState
from main import DescartarDuplicatas

# OBS: o dataset aqui é o Cic IDS 2017
def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    BENIGN = "BENIGN"
    
    # Nesse caso já está dividido entre treino e teste, isto é, entre o primeiro e o segundo dia
    df_wednesday = GetDataset2017(sys.argv[1], rs, DATASET_FORMAT, filter=True, do_print=False)
    df_wednesday = DescartarDuplicatas(df_wednesday, do_print=True)
    
    for kind in [BENIGN]:
        num = 50000
        df_label = df_wednesday[df_wednesday["Label"] == kind].head(n=num).copy(deep=True)
        df_wednesday = df_wednesday[df_wednesday["Label"] != kind]
        df_wednesday = pd.concat([df_wednesday, df_label])
        
    
    for kind in [
            "DoS Hulk",
            "DoS GoldenEye",
            "DrDoS_NetBIOS",
            "DoS slowloris",
            "DoS Slowhttptest",
            "Heartbleed",
        ]:
        num = 40000
        df_label = df_wednesday[df_wednesday["Label"] == kind].head(n=num).copy(deep=True)
        df_wednesday = df_wednesday[df_wednesday["Label"] != kind]
        df_wednesday = pd.concat([df_wednesday, df_label])
    
    print(df_wednesday['Label'].value_counts())
    df_wednesday.to_csv("dataset_alternative_filtered/Day3/wednesday.csv", encoding='cp1252', index=False)
    print()

if __name__ == '__main__':
    main()