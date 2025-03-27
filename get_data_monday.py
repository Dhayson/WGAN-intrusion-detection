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
    df_monday = GetDataset2017(sys.argv[1], rs, DATASET_FORMAT, filter=True, do_print=False)
    df_monday = DescartarDuplicatas(df_monday, do_print=True)
    
    for kind in [BENIGN]:
        num = 50000
        df_label = df_monday[df_monday["Label"] == kind].head(n=num).copy(deep=True)
        df_monday = df_monday[df_monday["Label"] != kind]
        df_monday = pd.concat([df_monday, df_label])
    
    print(df_monday['Label'].value_counts())
    df_monday.to_csv("dataset_alternative_filtered/Day1/monday.csv", encoding='cp1252', index=False)
    print()

if __name__ == '__main__':
    main()