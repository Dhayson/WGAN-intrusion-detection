from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan import Train
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.random import RandomState

# OBS: o dataset grande não cabe no repositório, mas pode ser baixado em http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/

def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"
    
    # Nesse caso já está dividido entre treino e teste, isto é, entre o primeiro e o segundo dia
    df_day_1 = GetDataset(sys.argv[1], rs, DATASET_FORMAT, filter=False)
    
    # Descartando duplicadas
    initial_len = df_day_1.shape[0]
    df_day_1 = df_day_1.drop_duplicates()
    # print(f'Tamanho inicial: {initial_len}, tamanho final {df_day_1.shape[0]} | Descartadas {initial_len - df_day_1.shape[0]} duplicadas')
    # print(df_day_1['Label'].value_counts())
    # print()
    
    
    df_day_2 = GetDataset(sys.argv[2], rs, DATASET_FORMAT, filter=False)
    
    # Descartando duplicadas
    initial_len = df_day_2.shape[0]
    df_day_2 = df_day_2.drop_duplicates()
    # print(f'Tamanho inicial: {initial_len}, tamanho final {df_day_2.shape[0]} | Descartadas {initial_len - df_day_2.shape[0]} duplicadas')
    # print(df_day_2['Label'].value_counts())
    # print()
    # print()
    
    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    # print()
    # print(f"Tamanho do conjunto de treino: {df_train.__len__()} benignos")
    #print(df_train['Label'].value_counts())
    
    # print()
    # print(f'Tamanho do conjunto de validação: {df_val[df_val["Label"] == BENIGN].__len__()} benignos, {df_val[df_val["Label"] != BENIGN].__len__()} ataque')
    #print(df_val['Label'].value_counts())
    
    # print()
    # print(f'Tamanho do conjunto de teste: {df_test[df_test["Label"] == BENIGN].__len__()} benignos, {df_test[df_test["Label"] != BENIGN].__len__()} ataque')
    # print(df_test['Label'].value_counts())
    
    
    # Limpar dados
    # Essas colunas geram dados de string ou não normalizáveis
    # TODO: ver features consideradas importantes no artigo
    df_train = df_train.drop(["Label", "Unnamed: 0", 'Flow Bytes/s', 'Flow Packets/s', 'Bwd PSH Flags',
                              'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count', 'PSH Flag Count', 'ECE Flag Count',
                              'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
                              'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'], axis=1)
    
    # Essa coluna é importante para a dependência temporal!
    df_train['Timestamp'] = df_train['Timestamp'].astype('int64')
    df_train = df_train.astype('float64')
    
    # Normalização 
    df_train = (df_train - df_train.min())/(df_train.max() - df_train.min())
    
    # print(df_train.loc[0])
    Train(df_train, 0.00005, 100)
    
    
    

if __name__ == '__main__':
    main()