from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan.linear_wgan import TrainLinear
from src.wgan.self_attention_wgan import TrainSelfAttention
from src.wgan.wgan import discriminate, Discriminator, Generator, cuda
import src.metrics as metrics
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.random import RandomState
import torch
import torch_optimizer
from ipaddress import IPv4Address
from torch.autograd import Variable
import random
from src.early_stop import EarlyStopping

# OBS: o dataset completo não cabe no repositório, mas pode ser baixado em http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/

def DescartarDuplicatas(dataset: pd.DataFrame, do_print = False):
    initial_len = dataset.shape[0]
    dataset = dataset.drop_duplicates()
    if do_print:
        print(f'Tamanho inicial: {initial_len}, tamanho final {dataset.shape[0]} | Descartadas {initial_len - dataset.shape[0]} duplicadas')
        print()
    return dataset

def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    BENIGN = "BENIGN"
    
    # Nesse caso já está dividido entre treino e teste, isto é, entre o primeiro e o segundo dia
    df_day_1 = GetDataset(sys.argv[1], rs, DATASET_FORMAT, filter=False, do_print=False)
    # Descartando duplicadas
    df_day_1 = DescartarDuplicatas(df_day_1, do_print=False)
    
    df_day_2 = GetDataset(sys.argv[2], rs, DATASET_FORMAT, filter=False, do_print=False)
    # Descartando duplicadas
    df_day_2 = DescartarDuplicatas(df_day_2, do_print=False)
    
    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    
    # Essa coluna é importante para a dependência temporal!
    df_train = df_train.sort_values(by = "Timestamp", ignore_index=True)
    df_val = df_val.sort_values(by = "Timestamp", ignore_index=True)
    df_test = df_test.sort_values(by = "Timestamp", ignore_index=True)
    
    df_train_label = df_train["Label"]
    df_val_label = df_val["Label"]
    df_test_label = df_test["Label"]
    
    # Limpar dados
    # Essas colunas geram dados de string ou não normalizáveis
    df_train = df_train.drop(["Label"], axis=1)
    df_val = df_val.drop(["Label"], axis=1)
    df_test = df_test.drop(["Label"], axis=1)
    df_train = df_train.drop(["Timestamp"], axis=1)
    df_val = df_val.drop(["Timestamp"], axis=1)
    df_test = df_test.drop(["Timestamp"], axis=1)
    
    # Mapeando endereços ip para valores inteiros
    df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
    df_val["Source IP"] = df_val["Source IP"].map(lambda x: int(IPv4Address(x)))
    df_test["Source IP"] = df_test["Source IP"].map(lambda x: int(IPv4Address(x)))
    df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
    df_val["Destination IP"] = df_val["Destination IP"].map(lambda x: int(IPv4Address(x)))
    df_test["Destination IP"] = df_test["Destination IP"].map(lambda x: int(IPv4Address(x)))
    
    
    cuda = True if torch.cuda.is_available() else False
    device = "cuda" if cuda else "cpu"
    
    train_min = df_train.min().astype("float64")
    train_max = df_train.max().astype("float64")
    min_max = (train_max - train_min).to_numpy()
    
    # Normalização
    # TODO: fazer de uma forma que fique salvo para depois
    df_train = (df_train - df_train.min())/(df_train.max() - df_train.min())
    df_train = df_train.fillna(0)
    
    df_val: pd.DataFrame = (df_val - train_min)/(min_max)
    df_val = df_val.fillna(0)

    df_test: pd.DataFrame = (df_test - train_min)/(min_max)
    df_test = df_test.fillna(0)
    
    # Validação: diferenciar entre benignos (0) e ataques (1)
    y_val = df_val_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == 'BENIGN' else 1)

    # salvando csvs para posteridade
    df_train.to_csv("df_train.csv", index=False)
    df_val.to_csv("df_val.csv", index=False)
    df_test.to_csv("df_test.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    
    
    
if __name__ == '__main__':
    main()