from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan import Train, Discriminator, Generator, cuda
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.random import RandomState
import torch
from ipaddress import IPv4Address
from torch.autograd import Variable
import random

# OBS: o dataset grande não cabe no repositório, mas pode ser baixado em http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/

def DescartarDuplicatas(dataset: pd.DataFrame, do_print = False):
    initial_len = dataset.shape[0]
    dataset = dataset.drop_duplicates()
    if do_print:
        print(f'Tamanho inicial: {initial_len}, tamanho final {dataset.shape[0]} | Descartadas {initial_len - dataset.shape[0]} duplicadas')
        print(dataset['Label'].value_counts())
        print()
    return dataset

def DebugTrainValTest(df_train, df_val, df_test, BENIGN):
    print()
    print(f"Tamanho do conjunto de treino: {df_train.__len__()} benignos")
    print(df_train['Label'].value_counts())
    
    print()
    print(f'Tamanho do conjunto de validação: {df_val[df_val["Label"] == BENIGN].__len__()} benignos, {df_val[df_val["Label"] != BENIGN].__len__()} ataque')
    print(df_val['Label'].value_counts())
    
    print()
    print(f'Tamanho do conjunto de teste: {df_test[df_test["Label"] == BENIGN].__len__()} benignos, {df_test[df_test["Label"] != BENIGN].__len__()} ataque')
    print(df_test['Label'].value_counts())

def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"
    
    # Nesse caso já está dividido entre treino e teste, isto é, entre o primeiro e o segundo dia
    df_day_1 = GetDataset(sys.argv[1], rs, DATASET_FORMAT, filter=False, do_print=False)
    # Descartando duplicadas
    df_day_1 = DescartarDuplicatas(df_day_1, do_print=False)
    
    df_day_2 = GetDataset(sys.argv[2], rs, DATASET_FORMAT, filter=False, do_print=False)
    # Descartando duplicadas
    df_day_2 = DescartarDuplicatas(df_day_2, do_print=False)
    
    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    # DebugTrainValTest(df_train, df_val, df_test, BENIGN)
    
    if len(sys.argv) > 3 and sys.argv[3] == "train":
        # Limpar dados
        # Essas colunas geram dados de string ou não normalizáveis
        # TODO: ver features consideradas importantes no artigo
        df_train = df_train.drop(["Label"], axis=1)
        
        # Essa coluna é importante para a dependência temporal!
        df_train.sort_values(by = "Timestamp")
        df_train = df_train.drop(["Timestamp"], axis=1)
        
        # Mapeando endereços ip para valores inteiros
        df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
        df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
        
        # Normalização
        # TODO: fazer de uma forma que fique salvo para depois
        df_train = (df_train - df_train.min())/(df_train.max() - df_train.min())
        df_train = df_train.fillna(0)
        
        generator, discriminator = Train(df_train, 0.001, 50)
        torch.save(generator, "Generator2.torch")
        torch.save(discriminator, "Discriminator2.torch")
        
    elif len(sys.argv) > 3 and sys.argv[3] == "val":
        df_train_label = df_train["Label"]
        df_train = df_train.drop(["Label"], axis=1)
        # Essa coluna é importante para a dependência temporal!
        df_train.sort_values(by = "Timestamp")
        df_train = df_train.drop(["Timestamp"], axis=1)
        # Mapeando endereços ip para valores inteiros
        df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
        df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
        
        Tensor: type[torch.FloatTensor] = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        train_min = df_train.min().astype("float64")
        train_max = df_train.max().astype("float64")
        min_max = (train_max - train_min).to_numpy()
        
        discriminator: Discriminator = torch.load("Discriminator2.torch", weights_only = False)
        generator: Generator = torch.load("Generator2.torch", weights_only = False)
        discriminator = discriminator.eval()
        generator = generator.eval()
        
        df_train["Label"] = df_train_label
        
        df_val = df_val.sample(frac = 1, random_state=rs)
        for i, val in df_val.iterrows():
            
            label = val["Label"]
            timestamp = val["Timestamp"]
            
            val0: pd.Series = val
            
            val_f = val0.drop(labels=["Label","Timestamp"])
            
            val_f["Source IP"] = int(IPv4Address(val_f["Source IP"]))
            val_f["Destination IP"] = int(IPv4Address(val_f["Destination IP"]))
            val_f = val_f.to_numpy().astype("float64")
            
            val_f = np.divide((val_f - train_min.to_numpy()), min_max, out=np.zeros_like((val_f - train_min.to_numpy()), dtype=float), where=min_max!= 0)
            
            val_f = Variable(torch.from_numpy(val_f).type(Tensor))
            
            if random.randint(0,1) == -1:
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (30,))))
                gen = generator(z).detach()
                result = discriminator(gen)
                print("FAKE", result, gen)
            else:
                result = discriminator(val_f)
                print(label, result, val_f)
            input()
            
    
    elif len(sys.argv) > 3 and sys.argv[3] == "minmax":
        # Mapeando endereços ip para valores inteiros
        df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
        df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
        
        print(df_train.min())
        print()
        print(df_train.max())
    
if __name__ == '__main__':
    main()