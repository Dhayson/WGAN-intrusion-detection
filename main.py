from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan import Train, discriminate, Discriminator, Generator, cuda
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
    
    if len(sys.argv) > 3 and sys.argv[3] == "train":
        generator, discriminator = Train(df_train, 1e-2, 4e-3, 50, df_val, y_val, wdd=3e-2, wdg=3e-3, optim=torch_optimizer.Yogi,
            early_stopping=EarlyStopping(15, 0), latent_dim=10, batch_size=64)
        torch.save(generator, "Generator.torch")
        torch.save(discriminator, "Discriminator.torch")
        
    elif len(sys.argv) > 3 and (sys.argv[3] == "val" or sys.argv[3] == "test"):
        if sys.argv[3] == "val":
            df_x = df_val
            df_x_label = df_val_label
            y_x = y_val
        else:
            df_x = df_test
            df_x_label = df_test_label
            y_x = y_test
        discriminator: Discriminator = torch.load("Discriminator.torch", weights_only = False).to(device)
        generator: Generator = torch.load("Generator.torch", weights_only = False).to(device)
        discriminator = discriminator.eval()
        generator = generator.eval()
        if len(sys.argv) == 4 or sys.argv[4] == "look":
            preds = discriminate(discriminator, df_x)
            for i, val in df_x.iterrows():
                label = df_x_label.loc[i]
                result = preds[i]
                if random.randint(0,1) == -1:
                    # Sample noise as generator input
                    z = torch.tensor(np.random.normal(0, 1, (30,)))
                    gen = generator(z).detach()
                    result_fake = discriminator(gen)
                    print("FAKE", result_fake.item())
                    # print((gen*min_max)+train_min.to_numpy())
                else:
                    print(label, result.item())
                    # print(val_f_old)
        elif sys.argv[4] == "thresh":
            # Get predicitons of df_val
            preds = discriminate(discriminator, df_x)
            best_thresh = metrics.best_validation_threshold(y_x, preds)
            thresh = best_thresh["thresholds"]
            if len(sys.argv) == 5 or sys.argv[5] == "metrics" or sys.argv[5] == "both":
                X = "Validation" if sys.argv[3] == "val" else "Test"
                print(f"{X} accuracy: ", metrics.accuracy(y_x, preds > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
            if len(sys.argv) > 5:
                if sys.argv[5] == "matrix" or sys.argv[5] == "both":
                    metrics.plot_confusion_matrix(y_x, preds > thresh, name=sys.argv[3])
                if sys.argv[5] == "curve" or sys.argv[5] == "both":
                    metrics.plot_roc_curve(y_x, preds, name=sys.argv[3])
    elif len(sys.argv) > 3 and sys.argv[3] == "minmax":
        # Mapeando endereços ip para valores inteiros
        df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
        df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
        
        print(df_train.min())
        print()
        print(df_train.max())
    
if __name__ == '__main__':
    main()