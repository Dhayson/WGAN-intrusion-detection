from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan.TCN_wgan import Train, discriminate, Discriminator, Generator, cuda
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

def DescartarDuplicatas(dataset: pd.DataFrame, do_print=False):
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
    
    # Verifica se os paths dos datasets foram informados
    if len(sys.argv) < 3:
        print("Uso: python script.py <path_dataset_day1> <path_dataset_day2> [train|val|test|minmax] ...")
        sys.exit(1)
    
    dataset_path_day1 = sys.argv[1]
    dataset_path_day2 = sys.argv[2]
    
    # Carrega os datasets
    df_day_1 = GetDataset(dataset_path_day1, rs, DATASET_FORMAT, filter=False, do_print=False)
    df_day_1 = DescartarDuplicatas(df_day_1, do_print=False)
    
    df_day_2 = GetDataset(dataset_path_day2, rs, DATASET_FORMAT, filter=False, do_print=False)
    df_day_2 = DescartarDuplicatas(df_day_2, do_print=False)
    
    # Divide em treino, validação e teste
    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    # Ordena os conjuntos pela coluna "Timestamp" para manter a dependência temporal
    df_train = df_train.sort_values(by="Timestamp")
    df_val = df_val.sort_values(by="Timestamp", ignore_index=True)
    df_test = df_test.sort_values(by="Timestamp", ignore_index=True)
    
    # Guarda os rótulos originais
    df_train_label = df_train["Label"]
    df_val_label   = df_val["Label"]
    df_test_label  = df_test["Label"]
    
    # Remove colunas não utilizadas para treinamento
    df_train = df_train.drop(["Label", "Timestamp"], axis=1)
    df_val   = df_val.drop(["Label", "Timestamp"], axis=1)
    df_test  = df_test.drop(["Label", "Timestamp"], axis=1)
    
    # Converte endereços IP para inteiros
    for col in ["Source IP", "Destination IP"]:
        df_train[col] = df_train[col].map(lambda x: int(IPv4Address(x)))
        df_val[col]   = df_val[col].map(lambda x: int(IPv4Address(x)))
        df_test[col]  = df_test[col].map(lambda x: int(IPv4Address(x)))
    
    Tensor: type[torch.FloatTensor] = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Calcula os valores mínimos e máximos do conjunto de treino para normalização
    train_min = df_train.min().astype("float64")
    train_max = df_train.max().astype("float64")
    min_max = (train_max - train_min).to_numpy()
    
    # Normaliza os dados do conjunto de treino
    df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
    df_train = df_train.fillna(0)
    # Normaliza os conjuntos de validação e teste com base no treino
    df_val  = (df_val - train_min) / (min_max)
    df_val  = df_val.fillna(0)
    df_test = (df_test - train_min) / (min_max)
    df_test = df_test.fillna(0)
    
    # Cria rótulos binários: BENIGN -> 0, ataque -> 1
    y_val  = df_val_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    
    # Seleção do modo: train, val, test ou minmax
    mode = sys.argv[3] if len(sys.argv) > 3 else "train"
    
    if mode == "train":
        # Treinamento: chama a função Train definida em TCN_wgan.py
        generator, discriminator = Train(df_train, 2e-5, 3e-5, epochs=5, df_val=df_val, y_val=y_val, wd=2e-2, optim=torch_optimizer.Yogi)
        torch.save(generator, "Generator_TCN.torch")
        torch.save(discriminator, "Discriminator_TCN.torch")
        print("Treinamento concluído e modelos salvos!")
    
    elif mode == "val":
        # Validação: carrega os modelos e avalia no conjunto de validação
        discriminator: Discriminator = torch.load("Discriminator_TCN.torch", weights_only=False)
        generator: Generator = torch.load("Generator_TCN.torch", weights_only=False)
        discriminator.eval()
        generator.eval()
        
        if len(sys.argv) == 4 or sys.argv[4] == "look":
            preds = discriminate(discriminator, df_val)
            for i, _ in df_val.iterrows():
                label = df_val_label.loc[i]
                result = preds[i]
                # A condição abaixo (random.randint(0,1)==-1) nunca será verdadeira; mantida apenas para ilustração.
                if random.randint(0,1) == -1:
                    z = Variable(Tensor(np.random.normal(0, 1, (30,))))
                    gen = generator(z).detach()
                    result_fake = discriminator(gen)
                    print("FAKE", result_fake.item())
                else:
                    print(label, result)
        elif sys.argv[4] == "thresh":
            preds = discriminate(discriminator, df_val)
            best_thresh = metrics.best_validation_threshold(y_val, preds)
            thresh = best_thresh["thresholds"]
            if len(sys.argv) == 5 or sys.argv[5] == "metrics":
                print("Validation accuracy: ", metrics.accuracy(y_val, preds > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
            elif sys.argv[5] == "matrix":
                metrics.plot_confusion_matrix(y_val, preds > thresh)
            elif sys.argv[5] == "curve":
                metrics.plot_roc_curve(y_val, preds)
    
    elif mode == "test":
        # Teste: semelhante à validação, mas utilizando o conjunto de teste
        discriminator: Discriminator = torch.load("Discriminator_TCN.torch", weights_only=False)
        generator: Generator = torch.load("Generator_TCN.torch", weights_only=False)
        discriminator.eval()
        generator.eval()
        
        if len(sys.argv) == 4 or sys.argv[4] == "look":
            preds = discriminate(discriminator, df_test)
            for i, _ in df_test.iterrows():
                label = df_test_label.loc[i]
                result = preds[i]
                print(label, result)
        elif sys.argv[4] == "thresh":
            preds = discriminate(discriminator, df_test)
            best_thresh = metrics.best_validation_threshold(y_test, preds)
            thresh = best_thresh["thresholds"]
            if len(sys.argv) == 5 or sys.argv[5] == "metrics":
                print("Test accuracy: ", metrics.accuracy(y_test, preds > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
            elif sys.argv[5] == "matrix":
                metrics.plot_confusion_matrix(y_test, preds > thresh)
            elif sys.argv[5] == "curve":
                metrics.plot_roc_curve(y_test, preds)
    
    elif mode == "minmax":
        # Apenas imprime os valores mínimos e máximos do conjunto de treino
        for col in ["Source IP", "Destination IP"]:
            df_train[col] = df_train[col].map(lambda x: int(IPv4Address(x)))
        print(df_train.min())
        print()
        print(df_train.max())
    
    else:
        print("Modo inválido. Use 'train', 'val', 'test' ou 'minmax'.")

if __name__ == '__main__':
    main()
