from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan.TCN_wgan import Train, Discriminator, Generator, cuda
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
import optuna
from src.early_stop import EarlyStopping

# Função para criar janelas temporais dos dados.
def create_windows(df, window_size):
    data = df.to_numpy()  # shape: (N, F)
    windows = []
    for i in range(len(data) - window_size + 1):
        # Cada janela é uma sequência de 'window_size' instantes consecutivos
        window = data[i : i + window_size]  # shape: (window_size, F)
        windows.append(window)
    return np.array(windows)  # shape: (num_windows, window_size, F)

# Função para criar janelas de rótulos.
def create_label_windows(labels, window_size):
    labels = np.array(labels)
    window_labels = []
    for i in range(len(labels) - window_size + 1):
        window = labels[i : i + window_size]
        # Rotula a janela como 1 se houver ao menos um ataque; caso contrário, 0.
        win_label = 1 if np.any(window == 1) else 0
        window_labels.append(win_label)
    return np.array(window_labels)

# Função de avaliação (discriminate) que gera uma predição para cada janela.
def discriminate(discriminator, df, window_size=50):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    windows = create_windows(df, window_size)
    results = []
    for i in range(len(windows)):
        # Converte cada janela de (window_size, F) para (1, F, window_size)
        window = torch.from_numpy(windows[i]).float().permute(1, 0).unsqueeze(0)
        window = window.type(Tensor)
        output = discriminator(window)
        results.append(output.item())
    return results

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
    y_val  = df_val_label.apply(lambda c: 0 if c == BENIGN else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == BENIGN else 1)
    
    # Seleção do modo: train, val, test ou minmax
    mode = sys.argv[3] if len(sys.argv) > 3 else "train"
    
    # Defina o tamanho da janela para o processamento temporal
    window_size = 50
    
    if mode == "train":
        # --- Tunagem de hiperparâmetros com Optuna e EarlyStopping ---
        def objective(trial):
            # Sugere hiperparâmetros utilizando a nova API do Optuna
            lrd = trial.suggest_float("lrd", 1e-6, 1e-3, log=True)
            lrg = trial.suggest_float("lrg", 1e-6, 1e-3, log=True)
            latent_dim = trial.suggest_int("latent_dim", 10, 100)
            n_critic = trial.suggest_int("n_critic", 1, 10)
            clip_value = trial.suggest_float("clip_value", 0.001, 0.1)
            wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
            
            # Instancia o early stopping (por exemplo, com paciência de 3 épocas)
            stopper = EarlyStopping(patience=3, delta=0)
            
            # Treina por um número reduzido de épocas para o tuning (ex.: 10)
            generator, discriminator = Train(
                df_train, lrd, lrg, epochs=10,
                df_val=df_val, y_val=y_val, wd=wd,
                optim=torch_optimizer.Yogi, 
                latent_dim=latent_dim,
                n_critic=n_critic, clip_value=clip_value,
                window_size=window_size
            )
            # Gera as predições do discriminador a partir do conjunto de validação
            preds = discriminate(discriminator, df_val, window_size)
            # Converte os rótulos originais de validação em rótulos para janelas
            y_val_windows = create_label_windows(y_val, window_size)
            auc_val = metrics.roc_auc_score(y_val_windows, preds)
            return auc_val

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        best_params = study.best_params
        print("Melhores hiperparâmetros:", best_params)
        print("Melhor AUC:", study.best_value)
        
        # --- Treinamento final com os melhores hiperparâmetros e EarlyStopping ---
        final_stopper = EarlyStopping(patience=5, delta=0)
        generator, discriminator = Train(
            df_train,
            best_params["lrd"],
            best_params["lrg"],
            epochs=20,
            df_val=df_val,
            y_val=y_val,
            wd=best_params["wd"],
            optim=torch_optimizer.Yogi,
            latent_dim=best_params["latent_dim"],
            n_critic=best_params["n_critic"],
            clip_value=best_params["clip_value"],
            window_size=window_size
        )
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
            preds = discriminate(discriminator, df_val, window_size)
            y_val_windows = create_label_windows(y_val, window_size)
            for i, _ in enumerate(y_val_windows):
                label = y_val_windows[i]
                result = preds[i]
                print(label, result)
        elif sys.argv[4] == "thresh":
            preds = discriminate(discriminator, df_val, window_size)
            y_val_windows = create_label_windows(y_val, window_size)
            best_thresh = metrics.best_validation_threshold(y_val_windows, preds)
            thresh = best_thresh["thresholds"]
            if len(sys.argv) == 5 or sys.argv[5] == "metrics":
                print("Validation accuracy: ", metrics.accuracy(y_val_windows, np.array(preds) > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
            elif sys.argv[5] == "matrix":
                metrics.plot_confusion_matrix(y_val_windows, np.array(preds) > thresh)
            elif sys.argv[5] == "curve":
                metrics.plot_roc_curve(y_val_windows, preds)
    
    elif mode == "test":
        # Teste: semelhante à validação, mas utilizando o conjunto de teste
        discriminator: Discriminator = torch.load("Discriminator_TCN.torch", weights_only=False)
        generator: Generator = torch.load("Generator_TCN.torch", weights_only=False)
        discriminator.eval()
        generator.eval()
        
        if len(sys.argv) == 4 or sys.argv[4] == "look":
            preds = discriminate(discriminator, df_test, window_size)
            y_test_windows = create_label_windows(y_test, window_size)
            for i, _ in enumerate(y_test_windows):
                label = y_test_windows[i]
                result = preds[i]
                print(label, result)
        elif sys.argv[4] == "thresh":
            preds = discriminate(discriminator, df_test, window_size)
            y_test_windows = create_label_windows(y_test, window_size)
            best_thresh = metrics.best_validation_threshold(y_test_windows, preds)
            thresh = best_thresh["thresholds"]
            if len(sys.argv) == 5 or sys.argv[5] == "metrics":
                print("Test accuracy: ", metrics.accuracy(y_test_windows, np.array(preds) > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
            elif sys.argv[5] == "matrix":
                metrics.plot_confusion_matrix(y_test_windows, np.array(preds) > thresh)
            elif sys.argv[5] == "curve":
                metrics.plot_roc_curve(y_test_windows, preds)
    
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
