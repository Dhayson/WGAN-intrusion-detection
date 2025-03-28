from src.import_dataset import GetDataset
from src.import_dataset_alt import GetDataset2017
from src.dataset_split import SplitDataset
from src.wgan.linear_wgan import TrainLinear
from src.wgan.lstm_wgan import TrainLSTM
from src.wgan.self_attention_wgan import RunModelSelfAttention2019, RunModelSelfAttention2017
from src.tuning import TuneSA
from src.wgan.wgan import discriminate, Discriminator, Generator, cuda
import src.metrics as metrics
import sys
import numpy as np
import pandas as pd
from numpy.random import RandomState
import torch
import torch_optimizer
from ipaddress import IPv4Address
import random
from src.early_stop import EarlyStopping
from src.into_dataloader import IntoDataset, IntoDatasetNoTime
from src.transform import MeanNormalizeTensor, MinMaxNormalizeTensor

def DescartarDuplicatas(dataset: pd.DataFrame, do_print = False):
    initial_len = dataset.shape[0]
    dataset = dataset.drop_duplicates()
    if do_print:
        print(f'Tamanho inicial: {initial_len}, tamanho final {dataset.shape[0]} | Descartadas {initial_len - dataset.shape[0]} duplicadas')
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
    
    args= sys.argv
    dataset_kind = args[1]
    
    if args[1] == "2019":
        # Nesse caso já está dividido entre treino e teste, isto é, entre o primeiro e o segundo dia
        path1 = "dataset_filtered/Day1"
        path2 = "dataset_filtered/Day2"

        df_day_1 = GetDataset(path1, rs, DATASET_FORMAT, filter=False, do_print=False)
        # Descartando duplicadas
        df_day_1 = DescartarDuplicatas(df_day_1, do_print=False)
        
        df_day_2 = GetDataset(path2, rs, DATASET_FORMAT, filter=False, do_print=False)
        # Descartando duplicadas
        df_day_2 = DescartarDuplicatas(df_day_2, do_print=False)
        
        df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    elif args[1] == "2017":
        path1 = "dataset_alternative_filtered/Day1"
        path2 = "dataset_alternative_filtered/Day3"
        path3 = "dataset_alternative_filtered/Day5"

        df_train = GetDataset2017(path1, rs, DATASET_FORMAT, filter=False, do_print=False)
        df_val = GetDataset2017(path2, rs, DATASET_FORMAT, filter=False, do_print=False)
        df_test = GetDataset2017(path3, rs, DATASET_FORMAT, filter=False, do_print=False)
        args.pop(0)
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
    # min_max = (train_max - train_min).to_numpy()
    
    # Normalização
    df_train = df_train.fillna(0)
    df_val = df_val.fillna(0)
    df_test = df_test.fillna(0)
    
    # normalization = MeanNormalizeTensor(df_train.mean().to_numpy(dtype=np.float32), df_train.std().to_numpy(dtype=np.float32))
    normalization = MinMaxNormalizeTensor(df_train.max().to_numpy(dtype=np.float32), df_train.min().to_numpy(dtype=np.float32))
        
    # Validação: diferenciar entre benignos (0) e ataques (1)
    y_val = df_val_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    


    if args[1] == "2019":
        time_window = 69
        dataset_train = IntoDataset(df_train, time_window, normalization)
        dataset_val = IntoDataset(df_val, time_window, normalization)
        dataset_test = IntoDataset(df_test, time_window, normalization)
    else:
        time_window = 80
        dataset_train = IntoDataset(df_train, time_window, normalization)
        dataset_val = IntoDataset(df_val, time_window, normalization)
        dataset_test = IntoDataset(df_test, time_window, normalization)
    
    
    
    if args[2] == "train":
        generator, discriminator = TrainLSTM(
            df_train, 0.00010870300025198446, 0.00028247627584454017, 10, df_val, y_val,
            wdd=2e-2, wdg=1e-2, optim=torch_optimizer.Yogi,
            early_stopping=EarlyStopping(15, 0), latent_dim=10,
            batch_size=128, n_critic=3, time_window=80,
            internal_d=512, internal_g=512, clip_value=0.1
        )
        torch.save(generator, "GeneratorLSTM.torch")
        torch.save(discriminator, "DiscriminatorLSTM.torch")
    elif args[2] == "test":
        
        dataset_x = dataset_test
        df_x_label: pd.Series = df_test_label
        y_x = y_test
        discriminator: Discriminator = torch.load("DiscriminatorLSTM.torch", weights_only = False, map_location=torch.device(device)).to(device)
        generator: Generator = torch.load("GeneratorLSTM.torch", weights_only = False, map_location=torch.device(device)).to(device)
        discriminator = discriminator.eval()
        generator = generator.eval()
        X = "Test"
        # Get predicitons of df_val
        if False:
            preds = discriminate(discriminator, dataset_x, 35, 1)
        else:
            preds = discriminate(discriminator, dataset_x, time_window)
        best_thresh = metrics.best_validation_threshold(y_x, preds)
        thresh = best_thresh["thresholds"]
    
        print(f"{X} AUC: ", metrics.roc_auc_score(y_x, preds))
        print(f"{X} accuracy: ", metrics.accuracy(y_x, preds > thresh))
        print(f"{X} precision: ", metrics.precision_score(y_x, preds > thresh))
        print(f"{X} recall: ", metrics.recall_score(y_x, preds > thresh))
        print(f"{X} f1: ", metrics.f1_score(y_x, preds > thresh))
        print("Tpr: ", best_thresh['tpr'])
        print("Fpr: ", best_thresh['fpr'])
        print()
        if dataset_kind == "2019":
            if args[4] == "val":
                lista_attacks = ["BENIGN", "LDAP", "MSSQL", "NetBIOS", "UDPLag", "UDP", "Syn"]
            elif args[4] == "test":
                lista_attacks = ["BENIGN", "LDAP", "MSSQL", "NetBIOS", "UDPLag", "UDP", "Syn", "Portmap"]
        elif dataset_kind == "2017":
            if args[4] == "val":
                lista_attacks = [
                    "DoS Hulk",
                    "DoS GoldenEye",
                    "DoS slowloris",
                    "DoS Slowhttptest",
                    "Heartbleed",
                ]
            elif args[4] == "test":
                lista_attacks = [
                    "Bot",
                    "PortScan",
                    "DDoS"
                ]
        for i in lista_attacks:
            idxs = df_x_label[df_x_label==i].index
            y_x_i = y_x.loc[idxs]
            preds_i = [preds[i] for i in idxs.tolist()]
            print(f"{X} accuracy of {i}: ", metrics.accuracy(y_x_i, preds_i > thresh))
                        
if __name__ == '__main__':
    main()