from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan.linear_wgan import TrainLinear
from src.wgan.self_attention_wgan import TrainSelfAttention, RunModelSelfAttention
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

# OBS: o dataset completo não cabe no repositório, mas pode ser baixado em http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/

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
    # min_max = (train_max - train_min).to_numpy()
    
    # Normalização
    df_train = df_train.fillna(0)
    df_val = df_val.fillna(0)
    df_test = df_test.fillna(0)
    
    normalization = MeanNormalizeTensor(df_train.mean().to_numpy(dtype=np.float32), df_train.std().to_numpy(dtype=np.float32))
    # normalization = MinMaxNormalizeTensor(df_train.max().to_numpy(dtype=np.float32), df_train.min().to_numpy(dtype=np.float32))
        
    # Validação: diferenciar entre benignos (0) e ataques (1)
    y_val = df_val_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    
    time_window = 52
    dataset_train = IntoDataset(df_train, time_window, normalization)
    dataset_val = IntoDataset(df_val, time_window, normalization)
    dataset_test = IntoDataset(df_test, time_window, normalization)
    if len(sys.argv) > 3 and sys.argv[3] == "train":
        if sys.argv[4] == "linear":
            dataset_train = IntoDatasetNoTime(df_train)
            dataset_val = IntoDatasetNoTime(df_val)
            generator, discriminator = TrainLinear(df_train, 2e-5, 3e-5, 10, df_val, y_val,
                n_critic=3, optim=torch_optimizer.Yogi, wdd=2e-2, wdg=2e-2, early_stopping=EarlyStopping(3, 0), batch_size=100)
            torch.save(generator, "GeneratorLinear.torch")
            torch.save(discriminator, "DiscriminatorLinear.torch")
        elif sys.argv[4] == "sa":
            RunModelSelfAttention(dataset_train, dataset_val, y_val)
    elif len(sys.argv) > 3 and sys.argv[3] == "tune":
        if sys.argv[4] == "sa":
            if sys.argv[5] == "2layers":
                print("Using 2 self attention blocks")
                TuneSA(df_train, df_val, y_val, sa_layers=2)
            else:
                TuneSA(df_train, df_val, y_val)
    elif len(sys.argv) > 3 and (sys.argv[3] == "val" or sys.argv[3] == "test"):
        if sys.argv[3] == "val":
            dataset_x = dataset_val
            df_x_label: pd.Series = df_val_label
            y_x = y_val
        else:
            dataset_x = dataset_test
            df_x_label: pd.Series = df_test_label
            y_x = y_test
        discriminator_sa: Discriminator = torch.load("DiscriminatorSA.torch", weights_only = False).to(device)
        generator_sa: Generator = torch.load("GeneratorSA.torch", weights_only = False).to(device)
        discriminator_sa = discriminator_sa.eval()
        generator_sa = generator_sa.eval()
        if len(sys.argv) == 4 or sys.argv[4] == "look":
            preds = discriminate(discriminator_sa, dataset_x, 400)
            for i, val in dataset_x.iterrows():
                label = df_x_label.loc[i]
                result = preds[i]
                if random.randint(0,1) == -1:
                    # Sample noise as generator input
                    z = torch.tensor(np.random.normal(0, 1, (30,)))
                    gen = generator_sa(z).detach()
                    result_fake = discriminator_sa(gen)
                    print("FAKE", result_fake.item())
                    # print((gen*min_max)+train_min.to_numpy())
                else:
                    print(label, result.item())
                    # print(val_f_old)
        elif sys.argv[4] == "thresh":
            X = "Validation" if sys.argv[3] == "val" else "Test"
            # Get predicitons of df_val
            if False:
                preds = discriminate(discriminator_sa, dataset_x, 35, 1)
            else:
                preds = discriminate(discriminator_sa, dataset_x, time_window)
            best_thresh = metrics.best_validation_threshold(y_x, preds)
            thresh = best_thresh["thresholds"]
            if len(sys.argv) == 5 or sys.argv[5] == "metrics" or sys.argv[5] == "both":
                print(f"{X} AUC: ", metrics.roc_auc_score(y_x, preds))
                print(f"{X} accuracy: ", metrics.accuracy(y_x, preds > thresh))
                print(f"{X} precision: ", metrics.precision_score(y_x, preds > thresh))
                print(f"{X} recall: ", metrics.recall_score(y_x, preds > thresh))
                print(f"{X} f1: ", metrics.f1_score(y_x, preds > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
            if len(sys.argv) > 5:
                if sys.argv[5] == "matrix" or sys.argv[5] == "both":
                    metrics.plot_confusion_matrix(y_x, preds > thresh, name=sys.argv[3])
                if sys.argv[5] == "curve" or sys.argv[5] == "both":
                    metrics.plot_roc_curve(y_x, preds, name=sys.argv[3])
                if sys.argv[5] == "attacks":
                    for i in ["BENIGN", "LDAP", "MSSQL", "NetBIOS", "UDPLag", "UDP", "Syn", "Portmap"]:
                        idxs = df_x_label[df_x_label==i].index
                        y_x_i = y_x.loc[idxs]
                        preds_i = [preds[i] for i in idxs.tolist()]
                        print(f"{X} accuracy of {i}: ", metrics.accuracy(y_x_i, preds_i > thresh))
                        
    elif len(sys.argv) > 3 and sys.argv[3] == "minmax":
        # Mapeando endereços ip para valores inteiros
        df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
        df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
        
        print(df_train.min())
        print()
        print(df_train.max())
    
if __name__ == '__main__':
    main()