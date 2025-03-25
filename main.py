from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.wgan.wgan import discriminate, Discriminator, Generator, cuda
from src.wgan.lstm_wgan import TrainLSTM
from src.wgan.linear_wgan import TrainLinear
from src.wgan.self_attention_wgan import TrainSelfAttention
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
from optuna.pruners import MedianPruner


def DescartarDuplicatas(dataset: pd.DataFrame, do_print=False):
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

    # Carrega os datasets (divididos entre dia 1 e dia 2)
    df_day_1 = GetDataset(sys.argv[1], rs, DATASET_FORMAT, filter=False, do_print=False)
    df_day_1 = DescartarDuplicatas(df_day_1, do_print=False)

    df_day_2 = GetDataset(sys.argv[2], rs, DATASET_FORMAT, filter=False, do_print=False)
    df_day_2 = DescartarDuplicatas(df_day_2, do_print=False)

    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    # DebugTrainValTest(df_train, df_val, df_test, BENIGN)

    # Ordena os dados pelo Timestamp para preservar a dependência temporal
    df_train = df_train.sort_values(by="Timestamp", ignore_index=True)
    df_val = df_val.sort_values(by="Timestamp", ignore_index=True)
    df_test = df_test.sort_values(by="Timestamp", ignore_index=True)

    df_train_label = df_train["Label"]
    df_val_label = df_val["Label"]
    df_test_label = df_test["Label"]

    # Remove as colunas que não serão usadas para treinamento
    df_train = df_train.drop(["Label"], axis=1)
    df_val = df_val.drop(["Label"], axis=1)
    df_test = df_test.drop(["Label"], axis=1)
    df_train = df_train.drop(["Timestamp"], axis=1)
    df_val = df_val.drop(["Timestamp"], axis=1)
    df_test = df_test.drop(["Timestamp"], axis=1)

    # Mapeia endereços IP para inteiros
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
    df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
    df_train = df_train.fillna(0)
    df_val = (df_val - train_min) / (min_max)
    df_val = df_val.fillna(0)
    df_test = (df_test - train_min) / (min_max)
    df_test = df_test.fillna(0)

    # Converte os rótulos para 0 (BENIGN) e 1 (ataque)
    y_val = df_val_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == 'BENIGN' else 1)

    if len(sys.argv) > 3 and sys.argv[3] == "train":
        if sys.argv[4] == "linear":
            generator, discriminator = TrainLinear(
                df_train, 2e-5, 3e-5, 10, df_val, y_val,
                n_critic=3, optim=torch_optimizer.Yogi,
                wdd=2e-2, wdg=2e-2,
                early_stopping=EarlyStopping(3, 0), batch_size=100
            )
            torch.save(generator, "GeneratorLinear.torch")
            torch.save(discriminator, "DiscriminatorLinear.torch")
        elif sys.argv[4] == "sa":
            generator_sa, discriminator_sa = TrainSelfAttention(
                df_train, 5e-5, 5e-5, 10, df_val, y_val,
                wdd=6e-4, wdg=9e-4, clip_value=0.9, optim=torch_optimizer.Yogi,
                early_stopping=EarlyStopping(3, 0), latent_dim=10,
                batch_size=128, n_critic=5, time_window=40,
                headsd=40, embedd=240, headsg=40, embedg=240
            )
            torch.save(generator_sa, "Generator.torch")
            torch.save(discriminator_sa, "Discriminator.torch")
        elif sys.argv[4] == "lstm":
            generator, discriminator = TrainLSTM(
                df_train, 0.00010870300025198446, 0.00028247627584454017, 10, df_val, y_val,
                wdd=2e-2, wdg=1e-2, optim=torch_optimizer.Yogi,
                early_stopping=EarlyStopping(15, 0), latent_dim=10,
                batch_size=128, n_critic=3, time_window=80,
                internal_d=512, internal_g=512, clip_value=0.1
            )
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

        discriminator_sa: Discriminator = torch.load("Discriminator.torch", weights_only=False).to(device)
        generator_sa: Generator = torch.load("Generator.torch", weights_only=False).to(device)
        discriminator_sa.eval()
        generator_sa.eval()

        if len(sys.argv) == 4 or sys.argv[4] == "look":
            from src.into_dataloader import IntoDataset
            dataset_test = IntoDataset(df_x, time_window=40)
            scores = discriminate(discriminator_sa, dataset_test, time_window=40, batch_size=400)
            # Para cada amostra, agrega os scores (por exemplo, média)
            for i, row in df_x.iterrows():
                # Calcula a média dos scores para a amostra i
                sample_score = np.mean(scores[i])
                print(df_x_label.loc[i], sample_score)
        elif sys.argv[4] == "thresh":
            from src.into_dataloader import IntoDataset
            dataset_test = IntoDataset(df_x, time_window=40)
            scores = discriminate(discriminator_sa, dataset_test, time_window=40, batch_size=400)
            preds = np.array(scores).reshape(len(scores), -1).mean(axis=1)
            best_thresh = metrics.best_validation_threshold(y_x, preds)
            thresh = best_thresh["thresholds"]
            if len(sys.argv) == 5 or sys.argv[5] in ("metrics", "both"):
                X = "Validation" if sys.argv[3] == "val" else "Test"
                print(f"{X} AUC: ", metrics.roc_auc_score(y_x, preds))
                print(f"{X} accuracy: ", metrics.accuracy(y_x, preds > thresh))
                print(f"{X} precision: ", metrics.precision_score(y_x, preds > thresh))
                print(f"{X} recall: ", metrics.recall_score(y_x, preds > thresh))
                print(f"{X} f1: ", metrics.f1_score(y_x, preds > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
            if len(sys.argv) > 5:
                if sys.argv[5] in ("matrix", "both"):
                    metrics.plot_confusion_matrix(y_x, preds > thresh, name=sys.argv[3])
                if sys.argv[5] in ("curve", "both"):
                    metrics.plot_roc_curve(y_x, preds, name=sys.argv[3])
    elif len(sys.argv) > 3 and sys.argv[3] == "minmax":
        df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
        df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
        print(df_train.min())
        print()
        print(df_train.max())
    elif len(sys.argv) > 3 and sys.argv[3] == "optuna":
        import optuna
        model_type = sys.argv[4] if len(sys.argv) > 4 else "lstm"
        if model_type == "lstm":
            def objective(trial):
                lrg = trial.suggest_loguniform("lrg", 1e-4, 1e-3)
                lrd = trial.suggest_loguniform("lrd", 1e-4, 1e-3)
                n_critic = trial.suggest_categorical("n_critic", [3, 4, 5, 7])
                clip_value = trial.suggest_categorical("clip_value", [0.1, 0.5, 1.0])
                latent_dim = trial.suggest_categorical("latent_dim", [10, 20, 30])
                internal_dim = trial.suggest_categorical("internal_dim", [240, 400, 512])
                dropout = trial.suggest_uniform("dropout", 0.1, 0.3)
                batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
                epochs = 60
                generator, discriminator = TrainLSTM(
                    df_train=df_train,
                    lrd=lrd, lrg=lrg, epochs=epochs,
                    df_val=df_val, y_val=y_val,
                    n_critic=n_critic, clip_value=clip_value,
                    latent_dim=latent_dim,
                    optim=torch_optimizer.Yogi,
                    wdd=1e-2, wdg=1e-2,
                    early_stopping=EarlyStopping(patience=5, verbose=False),
                    dropout=dropout,
                    time_window=80,
                    batch_size=batch_size,
                    internal_d=internal_dim,
                    internal_g=internal_dim,
                    trial=trial
                )
                from src.into_dataloader import IntoDataset
                dataset_val = IntoDataset(df_val, time_window=80)
                preds = discriminate(discriminator, dataset_val, time_window=80, batch_size=400)
                preds = np.mean(np.array(preds), axis=1)
                auc = metrics.roc_auc_score(y_val, preds)
                print(f"Trial AUC: {auc}")
                return auc
            study = optuna.create_study(
                direction="maximize",
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            )
            study.optimize(objective, n_trials=50)
            print("Melhores hiperparâmetros:", study.best_trial.params)
            print("Melhor AUC:", study.best_trial.value)
        else:
            print("A otimização para o modelo", model_type, "ainda não foi implementada.")


if __name__ == '__main__':
    main()
