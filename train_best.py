import sys
import numpy as np
import pandas as pd
import torch
import torch_optimizer
from ipaddress import IPv4Address

# Ajuste os imports conforme a organização do seu projeto
from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
from src.early_stop import EarlyStopping
from src.wgan.lstm_wgan import TrainLSTM
from src.into_dataloader import IntoDataset
from src.wgan.wgan import discriminate
import src.metrics as metrics


def main():
    """
    Uso:
        python train_best.py /caminho/para/Day1 /caminho/para/Day2
    """
    if len(sys.argv) < 3:
        print("Uso: python train_best.py <path_day1> <path_day2>")
        sys.exit(1)

    path_day1 = sys.argv[1]
    path_day2 = sys.argv[2]

    # Hiperparâmetros “ótimos” (exemplo)
    lrg = 0.00048323877551367015
    lrd = 0.0004268399129272652
    n_critic = 4
    clip_value = 0.5
    latent_dim = 20
    internal_dim = 400
    dropout = 0.15
    batch_size = 64
    epochs = 60  # Ajuste conforme desejar

    # Caso queira reter consistência com seu main.py,
    # crie um RandomState
    rs = np.random.RandomState(5)

    # 1) Carrega dados Day1 e Day2 (sem usar nome de parâmetro 'random_state')
    df_day_1 = GetDataset(path_day1, rs, "csv", filter=False, do_print=False)
    df_day_2 = GetDataset(path_day2, rs, "csv", filter=False, do_print=False)

    # 2) Divide em treino, val e teste
    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, "csv")

    # Ordena pelo Timestamp (para manter dependência temporal)
    df_train = df_train.sort_values(by="Timestamp", ignore_index=True)
    df_val = df_val.sort_values(by="Timestamp", ignore_index=True)
    df_test = df_test.sort_values(by="Timestamp", ignore_index=True)

    # Salva os rótulos antes de remover colunas
    df_train_label = df_train["Label"]
    df_val_label = df_val["Label"]
    df_test_label = df_test["Label"]

    # Remove colunas desnecessárias
    df_train = df_train.drop(["Label", "Timestamp"], axis=1)
    df_val = df_val.drop(["Label", "Timestamp"], axis=1)
    df_test = df_test.drop(["Label", "Timestamp"], axis=1)

    # Mapeia IP -> inteiro
    df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
    df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
    df_val["Source IP"] = df_val["Source IP"].map(lambda x: int(IPv4Address(x)))
    df_val["Destination IP"] = df_val["Destination IP"].map(lambda x: int(IPv4Address(x)))
    df_test["Source IP"] = df_test["Source IP"].map(lambda x: int(IPv4Address(x)))
    df_test["Destination IP"] = df_test["Destination IP"].map(lambda x: int(IPv4Address(x)))

    # Normalização
    train_min = df_train.min().astype("float64")
    train_max = df_train.max().astype("float64")
    min_max = (train_max - train_min).to_numpy()

    df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
    df_train = df_train.fillna(0)

    df_val = (df_val - train_min) / min_max
    df_val = df_val.fillna(0)

    df_test = (df_test - train_min) / min_max
    df_test = df_test.fillna(0)

    # Converte labels para 0 e 1
    y_val = df_val_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == 'BENIGN' else 1)

    # 3) Treina o modelo com os hiperparâmetros “ótimos”
    print("Treinando LSTM-WGAN com hiperparâmetros:")
    print(f"  lrg={lrg}, lrd={lrd}, n_critic={n_critic}, clip_value={clip_value}")
    print(f"  latent_dim={latent_dim}, internal_dim={internal_dim}, dropout={dropout}, batch_size={batch_size}, epochs={epochs}")

    generator, discriminator = TrainLSTM(
        df_train=df_train,
        lrd=lrd,
        lrg=lrg,
        epochs=epochs,
        df_val=df_val,
        y_val=y_val,
        n_critic=n_critic,
        clip_value=clip_value,
        latent_dim=latent_dim,
        optim=torch_optimizer.Yogi,  # ou RMSprop, etc
        wdd=1e-2,
        wdg=1e-2,
        early_stopping=EarlyStopping(patience=5, verbose=True),
        dropout=dropout,
        print_each_n=100,
        time_window=40,  # Ajuste se você usar outro time_window
        batch_size=batch_size,
        do_print=False,
        step_by_step=False,
        internal_d=internal_dim,
        internal_g=internal_dim
    )

    # 4) Avalia no conjunto de VALIDAÇÃO
    print("\n--- Avaliação no Conjunto de Validação ---")
    dataset_val = IntoDataset(df_val, time_window=40)
    scores_val = discriminate(discriminator, dataset_val, time_window=40, batch_size=400)
    preds_val = np.array(scores_val).reshape(len(scores_val), -1).mean(axis=1)
    best_thresh_val = metrics.best_validation_threshold(y_val, preds_val)
    thresh_val = best_thresh_val["thresholds"]

    auc_val = metrics.roc_auc_score(y_val, preds_val)
    acc_val = metrics.accuracy(y_val, preds_val > thresh_val)
    prec_val = metrics.precision_score(y_val, preds_val > thresh_val)
    rec_val = metrics.recall_score(y_val, preds_val > thresh_val)
    f1_val = metrics.f1_score(y_val, preds_val > thresh_val)

    print("Validation AUC:", auc_val)
    print("Validation Accuracy:", acc_val)
    print("Validation Precision:", prec_val)
    print("Validation Recall:", rec_val)
    print("Validation F1:", f1_val)

    # 5) Avalia no conjunto de TESTE
    print("\n--- Avaliação no Conjunto de Teste ---")
    dataset_test = IntoDataset(df_test, time_window=40)
    scores_test = discriminate(discriminator, dataset_test, time_window=40, batch_size=400)
    preds_test = np.array(scores_test).reshape(len(scores_test), -1).mean(axis=1)
    best_thresh_test = metrics.best_validation_threshold(y_test, preds_test)
    thresh_test = best_thresh_test["thresholds"]

    auc_test = metrics.roc_auc_score(y_test, preds_test)
    acc_test = metrics.accuracy(y_test, preds_test > thresh_test)
    prec_test = metrics.precision_score(y_test, preds_test > thresh_test)
    rec_test = metrics.recall_score(y_test, preds_test > thresh_test)
    f1_test = metrics.f1_score(y_test, preds_test > thresh_test)

    print("Test AUC:", auc_test)
    print("Test Accuracy:", acc_test)
    print("Test Precision:", prec_test)
    print("Test Recall:", rec_test)
    print("Test F1:", f1_test)


if __name__ == "__main__":
    main()
