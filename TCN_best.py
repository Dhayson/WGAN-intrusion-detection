import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
import src.metrics as metrics
import pandas as pd
import torch_optimizer
from ipaddress import IPv4Address
import random

import sys

# Flag global para CUDA
cuda = torch.cuda.is_available()

#################################
# Definição dos módulos TCN
#################################
class CausalConv1d(nn.Module):
    """
    Convolução causal 1D: cada saída no tempo t utiliza somente dados do passado.
    Aplica padding manualmente apenas à esquerda.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        pad = (self.dilation * (self.kernel_size - 1), 0)
        x = F.pad(x, pad)
        return self.conv(x)

class TCNBlock(nn.Module):
    """
    Bloco TCN: inclui convolução causal dilatada, ReLU, BatchNorm, Dropout e conexão residual.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim, tcn_channels=32, kernel_size=3, num_tcn_layers=1):
        """
        data_shape: (T, F) onde T é o tamanho da janela (número de instantes) e F é o número de features.
        """
        super(Generator, self).__init__()
        self.T = data_shape[0]  # tamanho da janela
        self.F = data_shape[1]  # número de features
        self.tcn_channels = tcn_channels

        # Expande o vetor latente para uma sequência de tamanho T com tcn_channels canais
        self.fc = nn.Linear(latent_dim, tcn_channels * self.T)
        tcn_blocks = []
        for i in range(num_tcn_layers):
            tcn_blocks.append(
                TCNBlock(tcn_channels, tcn_channels, kernel_size=kernel_size, dilation=2**i, dropout=0.2)
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        self.out_conv = nn.Conv1d(tcn_channels, self.F, kernel_size=1)
    
    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.fc(z)  # (batch, tcn_channels * T)
        x = x.view(-1, self.tcn_channels, self.T)  # (batch, tcn_channels, T)
        x = self.tcn(x)
        x = self.out_conv(x)  # (batch, F, T)
        return x

class Discriminator(nn.Module):
    def __init__(self, data_shape, tcn_channels=32, kernel_size=3, num_tcn_layers=1):
        """
        data_shape: (T, F)
        """
        super(Discriminator, self).__init__()
        self.T = data_shape[0]
        self.F = data_shape[1]
        self.input_conv = nn.Conv1d(self.F, tcn_channels, kernel_size=1)
        tcn_blocks = []
        for i in range(num_tcn_layers):
            tcn_blocks.append(
                TCNBlock(tcn_channels, tcn_channels, kernel_size=kernel_size, dilation=2**i, dropout=0.2)
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        self.out_fc = nn.Linear(tcn_channels * self.T, 1)
    
    def forward(self, x):
        # x: (batch, F, T)
        x = self.input_conv(x)
        x = self.tcn(x)
        x = x.view(x.size(0), -1)
        validity = self.out_fc(x)
        return validity

#################################
# Funções auxiliares de pré-processamento
#################################
def create_windows(df, window_size):
    """
    Converte o DataFrame em um array 3D de janelas com shape (num_windows, window_size, num_features)
    """
    data = df.to_numpy()
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        windows.append(window)
    return np.array(windows)

def create_label_windows(labels, window_size):
    """
    Converte o vetor de rótulos (um por instante) em rótulos para janelas.
    Cada janela é considerada positiva (1) se houver ao menos um ataque nela.
    """
    labels = np.array(labels)
    window_labels = []
    for i in range(len(labels) - window_size + 1):
        window = labels[i : i + window_size]
        win_label = 1 if np.any(window == 1) else 0
        window_labels.append(win_label)
    return np.array(window_labels)

def discriminate(discriminator, df, window_size=50):
    """
    Calcula as predições do discriminador para cada janela gerada a partir do DataFrame.
    """
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    windows = create_windows(df, window_size)
    results = []
    for i in range(len(windows)):
        # Reorganiza cada janela de (window_size, F) para (1, F, window_size)
        window = torch.from_numpy(windows[i]).float().permute(1, 0).unsqueeze(0)
        window = window.type(Tensor)
        output = discriminator(window)
        results.append(output.item())
    return results

#################################
# Função de treinamento
#################################
def Train(df_train, lrd, lrg, epochs, window_size=50, batch_size=32, latent_dim=30, n_critic=5, clip_value=1, optim=torch.optim.RMSprop, wd=1e-2):
    num_features = df_train.shape[1]
    data_shape = (window_size, num_features)
    print_each_n = 3000

    generator = Generator(data_shape, latent_dim, tcn_channels=32, kernel_size=3, num_tcn_layers=1)
    discriminator = Discriminator(data_shape, tcn_channels=32, kernel_size=3, num_tcn_layers=1)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wd)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wd)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    windows = create_windows(df_train, window_size)
    num_windows = windows.shape[0]

    batches_done = 0
    for epoch in range(epochs):
        indices = np.random.permutation(num_windows)
        for i in range(0, num_windows, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_windows = windows[batch_indices]
            real_data = torch.from_numpy(batch_windows).float().permute(0, 2, 1)
            real_data = Variable(real_data).type(Tensor)
            
            optimizer_D.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (real_data.size(0), latent_dim))))
            fake_data = generator(z).detach()
            loss_D_true = torch.mean(discriminator(real_data))
            loss_D_fake = torch.mean(discriminator(fake_data))
            loss_D = -loss_D_true + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if batches_done % n_critic == 0:
                optimizer_G.zero_grad()
                gen_data = generator(z)
                loss_G = -torch.mean(discriminator(gen_data))
                loss_G.backward()
                optimizer_G.step()

                if batches_done % print_each_n == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D true loss: %f] [D fake loss: %f] [G loss: %f]" %
                          (epoch+1, epochs, batches_done % num_windows, num_windows,
                           loss_D_true.item(), loss_D_fake.item(), loss_G.item()))
            batches_done += 1
    return generator, discriminator

#################################
# Pipeline principal: Carregamento dos dados, treinamento e avaliação
#################################
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_dataset_day1> <path_dataset_day2>")
        sys.exit(1)
    dataset_path_day1 = sys.argv[1]
    dataset_path_day2 = sys.argv[2]

    rs = np.random.RandomState(5)
    DATASET_FORMAT = "csv"
    BENIGN = "BENIGN"

    # Carrega os datasets e remove duplicatas
    df_day_1 = GetDataset(dataset_path_day1, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df_day_2 = GetDataset(dataset_path_day2, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()

    # Divide em treino, validação e teste
    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    df_train = df_train.sort_values(by="Timestamp")
    df_val = df_val.sort_values(by="Timestamp", ignore_index=True)
    df_test = df_test.sort_values(by="Timestamp", ignore_index=True)

    # Guarda os rótulos originais
    y_val = df_val["Label"]
    y_test = df_test["Label"]

    # Remove colunas desnecessárias
    df_train = df_train.drop(["Label", "Timestamp"], axis=1)
    df_val = df_val.drop(["Label", "Timestamp"], axis=1)
    df_test = df_test.drop(["Label", "Timestamp"], axis=1)

    # Converte IPs para inteiros
    for col in ["Source IP", "Destination IP"]:
        df_train[col] = df_train[col].map(lambda x: int(IPv4Address(x)))
        df_val[col] = df_val[col].map(lambda x: int(IPv4Address(x)))
        df_test[col] = df_test[col].map(lambda x: int(IPv4Address(x)))

    # Normalização baseada no conjunto de treino
    train_min = df_train.min().astype("float64")
    train_max = df_train.max().astype("float64")
    min_max = (train_max - train_min).to_numpy()

    df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
    df_train = df_train.fillna(0)
    df_val = (df_val - train_min) / (min_max)
    df_val = df_val.fillna(0)
    df_test = (df_test - train_min) / (min_max)
    df_test = df_test.fillna(0)

    # Rótulos binários: BENIGN -> 0, ataque -> 1
    y_val = y_val.apply(lambda c: 0 if c == BENIGN else 1)
    y_test = y_test.apply(lambda c: 0 if c == BENIGN else 1)

    # Hiperparâmetros "melhores" (obtidos anteriormente)
    best_hyperparams = {
        "lrd": 9.901285708920077e-05,
        "lrg": 0.0004733011485494085,
        "latent_dim": 24,
        "n_critic": 7,
        "clip_value": 0.0023668487818747953,
        "wd": 0.0005823975459548492
    }
    window_size = 50
    epochs = 20
    batch_size = 32
    latent_dim = best_hyperparams["latent_dim"]
    n_critic = best_hyperparams["n_critic"]
    clip_value = best_hyperparams["clip_value"]
    lrd = best_hyperparams["lrd"]
    lrg = best_hyperparams["lrg"]
    wd = best_hyperparams["wd"]

    print("Training model with best hyperparameters...")
    generator, discriminator = Train(
        df_train, lrd, lrg, epochs,
        window_size=window_size, batch_size=batch_size,
        latent_dim=latent_dim, n_critic=n_critic, clip_value=clip_value,
        optim=torch_optimizer.Yogi, wd=wd
    )

    # Avaliação no conjunto de validação
    preds_val = discriminate(discriminator, df_val, window_size)
    y_val_windows = create_label_windows(y_val, window_size)
    auc_val = metrics.roc_auc_score(y_val_windows, preds_val)
    acc_val = metrics.accuracy(y_val_windows, np.array(preds_val) > 0)
    best_thresh_val = metrics.best_validation_threshold(y_val_windows, preds_val)
    print("\nValidation metrics:")
    print("AUC:", auc_val)
    print("Accuracy:", acc_val)
    print("TPR:", best_thresh_val['tpr'])
    print("FPR:", best_thresh_val['fpr'])
    
    # Avaliação no conjunto de teste
    preds_test = discriminate(discriminator, df_test, window_size)
    y_test_windows = create_label_windows(y_test, window_size)
    auc_test = metrics.roc_auc_score(y_test_windows, preds_test)
    acc_test = metrics.accuracy(y_test_windows, np.array(preds_test) > 0)
    best_thresh_test = metrics.best_validation_threshold(y_test_windows, preds_test)
    print("\nTest metrics:")
    print("AUC:", auc_test)
    print("Accuracy:", acc_test)
    print("TPR:", best_thresh_test['tpr'])
    print("FPR:", best_thresh_test['fpr'])

if __name__ == '__main__':
    main()
