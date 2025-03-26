#!/usr/bin/env python3
import sys
import torch
import numpy as np
import pandas as pd
from ipaddress import IPv4Address
from numpy.random import RandomState
import time

# Importa as classes do módulo TCN_wgan
import TCN_wgan
from TCN_wgan import TCNDiscriminator, discriminate

# Injeta as classes necessárias no namespace __main__
import TCN_wgan
sys.modules['__main__'].TCNDiscriminator = TCN_wgan.TCNDiscriminator
sys.modules['__main__'].TCNBlock = TCN_wgan.TCNBlock
sys.modules['__main__'].TCNGenerator = TCN_wgan.TCNGenerator

# Define um Dataset que cria janelas temporais a partir de um DataFrame
class TimeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, time_window: int = 40):
        super().__init__()
        self.data = df.to_numpy().astype(np.float32)
        self.time_window = time_window
        self.num_samples = len(self.data) - time_window + 1
        if self.num_samples < 1:
            raise ValueError(f"DataFrame com {len(df)} linhas não comporta janela de {time_window} timesteps.")
        print(f"[TimeWindowDataset] data.shape: {self.data.shape}, type: {type(self.data)}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.time_window]
        return torch.tensor(seq, dtype=torch.float32)

# Função de inferência que gera scores do discriminador para cada janela
@torch.no_grad()
def my_discriminate(discriminator: torch.nn.Module, df: pd.DataFrame, time_window=40, batch_size=400, device="cpu"):
    dataset = TimeWindowDataset(df, time_window=time_window)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    start_inf = time.time()
    for batch in loader:
        batch = batch.to(device)
        out = discriminator(batch).cpu().numpy()
        scores.extend([-s for s in out])
    end_inf = time.time()
    total_time = end_inf - start_inf
    avg_time = total_time / len(dataset)
    print(f"Total de inferência: {total_time:.3f} s, Tempo médio por janela: {avg_time*1000:.3f} ms")
    return scores

# Importa funções do seu projeto
from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
import src.metrics as metrics

# Normaliza os dados: usa somente colunas numéricas
def normalize(df):
    df = df.copy()
    for col in ["Source IP", "Destination IP"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: int(IPv4Address(x)))
    df = df.select_dtypes(include=[np.number])
    df_min = df.min().astype("float64")
    df_max = df.max().astype("float64")
    df = (df - df_min) / (df_max - df_min)
    return df.fillna(0)

# Pré-processamento conforme sua main de exemplo
def preprocess_data(path1, path2, rs, DATASET_FORMAT="csv"):
    df1 = GetDataset(path1, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df2 = GetDataset(path2, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df_train, df_val, df_test = SplitDataset(df1, df2, rs, DATASET_FORMAT)
    df_train = df_train.sort_values("Timestamp", ignore_index=True)
    df_val   = df_val.sort_values("Timestamp", ignore_index=True)
    df_test  = df_test.sort_values("Timestamp", ignore_index=True)
    return df_train, df_val, df_test, df_train["Label"], df_val["Label"], df_test["Label"]

def main():
    if len(sys.argv) != 5:
        print("Uso: python curvaroc.py <path_day1> <path_day2> <mode: val/test> <model_type: tcn>")
        sys.exit(1)
    
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    mode = sys.argv[3]      # "val" ou "test"
    model_type = sys.argv[4]  # "tcn"
    
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    # Carrega e pré-processa os dados
    df_train, df_val, df_test, lbl_train, lbl_val, lbl_test = preprocess_data(path1, path2, rs, DATASET_FORMAT)
    
    # Para "val" ou "test", escolhe o DataFrame e os rótulos correspondentes
    if mode == "val":
        df_x = df_val.copy()
        orig_labels = lbl_val.copy()
    elif mode == "test":
        df_x = df_test.copy()
        orig_labels = lbl_test.copy()
    else:
        print("Modo inválido. Use 'val' ou 'test'.")
        sys.exit(1)
    
    # Normaliza df_x (selecionando somente as colunas numéricas)
    df_x = normalize(df_x)
    
    # Converte os rótulos para binário robustamente:
    # Removemos espaços, convertemos para maiúsculas e comparamos com "BENIGN"
    orig_labels = orig_labels.apply(lambda c: c.strip().upper() if isinstance(c, str) else c)
    unique_labels = orig_labels.unique()
    print("Rótulos originais (únicos):", unique_labels)
    
    y = orig_labels.iloc[40-1:].apply(lambda c: 0 if c == "BENIGN" else 1).to_numpy()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Carrega o discriminador salvo
    disc = torch.load("/teamspace/studios/this_studio/WGAN-intrusion-detection-lstm/WGAN-intrusion-detection-lstm/DiscriminatorTCN.torch", map_location=device).eval()
    
    # Gera os scores de inferência
    scores = my_discriminate(disc, df_x, time_window=40, batch_size=400, device=device)
    
    # Plota a ROC curve e salva em out_curve_<mode>.png
    metrics.plot_roc_curve(y, scores, name=mode)
    
    auc = metrics.roc_auc_score(y, scores)
    best_thresh = metrics.best_validation_threshold(y, scores)
    thresh = best_thresh["thresholds"]
    overall_acc = metrics.accuracy(y, np.array(scores) > thresh)
    
    print(f"{mode.upper()} - AUC: {auc:.4f}, Overall Accuracy: {overall_acc:.4f}")
    print("Threshold:", thresh)
    print(f"Curva ROC salva em out_curve_{mode}.png")
    
    # Acurácia por ataque
    pred_binary = np.array(scores) > thresh
    unique = np.unique(orig_labels.iloc[40-1:].to_numpy())
    print("\nAcurácia por ataque:")
    for lab in unique:
        # Converte para 0 ou 1
        expected = 0 if lab.strip().upper() == "BENIGN" else 1
        idx = (orig_labels.iloc[40-1:].to_numpy() == lab)
        if np.sum(idx) == 0:
            continue
        acc_lab = np.mean(pred_binary[idx] == expected)
        print(f"{lab.strip().upper()}: {acc_lab:.4f}")

if __name__ == "__main__":
    main()
