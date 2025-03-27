  #!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import pandas as pd
import optuna
from ipaddress import IPv4Address
from numpy.random import RandomState
import time
import random
import torch_optimizer
import src.metrics as metrics
from src.early_stop import EarlyStopping

# ============================================================
# Função para definir a seed e garantir replicabilidade
def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Função GetDataset adaptada para CSVs (mantendo Timestamp se existir)
def GetDataset(path: str, rs: RandomState, dataset_format="csv", filter=False, do_print=True):
    """
    Lê todos os arquivos CSV de um diretório e retorna um DataFrame contendo todas as linhas.
    Se existir a coluna "Timestamp", ela é mantida para cálculos de tempo.
    """
    df_list = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if dataset_format == "csv":
            if do_print:
                print(f"Reading {full_path}")
            df_aux = pd.read_csv(full_path)
        elif dataset_format == "parquet":
            df_aux = pd.read_parquet(full_path)
        df_aux.columns = df_aux.columns.str.strip()
        df_aux = df_aux.replace([-np.inf, np.inf], np.nan)
        df_aux = df_aux.dropna(axis=0)
        df_list.append(df_aux)
    df = pd.concat(df_list)
    return df

# ============================================================
# Dataset: cada amostra é uma janela temporal
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
        seq = self.data[idx: idx+self.time_window]
        return torch.tensor(seq, dtype=torch.float32)

# ============================================================
# Bloco TCN, Gerador e Discriminador
class TCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        # Ajuste do padding para manter shape
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.norm(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNGenerator(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, num_channels, num_layers, kernel_size=3, dropout=0.2):
        super(TCNGenerator, self).__init__()
        self.input_linear = torch.nn.Linear(latent_dim, num_channels)
        self.tcn_blocks = torch.nn.ModuleList([
            TCNBlock(num_channels, num_channels, kernel_size=kernel_size,
                     dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_linear = torch.nn.Linear(num_channels, output_dim)
    
    def forward(self, z):
        # z: (batch, T, latent_dim)
        x = self.input_linear(z)  # (batch, T, num_channels)
        x = x.transpose(1, 2)     # (batch, num_channels, T)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)     # (batch, T, num_channels)
        out = self.output_linear(x)  # (batch, T, output_dim)
        return out

class TCNDiscriminator(torch.nn.Module):
    def __init__(self, input_dim, num_channels, num_layers, kernel_size=3, dropout=0.2):
        super(TCNDiscriminator, self).__init__()
        self.input_linear = torch.nn.Linear(input_dim, num_channels)
        self.tcn_blocks = torch.nn.ModuleList([
            TCNBlock(num_channels, num_channels, kernel_size=kernel_size,
                     dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_linear = torch.nn.Linear(num_channels, 1)
    
    def forward(self, x):
        # x: (batch, F, T)
        x = self.input_linear(x.transpose(1,2))
        x = x.transpose(1,2)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.mean(dim=2)
        out = self.output_linear(x)
        return out.squeeze(1)

# ============================================================
# Função de treinamento para TCN-based WGAN com Optuna opcional
def TrainTCN(df_train: pd.DataFrame,
             latent_dim, output_dim, input_dim,
             lrd, lrg, epochs,
             dataset_val: pd.DataFrame = None,
             y_val=None,
             n_critic=5, clip_value=1,
             optim=torch.optim.RMSprop,
             wdd=1e-2, wdg=1e-2,
             early_stopping=None,
             dropout=0.2,
             print_each_n=1000,
             time_window=40,
             batch_size=5,
             num_channels=32,
             num_layers=1,
             do_print=False,
             step_by_step=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_train = TimeWindowDataset(df_train, time_window=time_window)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    generator = TCNGenerator(latent_dim, output_dim, num_channels, num_layers, dropout=dropout)
    discriminator = TCNDiscriminator(input_dim, num_channels, num_layers, dropout=dropout)
    generator.to(device)
    discriminator.to(device)
    
    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wdg)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wdd)
    
    batches_done = 0
    i = -print_each_n // 2
    start_all = time.time()
    
    def create_windows(df, window_size):
        data = df.to_numpy()
        windows = []
        for j in range(len(data) - window_size + 1):
            window = data[j: j+window_size]
            windows.append(window)
        return np.array(windows)
    
    windows = create_windows(df_train, time_window)
    num_windows = windows.shape[0]
    indices = np.random.permutation(num_windows)
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        start = time.time()
        for idx in range(0, num_windows, batch_size):
            batch_indices = indices[idx: idx+batch_size]
            batch_windows = windows[batch_indices]
            real_data = torch.from_numpy(batch_windows).float().permute(0, 2, 1).to(device)
            
            optimizer_D.zero_grad()
            z = torch.randn(real_data.size(0), time_window, latent_dim, device=device)
            fake_data = generator(z).detach().permute(0, 2, 1)
            
            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data)
            loss_D = -torch.mean(disc_real) + torch.mean(disc_fake)
            loss_D.backward()
            optimizer_D.step()
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)
            i += 1
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                gen_data = generator(z).permute(0, 2, 1)
                loss_G = -torch.mean(discriminator(gen_data))
                loss_G.backward()
                optimizer_G.step()
                if i % print_each_n == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] [Batch {idx}/{num_windows}] "
                          f"[D real: {disc_real.mean().item():.6f}] [D fake: {disc_fake.mean().item():.6f}] "
                          f"[G loss: {loss_G.item():.6f}]")
                    if do_print and step_by_step:
                        input("Pressione Enter para continuar...")
            batches_done += 1
        end = time.time()
        print(f"Tempo de treinamento da Epoch {epoch+1}: {end - start:.3f} segundos")
        
        if dataset_val is not None and y_val is not None:
            discriminator.eval()
            scores = discriminate(discriminator, dataset_val, time_window=time_window, batch_size=400, device=device)
            try:
                auc = metrics.roc_auc_score(y_val, scores)
            except Exception as e:
                auc = 0.0
                print("Erro ao calcular AUC:", e)
            print(f"Epoch {epoch+1} - Validation AUC: {auc:.6f}")
            if early_stopping is not None:
                early_stopping(auc, discriminator, generator)
                if early_stopping.early_stop:
                    print(f"Treinamento interrompido pelo early stopping na Epoch {epoch+1}")
                    generator = torch.load(early_stopping.path2, map_location=device)
                    discriminator = torch.load(early_stopping.path, map_location=device)
                    return generator, discriminator
            discriminator.train()
    end_all = time.time()
    print(f"Tempo total de treinamento: {end_all - start_all:.3f} segundos")
    if early_stopping is not None:
        generator = torch.load(early_stopping.path2, map_location=device)
        discriminator = torch.load(early_stopping.path, map_location=device)
    return generator, discriminator

# ============================================================
# Função de inferência: cria janelas, retorna scores e mostra tempo de inferência
@torch.no_grad()
def discriminate(discriminator: torch.nn.Module, df_val: pd.DataFrame, time_window=40, batch_size=400, device="cpu"):
    dataset = TimeWindowDataset(df_val, time_window=time_window)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    start_inf = time.time()
    for batch in loader:
        batch = batch.to(device)
        # Ajusta a forma para (batch, features, time_window)
        batch = batch.permute(0, 2, 1)
        out = discriminator(batch).cpu().numpy()
        scores.extend([-s for s in out])
    end_inf = time.time()
    total_time = end_inf - start_inf
    avg_time = total_time / len(dataset)
    print(f"Total de inferência: {total_time:.3f} s, Tempo médio por janela: {avg_time*1000:.3f} ms")
    return scores

# ============================================================
# Função para calcular a acurácia por tipo de ataque (sem tempo de detecção)
def compute_accuracy_by_attack(df_orig: pd.DataFrame, scores, time_window, threshold):
    """
    Calcula a quantidade de janelas e a acurácia média para cada tipo de ataque,
    considerando a janela rotulada pelo rótulo da última linha (time_window-1).
    """
    labels_window = df_orig['Label'].iloc[time_window-1:].reset_index(drop=True)
    predicted = np.array(scores) > threshold

    results = {}
    counts = labels_window.value_counts()
    for attack in counts.index:
        if attack.strip().upper() == "BENIGN":
            continue
        num_examples = int(counts[attack])
        attack_indices = labels_window[labels_window == attack].index.to_list()
        if len(attack_indices) == 0:
            continue
        acc = np.mean(predicted[attack_indices])
        results[attack] = {
            "num_examples": num_examples,
            "accuracy": acc
        }
    return results

# ============================================================
# Main
def main():
    if len(sys.argv) != 6:
        print("Uso: python TCN_train_new.py <monday.csv> <wednesday.csv> <friday.csv> <mode: train/val/test/optuna> <model_type: tcn>")
        sys.exit(1)
    
    path_monday = sys.argv[1]
    path_wed = sys.argv[2]
    path_friday = sys.argv[3]
    mode = sys.argv[4].lower()  # "train", "val", "test" ou "optuna"
    model_type = sys.argv[5].lower()  # para TCN, "tcn"
    
    RANDOM_SEED = 5
    set_seed(RANDOM_SEED)
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    df_monday = GetDataset(path_monday, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df_wed = GetDataset(path_wed, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df_friday = GetDataset(path_friday, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    
    # Ordena os dataframes de validação e teste pelo timestamp
    df_wed = df_wed.sort_values("Timestamp").reset_index(drop=True)
    df_friday = df_friday.sort_values("Timestamp").reset_index(drop=True)
    
    print("Unique labels (raw) in wednesday:", df_wed["Label"].unique())
    
    # Guarda cópias dos dados originais (com Timestamp e Label) para métricas
    df_monday_orig = df_monday.copy()
    df_wed_orig = df_wed.copy()
    df_friday_orig = df_friday.copy()
    
    # Separa train/val/test
    df_train = df_monday[df_monday["Label"]=="BENIGN"].copy()
    df_val = df_wed.copy()
    df_test = df_friday.copy()
    
    # Converte o label para "BENIGN"/"ATTACK"
    lbl_train = df_train["Label"].apply(lambda c: "BENIGN" if c.strip().upper()=="BENIGN" else "ATTACK")
    lbl_val = df_val["Label"].apply(lambda c: "BENIGN" if c.strip().upper()=="BENIGN" else "ATTACK")
    print("Unique labels (after conversion) in validation:", np.unique(lbl_val.values))
    lbl_test = df_test["Label"].apply(lambda c: "BENIGN" if c.strip().upper()=="BENIGN" else "ATTACK")
    
    # Remove colunas de Label/Timestamp do input
    drop_cols = []
    for col in ["Label", "Timestamp"]:
        if col in df_train.columns:
            drop_cols.append(col)
    df_train = df_train.drop(drop_cols, axis=1)
    df_val = df_val.drop(drop_cols, axis=1)
    df_test = df_test.drop(drop_cols, axis=1)
    
    # Converte IP para valor numérico e normaliza
    for col in ["Source IP", "Destination IP"]:
        df_train[col] = df_train[col].map(lambda x: int(IPv4Address(x)))
        df_val[col] = df_val[col].map(lambda x: int(IPv4Address(x)))
        df_test[col] = df_test[col].map(lambda x: int(IPv4Address(x)))
    
    df_train = df_train.apply(pd.to_numeric, errors='coerce')
    df_val = df_val.apply(pd.to_numeric, errors='coerce')
    df_test = df_test.apply(pd.to_numeric, errors='coerce')
    
    train_min = df_train.min().astype("float64")
    train_max = df_train.max().astype("float64")
    
    def norm_other(df):
        df = df.copy()
        df = (df - train_min) / (train_max - train_min)
        return df.fillna(0)
    
    df_train_norm = norm_other(df_train)
    df_val_norm = norm_other(df_val)
    df_test_norm = norm_other(df_test)
    
    print("Número de features:", df_train_norm.shape[1])
    time_window = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Para cálculo de métricas, os rótulos de janela usam o rótulo da última linha
    y_val_window = lbl_val.iloc[time_window-1:].apply(lambda c: 0 if c.strip().upper()=="BENIGN" else 1).to_numpy()
    
    if mode == "train" and model_type == "tcn":
        print("Entrando no modo de treinamento TCN...")
        latent_dim = 30
        batch_size = 32
        epochs = 10
        set_seed(RANDOM_SEED)
        generator, discriminator = TrainTCN(
            df_train_norm,
            latent_dim=latent_dim,
            output_dim=df_train_norm.shape[1],
            input_dim=df_train_norm.shape[1],
            lrd=2e-4,
            lrg=1e-4,
            epochs=epochs,
            dataset_val=df_val_norm,
            y_val=y_val_window,
            n_critic=5,
            clip_value=1,
            optim=torch.optim.RMSprop,
            wdd=1e-2,
            wdg=1e-2,
            early_stopping=EarlyStopping(5, 0),
            dropout=0.2,
            time_window=time_window,
            batch_size=batch_size,
            num_channels=32,
            num_layers=1,
            do_print=True
        )
        torch.save(generator, "GeneratorTCN.torch")
        torch.save(discriminator, "DiscriminatorTCN.torch")
    
    elif mode == "optuna" and model_type=="tcn":
        print("Entrando no modo de otimização com Optuna para TCN...")
        def objective(trial):
            RANDOM_SEED = 5
            set_seed(RANDOM_SEED)
            latent_dim = trial.suggest_int("latent_dim", 10, 50)
            num_channels = trial.suggest_int("num_channels", 16, 64)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
            lrd = trial.suggest_loguniform("lrd", 1e-5, 1e-3)
            lrg = trial.suggest_loguniform("lrg", 1e-5, 1e-3)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            epochs = 10
            
            gen, disc = TrainTCN(
                df_train_norm,
                latent_dim=latent_dim,
                output_dim=df_train_norm.shape[1],
                input_dim=df_train_norm.shape[1],
                lrd=lrd,
                lrg=lrg,
                epochs=epochs,
                dataset_val=df_val_norm,
                y_val=y_val_window,
                n_critic=5,
                clip_value=1,
                optim=torch.optim.RMSprop,
                wdd=1e-2,
                wdg=1e-2,
                early_stopping=None,
                dropout=dropout,
                time_window=time_window,
                batch_size=batch_size,
                num_channels=num_channels,
                num_layers=num_layers,
                do_print=False
            )
            scores = discriminate(disc, df_val_norm, time_window=time_window, batch_size=400, device=device)
            try:
                auc = metrics.roc_auc_score(y_val_window, scores)
            except Exception as e:
                auc = 0.0
                print("Erro ao calcular AUC:", e)
            return auc
        
        from optuna.samplers import TPESampler
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=20)
        print("Melhor trial:")
        print("Objetivo (AUC):", study.best_trial.value)
        print("Parâmetros:", study.best_trial.params)
        best_params = study.best_trial.params
        final_epochs = 20
        
        set_seed(RANDOM_SEED)
        gen, disc = TrainTCN(
            df_train_norm,
            latent_dim=best_params["latent_dim"],
            output_dim=df_train_norm.shape[1],
            input_dim=df_train_norm.shape[1],
            lrd=best_params["lrd"],
            lrg=best_params["lrg"],
            epochs=final_epochs,
            dataset_val=df_val_norm,
            y_val=y_val_window,
            n_critic=5,
            clip_value=1,
            optim=torch.optim.RMSprop,
            wdd=1e-2,
            wdg=1e-2,
            early_stopping=EarlyStopping(15, 0),
            dropout=best_params["dropout"],
            time_window=time_window,
            batch_size=best_params["batch_size"],
            num_channels=best_params["num_channels"],
            num_layers=best_params["num_layers"],
            do_print=True
        )
        torch.save(gen, "GeneratorTCN.torch")
        torch.save(disc, "DiscriminatorTCN.torch")
    
    elif mode == "val" and model_type=="tcn":
        print("Entrando no modo de validação TCN...")
        disc_model = torch.load("DiscriminatorTCN.torch", map_location=device).eval()
        scores = discriminate(disc_model, df_val_norm, time_window=time_window, batch_size=400, device=device)
        
        y_val_window = lbl_val.iloc[time_window-1:].apply(lambda c: 0 if c.strip().upper()=="BENIGN" else 1).to_numpy()
        metrics.plot_roc_curve(y_val_window, scores, name="val")
        auc = metrics.roc_auc_score(y_val_window, scores)
        thresh = metrics.best_validation_threshold(y_val_window, scores)["thresholds"]
        acc = metrics.accuracy(y_val_window, np.array(scores) > thresh)
        print(f"VAL - AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        metrics.plot_confusion_matrix(y_val_window, np.array(scores) > thresh, name="val")
        
        # Acurácia por ataque (sem tempo)
        attack_results = compute_accuracy_by_attack(df_wed_orig, scores, time_window, thresh)
        print("\nAcurácia por ataque:")
        for attack_type, res in attack_results.items():
            print(f"Tipo: {attack_type}")
            print(f"  - Número de exemplos: {res['num_examples']}")
            print(f"  - Acurácia média: {res['accuracy']:.4f}")
    
    elif mode == "test" and model_type=="tcn":
        print("Entrando no modo de teste TCN...")
        disc_model = torch.load("DiscriminatorTCN.torch", map_location=device).eval()
        scores = discriminate(disc_model, df_test_norm, time_window=time_window, batch_size=400, device=device)
        
        y_test_window = lbl_test.iloc[time_window-1:].apply(lambda c: 0 if c.strip().upper()=="BENIGN" else 1).to_numpy()
        metrics.plot_roc_curve(y_test_window, scores, name="test")
        auc = metrics.roc_auc_score(y_test_window, scores)
        thresh = metrics.best_validation_threshold(y_test_window, scores)["thresholds"]
        acc = metrics.accuracy(y_test_window, np.array(scores) > thresh)
        print(f"TEST - AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        metrics.plot_confusion_matrix(y_test_window, np.array(scores) > thresh, name="test")
        
        # Acurácia por ataque (sem tempo)
        attack_results = compute_accuracy_by_attack(df_friday_orig, scores, time_window, thresh)
        print("\nAcurácia por ataque:")
        for attack_type, res in attack_results.items():
            print(f"Tipo: {attack_type}")
            print(f"  - Número de exemplos: {res['num_examples']}")
            print(f"  - Acurácia média: {res['accuracy']:.4f}")
    
    else:
        print("Modo inválido. Use 'train', 'optuna', 'val' ou 'test' com 'tcn'.")
    
if __name__ == "__main__":
    main()
