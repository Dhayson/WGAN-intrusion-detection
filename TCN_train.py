import sys
import pandas as pd
import numpy as np
from numpy.random import RandomState
import torch
import torch_optimizer
from ipaddress import IPv4Address
import random

# Importa as funções do projeto
from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
import src.metrics as metrics
from src.early_stop import EarlyStopping

# ------------------------------------------------------------
# Dataset personalizado: cada amostra é uma janela temporal
class TimeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, time_window: int = 40):
        super().__init__()
        # Converte o DataFrame para um array NumPy para indexação por posição
        self.data = df.to_numpy().astype(np.float32)
        self.time_window = time_window
        self.num_samples = len(self.data) - time_window + 1
        if self.num_samples < 1:
            raise ValueError(f"DataFrame com {len(df)} linhas não comporta janela de {time_window} timesteps.")
        print(f"[TimeWindowDataset] data.shape: {self.data.shape}, type: {type(self.data)}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.time_window]  # (time_window, num_features)
        return torch.tensor(seq, dtype=torch.float32)

# ------------------------------------------------------------
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
        x = self.input_linear(z)  # (batch, time_window, num_channels)
        x = x.transpose(1, 2)     # (batch, num_channels, time_window)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)     # (batch, time_window, num_channels)
        out = self.output_linear(x)  # (batch, time_window, output_dim)
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
        x = self.input_linear(x)   # (batch, time_window, num_channels)
        x = x.transpose(1, 2)      # (batch, num_channels, time_window)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.mean(dim=2)          # Pooling global
        out = self.output_linear(x)  # (batch, 1)
        return out.squeeze(1)

# ------------------------------------------------------------
# Função de treinamento para TCN-based WGAN
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
             print_each_n=20,
             time_window=40,
             batch_size=5,
             num_channels=64,
             num_layers=3,
             do_print=False,
             step_by_step=False):
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cria dataset de treino (janelas temporais)
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
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        start = time.time()
        for batch in dataloader_train:
            real_data = batch.to(device)
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, time_window, latent_dim, device=device)
            fake_data = generator(z).detach()
            
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
                gen_data = generator(z)
                loss_G = -torch.mean(discriminator(gen_data))
                loss_G.backward()
                optimizer_G.step()
                if i % print_each_n == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] [Batch {batches_done % len(dataloader_train)}/{len(dataloader_train)}] "
                          f"[D real: {disc_real.mean().item():.6f}] [D fake: {disc_fake.mean().item():.6f}] "
                          f"[G loss: {loss_G.item():.6f}]")
                    if do_print and step_by_step:
                        input("Pressione Enter para continuar...")
            batches_done += 1
        end = time.time()
        print(f"Tempo de treinamento da Epoch {epoch+1}: {end - start:.3f} segundos")
        
        if dataset_val is not None and y_val is not None:
            discriminator.eval()
            val_dataset = TimeWindowDataset(dataset_val, time_window=time_window)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=400, shuffle=False)
            preds = []
            with torch.no_grad():
                for batch_val in val_loader:
                    batch_val = batch_val.to(device)
                    out = discriminator(batch_val).cpu().numpy()
                    preds.extend([-s for s in out])
            try:
                auc = metrics.roc_auc_score(y_val, preds)
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

# ------------------------------------------------------------
# Função de inferência: cria janelas e retorna scores
@torch.no_grad()
def discriminate(discriminator: torch.nn.Module, df_val: pd.DataFrame, time_window=40, batch_size=400, device="cpu"):
    dataset_val = TimeWindowDataset(df_val, time_window=time_window)
    loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    scores = []
    for batch in loader:
        batch = batch.to(device)
        out = discriminator(batch).cpu().numpy()
        scores.extend([-s for s in out])
    return scores

# ------------------------------------------------------------
# Main com modos 'train', 'val', 'test' e 'optuna'
def main():
    print("Iniciando main...")
    print("Argumentos passados:", sys.argv)
    if len(sys.argv) < 5:
        print("Uso: python TCN_train.py <path_day1> <path_day2> <mode: train/val/test/optuna> <model_type: tcn>")
        sys.exit(1)
    
    path_day1 = sys.argv[1]
    path_day2 = sys.argv[2]
    mode = sys.argv[3]      # "train", "val", "test" ou "optuna"
    model_type = sys.argv[4]  # Para TCN, "tcn"
    
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"
    
    # Carrega os datasets e remove duplicatas
    df_day_1 = GetDataset(path_day1, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df_day_2 = GetDataset(path_day2, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    
    df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    # Ordena por Timestamp para preservar a dependência temporal
    df_train = df_train.sort_values(by="Timestamp", ignore_index=True)
    df_val   = df_val.sort_values(by="Timestamp", ignore_index=True)
    df_test  = df_test.sort_values(by="Timestamp", ignore_index=True)
    
    # Guarda os rótulos originais
    df_train_label = df_train["Label"]
    df_val_label   = df_val["Label"]
    df_test_label  = df_test["Label"]
    
    # Remove as colunas "Label" e "Timestamp"
    df_train = df_train.drop(["Label", "Timestamp"], axis=1)
    df_val   = df_val.drop(["Label", "Timestamp"], axis=1)
    df_test  = df_test.drop(["Label", "Timestamp"], axis=1)
    
    # Converte os IPs para inteiros
    for col in ["Source IP", "Destination IP"]:
        df_train[col] = df_train[col].map(lambda x: int(IPv4Address(x)))
        df_val[col]   = df_val[col].map(lambda x: int(IPv4Address(x)))
        df_test[col]  = df_test[col].map(lambda x: int(IPv4Address(x)))
    
    # Normalização utilizando os mínimos e máximos do treino
    train_min = df_train.min().astype("float64")
    train_max = df_train.max().astype("float64")
    min_max   = (train_max - train_min).to_numpy()
    
    df_train = (df_train - df_train.min())/(df_train.max() - df_train.min())
    df_train = df_train.fillna(0)
    
    df_val = (df_val - train_min)/(min_max)
    df_val = df_val.fillna(0)
    
    df_test = (df_test - train_min)/(min_max)
    df_test = df_test.fillna(0)
    
    # Define o time_window e ajusta os rótulos para janelas (rótulo da última linha de cada janela)
    time_window = 40
    y_val_window = df_val_label.iloc[time_window-1:].apply(lambda c: 0 if c == "BENIGN" else 1).to_numpy()
    y_test_window = df_test_label.iloc[time_window-1:].apply(lambda c: 0 if c == "BENIGN" else 1).to_numpy()
    
    cuda = True if torch.cuda.is_available() else False
    device = "cuda" if cuda else "cpu"
    
    # Define parâmetros fixos para treino (para uso na branch de train/val/test)
    input_dim = df_train.shape[1]
    output_dim = df_train.shape[1]
    
    if mode == "train" and model_type == "tcn":
        print("Entrando no modo de treinamento TCN...")
        latent_dim = 10
        num_channels = 64
        num_layers = 3
        batch_size = 128
        epochs = 10
        
        generator, discriminator = TrainTCN(
            df_train,
            latent_dim=latent_dim,
            output_dim=output_dim,
            input_dim=input_dim,
            lrd=2e-4,
            lrg=1e-4,
            epochs=epochs,
            dataset_val=df_val,
            y_val=y_val_window,
            n_critic=4,
            clip_value=0.9,
            optim=torch_optimizer.Yogi,
            wdd=2e-2,
            wdg=1e-2,
            early_stopping=EarlyStopping(15, 0),
            dropout=0.2,
            time_window=time_window,
            batch_size=batch_size,
            num_channels=num_channels,
            num_layers=num_layers,
            do_print=True
        )
        torch.save(generator, "GeneratorTCN.torch")
        torch.save(discriminator, "DiscriminatorTCN.torch")
    
    elif mode == "optuna" and model_type == "tcn":
        print("Entrando no modo de otimização com Optuna...")
        import optuna
        def objective(trial):
            latent_dim = trial.suggest_int("latent_dim", 5, 20)
            num_channels = trial.suggest_int("num_channels", 32, 128)
            num_layers = trial.suggest_int("num_layers", 1, 4)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            lrd = trial.suggest_loguniform("lrd", 1e-5, 1e-3)
            lrg = trial.suggest_loguniform("lrg", 1e-5, 1e-3)
            dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
            epochs = 5  # Use menos épocas para tuning
            
            gen, disc = TrainTCN(
                df_train,
                latent_dim=latent_dim,
                output_dim=output_dim,
                input_dim=input_dim,
                lrd=lrd,
                lrg=lrg,
                epochs=epochs,
                dataset_val=df_val,
                y_val=y_val_window,
                n_critic=4,
                clip_value=0.9,
                optim=torch_optimizer.Yogi,
                wdd=2e-2,
                wdg=1e-2,
                early_stopping=None,
                dropout=dropout,
                time_window=time_window,
                batch_size=batch_size,
                num_channels=num_channels,
                num_layers=num_layers,
                do_print=False
            )
            preds = discriminate(disc, df_val, time_window=time_window, batch_size=400, device=device)
            auc = metrics.roc_auc_score(y_val_window, preds)
            return 1 - auc  # queremos maximizar AUC, logo minimizamos 1 - AUC
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        print("Melhor trial:")
        print("Valor objetivo (1 - AUC):", study.best_trial.value)
        print("Parâmetros:", study.best_trial.params)
    
    elif mode == "val" or mode == "test":
        if mode == "val":
            df_x = df_val
            y_x = y_val_window
        else:
            df_x = df_test
            y_x = y_test_window
        print("Entrando no modo de validação/teste TCN...")
        discriminator_model = torch.load("DiscriminatorTCN.torch", map_location=device)
        generator_model = torch.load("GeneratorTCN.torch", map_location=device)
        discriminator_model.eval()
        generator_model.eval()
        
        preds = discriminate(discriminator_model, df_x, time_window=time_window, batch_size=400, device=device)
        best_thresh = metrics.best_validation_threshold(y_x, preds)
        thresh = best_thresh["thresholds"]
        auc = metrics.roc_auc_score(y_x, preds)
        acc = metrics.accuracy(y_x, np.array(preds) > thresh)
        print(f"AUC = {auc:.4f}, Accuracy = {acc:.4f}")
        print("Threshold:", thresh)
    
    else:
        print("Nenhum modo válido selecionado. Use 'train tcn', 'optuna tcn', 'val tcn' ou 'test tcn'.")

if __name__ == "__main__":
    main()
