import sys
import pandas as pd
import numpy as np
import torch
from ipaddress import IPv4Address
from numpy.random import RandomState

# Importa funções auxiliares do projeto
from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
import src.metrics as metrics
from src.early_stop import EarlyStopping
import optuna
import torch_optimizer
# Importa as funções implementadas na WGAN-GP (arquivo separado)
from src.wgan.TCN_wgan import TrainTCN, discriminate
import sys
import pandas as pd
import numpy as np
import torch
import torch_optimizer
import random
import time
from ipaddress import IPv4Address
import optuna

# Importa funções auxiliares do projeto
from src.import_dataset import GetDataset
from src.import_dataset_alt import GetDataset2017

from src.dataset_split import SplitDataset
import src.metrics as metrics
from src.early_stop import EarlyStopping

# ------------------------------------------------------------
# Função para definir a seed e garantir replicabilidade
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# Dataset personalizado: cada amostra é uma janela temporal
class TimeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, time_window: int = 40):
        super().__init__()
        # Converte o DataFrame para array NumPy para indexação posicional
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
        # z: (batch, time_window, latent_dim)
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
        # x: (batch, time_window, input_dim)
        x = self.input_linear(x)   # (batch, time_window, num_channels)
        x = x.transpose(1, 2)      # (batch, num_channels, time_window)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.mean(dim=2)          # Pooling global
        out = self.output_linear(x)  # (batch, 1)
        return out.squeeze(1)

# ------------------------------------------------------------
# Função de treinamento para TCN-based WGAN-GP
def TrainTCNGP(df_train: pd.DataFrame,
             latent_dim, output_dim, input_dim,
             lrd, lrg, epochs,
             dataset_val: pd.DataFrame = None,
             y_val=None,
             n_critic=5,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_train = TimeWindowDataset(df_train, time_window)
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
    lambda_gp = 10  # Peso do gradient penalty
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        start = time.time()
        for batch in dataloader_train:
            real_data = batch.to(device)
            current_batch_size = real_data.size(0)
            optimizer_D.zero_grad()
            
            z = torch.randn(current_batch_size, time_window, latent_dim, device=device)
            fake_data = generator(z).detach()
            
            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data)
            
            loss_D = -torch.mean(disc_real) + torch.mean(disc_fake)
            
            alpha = torch.rand(current_batch_size, 1, 1, device=device)
            interpolates = alpha * real_data + (1 - alpha) * fake_data
            interpolates.requires_grad_(True)
            disc_interpolates = discriminator(interpolates)
            grad_outputs = torch.ones_like(disc_interpolates, device=device)
            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradients = gradients.view(current_batch_size, -1)
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            loss_D += gradient_penalty
            loss_D.backward()
            optimizer_D.step()
            
            i += 1
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                gen_data = generator(z)
                loss_G = -torch.mean(discriminator(gen_data))
                loss_G.backward()
                optimizer_G.step()
                if i % print_each_n == 0:
                    print(f"\r[Epoch {epoch+1}/{epochs}] [Batch {batches_done % len(dataloader_train)}/{len(dataloader_train)}] "
                          f"[D real: {disc_real.mean().item():.6f}] [D fake: {disc_fake.mean().item():.6f}] "
                          f"[G loss: {loss_G.item():.6f}]", end="")
                    if do_print and step_by_step:
                        input("Pressione Enter para continuar...")
            batches_done += 1
        end = time.time()
        print()
        print(f"Tempo de treinamento da Epoch {epoch+1}: {end - start:.3f} segundos")
        
        # Validação
        if dataset_val is not None and y_val is not None:
            discriminator.eval()
            val_dataset = TimeWindowDataset(dataset_val, time_window)
            preds = discriminate(discriminator, val_dataset, time_window=time_window, batch_size=400, device=device)
            adjusted_y_val = y_val[time_window - 1:]
            try:
                auc = metrics.roc_auc_score(adjusted_y_val, preds)
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
def discriminate(discriminator: torch.nn.Module, dataset_val: torch.utils.data.Dataset, time_window=40, batch_size=400, device="cpu"):
    loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    scores = []
    start = time.time()
    i = 0
    for batch in loader:
        batch = batch.to(device)
        out = discriminator(batch).cpu().numpy()
        scores.extend([-s for s in out])
        if i % 50 == 0:
            print(f"\r[Validating] [Sample {i} / {len(loader)}] [Score {np.squeeze(np.mean(out[0]))}]", end="")
        i += 1
    end = time.time()
    print(f"Validation time: {end-start}")
    return scores

# ------------------------------------------------------------
# Função principal para execução (modo train ou optuna)
def RunModelTCNGP2019():
    print("Iniciando main...")
    if len(sys.argv) < 6:
        print("Uso: python main.py <year> <path_day1> <path_day2> <mode: train/optuna> <model_type: tcn>")
        sys.exit(1)
    
    dataset_year = sys.argv[1]
    path_day1 = sys.argv[2]
    path_day2 = sys.argv[3]
    mode = sys.argv[4].lower()       # "train" ou "optuna"
    model_type = sys.argv[5].lower()   # para TCN, "tcn"
    
    RANDOM_SEED = 5
    set_seed(RANDOM_SEED)
    rs = np.random.RandomState(RANDOM_SEED)
    
    df_day1 = GetDataset(path_day1, rs, "csv", filter=False, do_print=False).drop_duplicates()
    df_day2 = GetDataset(path_day2, rs, "csv", filter=False, do_print=False).drop_duplicates()
    
    df_train, df_val, df_test = SplitDataset(df_day1, df_day2, rs, "csv")
    df_train = df_train.sort_values("Timestamp").reset_index(drop=True)
    df_val   = df_val.sort_values("Timestamp").reset_index(drop=True)
    df_test  = df_test.sort_values("Timestamp").reset_index(drop=True)
    
    y_val = df_val["Label"].apply(lambda c: 0 if c=="BENIGN" else 1).to_numpy()
    df_train = df_train.drop(["Label", "Timestamp"], axis=1)
    df_val   = df_val.drop(["Label", "Timestamp"], axis=1)
    df_test  = df_test.drop(["Label", "Timestamp"], axis=1)
    
    for col in ["Source IP", "Destination IP"]:
        df_train[col] = df_train[col].map(lambda x: int(IPv4Address(x)))
        df_val[col]   = df_val[col].map(lambda x: int(IPv4Address(x)))
        df_test[col]  = df_test[col].map(lambda x: int(IPv4Address(x)))
    
    train_min = df_train.min().astype("float64")
    train_max = df_train.max().astype("float64")
    df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
    df_train = df_train.fillna(0)
    df_val = (df_val - train_min) / (train_max - train_min)
    df_val = df_val.fillna(0)
    df_test = (df_test - train_min) / (train_max - train_min)
    df_test = df_test.fillna(0)
    
    time_window = 40
    input_dim = df_train.shape[1]
    output_dim = df_train.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if mode == "train" and model_type == "tcn":
        print("Entrando no modo de treinamento TCN com WGAN-GP...")
        latent_dim = 10
        num_channels = 64
        num_layers = 3
        batch_size = 128
        epochs = 50
        
        generator, discriminator = TrainTCN(
            df_train,
            latent_dim=latent_dim,
            output_dim=output_dim,
            input_dim=input_dim,
            lrd=2e-4,
            lrg=1e-4,
            epochs=epochs,
            dataset_val=df_val,
            y_val=y_val,
            n_critic=4,
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
        print("Entrando no modo de otimização com Optuna para TCN...")
        def objective(trial):
            global TimeWindowDataset  # Garante que a classe esteja acessível
            latent_dim = trial.suggest_int("latent_dim", 5, 20)
            num_channels = trial.suggest_int("num_channels", 32, 128)
            num_layers = trial.suggest_int("num_layers", 2, 2)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            lrd = trial.suggest_loguniform("lrd", 1e-5, 1e-3)
            lrg = trial.suggest_loguniform("lrg", 1e-5, 1e-3)
            dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
            epochs = 20  # Épocas reduzidas para tuning
            
            gen, disc = TrainTCN(
                df_train,
                latent_dim=latent_dim,
                output_dim=output_dim,
                input_dim=input_dim,
                lrd=lrd,
                lrg=lrg,
                epochs=epochs,
                dataset_val=df_val,
                y_val=y_val,
                n_critic=4,
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
            preds = discriminate(disc, TimeWindowDataset(df_val, time_window), time_window=time_window, batch_size=400, device=device)
            adjusted_y_val = y_val[time_window - 1:]
            auc = metrics.roc_auc_score(adjusted_y_val, preds)
            return auc
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25)
        print("Melhor trial:")
        print("Valor objetivo (AUC):", study.best_trial.value)
        print("Parâmetros:", study.best_trial.params)
        
        # Retreina o modelo com os melhores parâmetros por mais épocas
        final_epochs = 50  # Número maior de épocas para o treinamento final
        best_params = study.best_trial.params
        gen, disc = TrainTCN(
            df_train,
            latent_dim=best_params["latent_dim"],
            output_dim=output_dim,
            input_dim=input_dim,
            lrd=best_params["lrd"],
            lrg=best_params["lrg"],
            epochs=final_epochs,
            dataset_val=df_val,
            y_val=y_val,
            n_critic=4,
            optim=torch_optimizer.Yogi,
            wdd=2e-2,
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
    
    else:
        print("Modo inválido. Use 'train tcn' ou 'optuna tcn'.")
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

def RunModelTCNGP2017():
    if len(sys.argv) != 6:
        print("Uso: python TCN_train_new.py <monday.csv> <wednesday.csv> <friday.csv> <mode: train/val/test/optuna> <model_type: tcn>")
        sys.exit(1)
    
    path_monday = sys.argv[1]
    path_wed = sys.argv[2]
    path_friday = sys.argv[3]
    mode = sys.argv[4].lower()  # "train", "val", "test" ou "optuna"
    model_type = sys.argv[5].lower()  # para TCN, "tcn"
    
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    df_monday = GetDataset2017(path_monday, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df_wed = GetDataset2017(path_wed, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    df_friday = GetDataset2017(path_friday, rs, DATASET_FORMAT, filter=False, do_print=False).drop_duplicates()
    
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
    y_val_window = lbl_val.iloc[:].apply(lambda c: 0 if c.strip().upper()=="BENIGN" else 1).to_numpy()
    
    if mode == "train" and model_type == "tcn":
        print("Entrando no modo de treinamento TCN com WGAN-GP...")
        latent_dim = 19
        batch_size = 16
        epochs = 50
        generator, discriminator = TrainTCN(
            df_train_norm,
            latent_dim=latent_dim,
            output_dim=df_train_norm.shape[1],
            input_dim=df_train_norm.shape[1],
            lrd=9.48030648267979e-05,
            lrg=0.00016729683536668868,
            epochs=epochs,
            dataset_val=df_val_norm,
            y_val=y_val_window,
            n_critic=5,
            optim=torch.optim.RMSprop,  # Se preferir, substitua por um otimizador que apresente melhores resultados com WGAN-GP
            wdd=1e-2,
            wdg=1e-2,
            early_stopping=EarlyStopping(15, 0),
            dropout=0.4674443631751687,
            time_window=time_window,
            batch_size=batch_size,
            num_channels=58,
            num_layers=1,
            do_print=True
        )
        torch.save(generator, "GeneratorTCN.torch")
        torch.save(discriminator, "DiscriminatorTCN.torch")
    
    elif mode == "optuna" and model_type=="tcn":
        print("Entrando no modo de otimização com Optuna para TCN com WGAN-GP...")
        def objective(trial):
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
    
    else:
        print("Modo inválido. Use 'train', 'optuna', 'val' ou 'test' com 'tcn'.")
