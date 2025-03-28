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
from src.import_dataset_alt import GetDataset2017
from src.dataset_split import SplitDataset
import src.metrics as metrics
from src.early_stop import EarlyStopping
from src.wgan.TCN_wgan import *
import optuna

# ------------------------------------------------------------
# Main com modos 'train', 'val', 'test' e 'optuna'
def RunModelTCN2019():
    print("Iniciando main...")
    print("Argumentos passados:", sys.argv)
    if len(sys.argv) < 5:
        print("Uso: python TCN_train.py <path_day1> <path_day2> <mode: train/val/test/optuna> <model_type: tcn>")
        sys.exit(1)
    
    path_day1 = sys.argv[2]
    path_day2 = sys.argv[3]
    mode = sys.argv[4]      # "train", "val", "test" ou "optuna"
    model_type = sys.argv[5]  # Para TCN, "tcn"
    
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"
    set_seed(RANDOM_SEED)
    
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
    y_val_window = df_val_label.iloc[:].apply(lambda c: 0 if c == "BENIGN" else 1).to_numpy()
    y_test_window = df_test_label.iloc[:].apply(lambda c: 0 if c == "BENIGN" else 1).to_numpy()
    
    cuda = True if torch.cuda.is_available() else False
    device = "cuda" if cuda else "cpu"
    
    # Define parâmetros fixos para treino (para uso na branch de train/val/test)
    input_dim = df_train.shape[1]
    output_dim = df_train.shape[1]
    
    dataset_train = IntoDataset(df_train, time_window)
    dataset_val = IntoDataset(df_val, time_window)
    dataset_test = IntoDataset(df_test, time_window)
    
    if mode == "train" and model_type == "tcn":
        print("Entrando no modo de treinamento TCN...")
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
            preds = discriminate(disc, dataset_val, time_window=time_window, batch_size=400, device=device)
            auc = metrics.roc_auc_score(y_val_window, preds)
            return auc
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)
        print("Melhor trial:")
        print("Valor objetivo (1 - AUC):", study.best_trial.value)
        print("Parâmetros:", study.best_trial.params)
    else:
        print("Nenhum modo válido selecionado. Use 'train tcn', 'optuna tcn', 'val tcn' ou 'test tcn'.")


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

def RunModelTCN2017():
    if len(sys.argv) != 6:
        print("Uso: python TCN_train_new.py <monday.csv> <wednesday.csv> <friday.csv> <mode: train/val/test/optuna> <model_type: tcn>")
        sys.exit(1)
    
    path_monday = sys.argv[1]
    path_wed = sys.argv[2]
    path_friday = sys.argv[3]
    mode = sys.argv[4].lower()  # "train", "val", "test" ou "optuna"
    model_type = sys.argv[5].lower()  # para TCN, "tcn"
    
    RANDOM_SEED = 5
    # set_seed(RANDOM_SEED)
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
        print("Entrando no modo de treinamento TCN...")
        latent_dim = 19
        batch_size = 16
        epochs = 50
        # set_seed(RANDOM_SEED)
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
            clip_value=1,
            optim=torch.optim.RMSprop,
            wdd=1e-2,
            wdg=1e-2,
            early_stopping=EarlyStopping(5, 0),
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
    
    else:
        print("Modo inválido. Use 'train', 'optuna', 'val' ou 'test' com 'tcn'.")
    

if __name__ == "__main__":
    RunModelTCN2019()
