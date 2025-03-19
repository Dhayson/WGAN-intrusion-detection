import pandas as pd
import torch
import torch_optimizer
import optuna
from src.early_stop import EarlyStopping
from src.into_dataloader import IntoDataset
from src.wgan.self_attention_wgan import TrainSelfAttention

def TuneSA(df_train: pd.DataFrame, df_val: pd.DataFrame, y_val: pd.Series):
    def objective(trial: optuna.trial.Trial):
        # Parâmetros:
        # lrd: taxa de aprendizagem do discriminador
        lrd = trial.suggest_float("lrd", 1e-6, 1e-3)
        # lrg: taxa de aprendizagem do gerador
        lrg = trial.suggest_float("lrg", 1e-6, 1e-3)
        # n_critic: em 1/n das vezes o gerador treina
        n_critic = trial.suggest_int("n_critic", 2, 7)
        # clip_value: limite dos valores nos pesos, definindo a convergência da wgan
        clip_value = trial.suggest_float("clip_value", 0.5, 3)
        # latent_dim: dimensão do espaço latente
        latent_dim = trial.suggest_int("latent_dim", 8, 40)
        # optim: algoritmo de otimização de aprendizado
        optim = trial.suggest_categorical("optim", [torch.optim.Adam, torch_optimizer.Yogi])
        # wdd: decaimento dos pesos do discriminador
        wdd = trial.suggest_float("wdd", 1e-6, 1e-2)
        # wdg: decaimento dos pesos do gerador
        wdg = trial.suggest_float("wdg", 1e-6, 1e-2)
        # dropout: dropout aplicado no self-attention
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        # time_window: quantos valores no passado são inseridos no modelo. Tem um impacto considerável na performance
        time_window = trial.suggest_int("time_window", 10, 60)
        # batch_size: tamanho do batch. Requer uma GPU proporcionalmente poderosa.
        batch_size = trial.suggest_int("batch_size", 1, 128)
        # headsd: cabeças do self_attention no discriminador
        # embedd: dimensão das entradas do discriminador. NOTA: embedd / headsd deve ser inteiro
        headsd = trial.suggest_int("headsd", 20, 100, step=2)
        embedd = headsd*trial.suggest_int("embedd_factor", 1, 10)
        # headsg: cabeças do self_attention no gerador
        # embedg: dimensão das entradas do gerador. NOTA: embedg / headsg deve ser inteiro
        headsg = trial.suggest_int("headsg", 20, 100, step=2)
        embedg = headsg*trial.suggest_int("embedg_factor", 1, 10)
        #
        # Fatores:
        # early_stopping: quantas epochs treinar sem gerar uma melhoria
        early_stopping = EarlyStopping(3,0) # Usar um número baixo para otimizar a busca
        # epochs: número de epochs máximo
        epochs = 50
        # data_len: tamanho de uma entrada do dataset (40)
        data_len = 40
        
        dataset_train = IntoDataset(df_train, time_window)
        dataset_val = IntoDataset(df_val, time_window)
        _, _, auc_score = TrainSelfAttention(dataset_train, lrd, lrg, epochs, dataset_val, y_val,
            n_critic, clip_value, latent_dim, optim, wdd, wdg, early_stopping, dropout, 1e32, time_window, batch_size, headsd,
            embedd, headsg, embedg, return_auc=True)
        
        return auc_score
    study = optuna.create_study(direction="maximize")
    
    # Otimizar o AUC score
    print("Running optuna optimization")
    study.optimize(objective, n_trials=100)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)