from src.import_dataset import GetDataset
from src.import_dataset_alt import GetDataset2017
from src.dataset_split import SplitDataset
from src.wgan.wgan import discriminate, Discriminator, Generator, cuda
from src.wgan.lstm_wgan import TrainLSTM
from src.wgan.linear_wgan import TrainLinear
from src.wgan.self_attention_wgan import RunModelSelfAttention2019, RunModelSelfAttention2017
from src.wgan.TCN_train import RunModelTCN2019, RunModelTCN2017
from src.tuning import TuneSA
from src.wgan.wgan import discriminate, Discriminator, Generator, cuda
from src.wgan.TCN_wgan import discriminate as discriminateTCN
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
from optuna.pruners import MedianPruner
from src.into_dataloader import IntoDataset, IntoDatasetNoTime
from src.transform import MeanNormalizeTensor, MinMaxNormalizeTensor


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
    
    args= sys.argv
    dataset_kind = args[1]
    
    if args[1] == "2019":
        # Carrega os datasets (divididos entre dia 1 e dia 2)
        df_day_1 = GetDataset(args[2], rs, DATASET_FORMAT, filter=False, do_print=False)
        # Descartando duplicadas
        df_day_1 = DescartarDuplicatas(df_day_1, do_print=False)
        
        df_day_2 = GetDataset(args[3], rs, DATASET_FORMAT, filter=False, do_print=False)
        # Descartando duplicadas
        df_day_2 = DescartarDuplicatas(df_day_2, do_print=False)
        
        df_train, df_val, df_test = SplitDataset(df_day_1, df_day_2, rs, DATASET_FORMAT)
    elif args[1] == "2017":
        df_train = GetDataset2017(args[2], rs, DATASET_FORMAT, filter=False, do_print=False)
        df_val = GetDataset2017(args[3], rs, DATASET_FORMAT, filter=False, do_print=False)
        df_test = GetDataset2017(args[4], rs, DATASET_FORMAT, filter=False, do_print=False)
        args.pop(0)
    # DebugTrainValTest(df_train, df_val, df_test, BENIGN)
        
    # Essa coluna é importante para a dependência temporal!
    df_train = df_train.sort_values(by = "Timestamp", ignore_index=True)
    df_val = df_val.sort_values(by = "Timestamp", ignore_index=True)
    df_test = df_test.sort_values(by = "Timestamp", ignore_index=True)
    
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
    # min_max = (train_max - train_min).to_numpy()
    
    # Normalização
    df_train = df_train.fillna(0)
    df_val = df_val.fillna(0)
    df_test = df_test.fillna(0)
    
    # normalization = MeanNormalizeTensor(df_train.mean().to_numpy(dtype=np.float32), df_train.std().to_numpy(dtype=np.float32))
    normalization = MinMaxNormalizeTensor(df_train.max().to_numpy(dtype=np.float32), df_train.min().to_numpy(dtype=np.float32))
        
    # Validação: diferenciar entre benignos (0) e ataques (1)

    # Converte os rótulos para 0 (BENIGN) e 1 (ataque)
    y_val = df_val_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    y_test = df_test_label.apply(lambda c: 0 if c == 'BENIGN' else 1)
    
    if args[-1] == "sa":
        if args[1] == "2019":
            time_window = 77
        else:
            time_window = 80
    elif args[-1] == "lstm":
        min_max = (train_max - train_min).to_numpy()
        # Normalização
        df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
        df_train = df_train.fillna(0)
        df_val = (df_val - train_min) / (min_max)
        df_val = df_val.fillna(0)
        df_test = (df_test - train_min) / (min_max)
        df_test = df_test.fillna(0)
        
        time_window = 80
    elif args[-1] == "tcn":
        time_window = 80
        
    if args[-1] == "linear":
        dataset_train = IntoDatasetNoTime(df_train, normalization)
        dataset_val = IntoDatasetNoTime(df_val, normalization)
        dataset_test = IntoDatasetNoTime(df_test, normalization)
    elif args[-1] == "lstm":
        dataset_train = IntoDataset(df_train, time_window)
        dataset_val = IntoDataset(df_val, time_window)
        dataset_test = IntoDataset(df_test, time_window)
    else:
        dataset_train = IntoDataset(df_train, time_window, normalization)
        dataset_val = IntoDataset(df_val, time_window, normalization)
        dataset_test = IntoDataset(df_test, time_window, normalization)
    if len(args) > 4 and args[4] == "train":
        if args[5] == "linear":
            generator, discriminator = TrainLinear(dataset_train, 2e-5, 3e-5, 10, dataset_val, y_val,
                n_critic=3, optim=torch_optimizer.Yogi, wdd=2e-2, wdg=2e-2, early_stopping=EarlyStopping(3, 0), batch_size=10)
            torch.save(generator, "GeneratorLinear.torch")
            torch.save(discriminator, "DiscriminatorLinear.torch")
        elif args[5] == "sa":
            if dataset_kind == "2019":
                RunModelSelfAttention2019(dataset_train, dataset_val, y_val)
            elif dataset_kind == "2017":
                print("Using CIC-IDS-2017")
                RunModelSelfAttention2017(dataset_train, dataset_val, y_val)
        elif sys.argv[5] == "lstm":
            generator, discriminator = TrainLSTM(
                df_train, 0.00010870300025198446, 0.00028247627584454017, 10, df_val, y_val,
                wdd=2e-2, wdg=1e-2, optim=torch_optimizer.Yogi,
                early_stopping=EarlyStopping(15, 0), latent_dim=10,
                batch_size=128, n_critic=3, time_window=80,
                internal_d=512, internal_g=512, clip_value=0.1, print_each_n=1
            )
            torch.save(generator, "GeneratorLSTM.torch")
            torch.save(discriminator, "DiscriminatorLSTM.torch")
        elif sys.argv[5] == "tcn":
            if dataset_kind == "2019":
                RunModelTCN2019()
            elif dataset_kind == "2017":
                print("Using CIC-IDS-2017")
                RunModelTCN2017()
    elif len(args) > 4 and args[4] == "optuna":
        import optuna
        if sys.argv[5] == "tcn":
            if dataset_kind == "2019":
                RunModelTCN2019()
            elif dataset_kind == "2017":
                print("Using CIC-IDS-2017")
                RunModelTCN2017()
        elif sys.argv[5] == "lstm":
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
            print("A otimização para o modelo", sys.argv[5], "ainda não foi implementada.")

    elif len(args) > 4 and args[4] == "tune":
        if args[5] == "sa":
            TuneSA(df_train, df_val, y_val)
    elif len(args) > 4 and (args[4] == "val" or args[4] == "test"):
        if args[4] == "val":
            df_x = df_val
            dataset_x = dataset_val
            df_x_label: pd.Series = df_val_label
            y_x = y_val
        else:
            df_x = df_test
            dataset_x = dataset_test
            df_x_label: pd.Series = df_test_label
            y_x = y_test
        if args[-1] == "sa":
            discriminator: Discriminator = torch.load("DiscriminatorSA.torch", weights_only = False, map_location=torch.device(device)).to(device)
            generator: Generator = torch.load("GeneratorSA.torch", weights_only = False, map_location=torch.device(device)).to(device)
        if args[-1] == "lstm":
            discriminator: Discriminator = torch.load("DiscriminatorLSTM.torch", weights_only = False, map_location=torch.device(device)).to(device)
            generator: Generator = torch.load("GeneratorLSTM.torch", weights_only = False, map_location=torch.device(device)).to(device)
        if args[-1] == "tcn":
            discriminator: Discriminator = torch.load("DiscriminatorTCN.torch", weights_only = False, map_location=torch.device(device)).to(device)
            generator: Generator = torch.load("GeneratorTCN.torch", weights_only = False, map_location=torch.device(device)).to(device)
        elif args[-1] == "linear":
            discriminator: Discriminator = torch.load("DiscriminatorLinear.torch", weights_only = False, map_location=torch.device(device)).to(device)
            generator: Generator = torch.load("GeneratorLinear.torch", weights_only = False, map_location=torch.device(device)).to(device)
        discriminator = discriminator.eval()
        generator = generator.eval()
        if len(args) == 5 or args[5] == "look":
            preds = discriminate(discriminator, dataset_x, 400)
            for i, val in dataset_x.iterrows():
                label = df_x_label.loc[i]
                result = preds[i]
                if random.randint(0,1) == -1:
                    # Sample noise as generator input
                    z = torch.tensor(np.random.normal(0, 1, (30,)))
                    gen = generator(z).detach()
                    result_fake = discriminator(gen)
                    print("FAKE", result_fake.item())
                    # print((gen*min_max)+train_min.to_numpy())
                else:
                    print(label, result.item())
                    # print(val_f_old)
        elif args[5] == "thresh":
            X = "Validation" if args[4] == "val" else "Test"
            # Get predicitons of df_val
            if args[-1] == "tcn":
                preds = discriminateTCN(discriminator, dataset_x, time_window=40)
            elif args[-1] == "lstm":
                preds = discriminate(discriminator, dataset_x, time_window)
                preds = np.mean(preds, axis=1)
                preds = np.squeeze(preds)
            else:
                preds = discriminate(discriminator, dataset_x, time_window)
            best_thresh = metrics.best_validation_threshold(y_x, preds)
            thresh = best_thresh["thresholds"]
            if len(args) == 6 or args[6] == "metrics" or args[6] == "both":
                print(f"{X} AUC: ", metrics.roc_auc_score(y_x, preds))
                print(f"{X} accuracy: ", metrics.accuracy(y_x, preds > thresh))
                print(f"{X} precision: ", metrics.precision_score(y_x, preds > thresh))
                print(f"{X} recall: ", metrics.recall_score(y_x, preds > thresh))
                print(f"{X} f1: ", metrics.f1_score(y_x, preds > thresh))
                print("Tpr: ", best_thresh['tpr'])
                print("Fpr: ", best_thresh['fpr'])
                print()
            if len(args) > 6:
                if args[6] == "matrix" or args[6] == "both":
                    metrics.plot_confusion_matrix(y_x, preds > thresh, name=args[4])
                if args[6] == "curve" or args[6] == "both":
                    metrics.plot_roc_curve(y_x, preds, name=args[4])
                if args[6] == "attacks" or args[6] == "both":
                    if dataset_kind == "2019":
                        if args[4] == "val":
                            lista_attacks = ["BENIGN", "LDAP", "MSSQL", "NetBIOS", "UDPLag", "UDP", "Syn"]
                        elif args[4] == "test":
                            lista_attacks = ["BENIGN", "LDAP", "MSSQL", "NetBIOS", "UDPLag", "UDP", "Syn", "Portmap"]
                    elif dataset_kind == "2017":
                        if args[4] == "val":
                            lista_attacks = [
                                "DoS Hulk",
                                "DoS GoldenEye",
                                "DoS slowloris",
                                "DoS Slowhttptest",
                                "Heartbleed",
                            ]
                        elif args[4] == "test":
                            lista_attacks = [
                                "Bot",
                                "PortScan",
                                "DDoS"
                            ]
                    for i in lista_attacks:
                        idxs = df_x_label[df_x_label==i].index
                        y_x_i = y_x.loc[idxs]
                        preds_i = [preds[i] for i in idxs.tolist()]
                        print(f"{X} accuracy of {i}: ", metrics.accuracy(y_x_i, preds_i > thresh))
        elif args[5] == "inference":
            X = "Validation" if args[4] == "val" else "Test"
            # Get predicitons of df_val
            if args[-1] == "tcn":
                preds = discriminateTCN(discriminator, dataset_x, time_window=40, batch_size=1)
            elif args[-1] == "lstm":
                preds = discriminate(discriminator, dataset_x, time_window, batch_size=1, lim=1000)
            else:
                preds = discriminate(discriminator, dataset_x, time_window, batch_size=1)
    elif len(args) > 4 and args[4] == "minmax":
        # Mapeando endereços ip para valores inteiros
        df_train["Source IP"] = df_train["Source IP"].map(lambda x: int(IPv4Address(x)))
        df_train["Destination IP"] = df_train["Destination IP"].map(lambda x: int(IPv4Address(x)))
        print(df_train.min())
        print()
        print(df_train.max())
        

if __name__ == '__main__':
    main()
