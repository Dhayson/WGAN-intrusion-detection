import src.import_dataset
import sys
import numpy as np
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def SplitDataset(df_day_1: pd.DataFrame, df_day_2: pd.DataFrame, rs: RandomState) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide entre treino, validação e teste segundo o artigo

    Args:
        df_day_1 (pd.DataFrame): Dataset do primeiro dia
        df_day_2 (pd.DataFrame): Dataset do segundo dia
    """
    # Para o treino, queremos apenas dados benignos do segundo dia
    df_train = df_day_2[df_day_2['Label'] == "Benign"].sample(frac=0.8, random_state=rs).sort_index()
    
    # Para validação, queremos os demais dados benignos e os dados malignos do segundo dia
    df_val = df_day_2.drop(df_train.index)
    df_val_ben = df_val[df_val["Label"] == "Benign"]
    df_val_mal = df_val[df_val["Label"].isin(["Syn", "DrDoS_UDP", "UDP-lag", "DrDoS_MSSQL", "DrDoS_NetBIOS", "DrDoS_LDAP"])]
    df_val_mal = df_val_mal.sample(
        frac=1, random_state=rs
    )
    df_val = pd.concat([df_val_ben, df_val_mal])
    
    
    # Para o teste, queremos os dados benignos do primeiro dia e uma quantidade igual de dados malignos,
    # com os mesmos ataques usados anteriormente + Portmap
    df_test_ben = df_day_1[df_day_1['Label']=='Benign']
    
    # Prioriza os casos 0 day
    df_val_port = df_day_1[df_day_1["Label"].isin(["Portmap"])]
    # ALERTA: Dados estão rotulados de forma diferente entre os dias
    df_val_mal = df_day_1[df_day_1["Label"].isin(["Syn", "UDP", "UDPLag", "MSSQL", "etBIOS", "LDAP"])]
    df_val_mal = df_val_mal.sample(
        n=df_test_ben.__len__() - df_val_port.__len__(), random_state=rs
    )
    df_test = pd.concat([df_test_ben, df_val_mal, df_val_port])
    
    
    return (df_train, df_val, df_test)
    