import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def GetDataset(path: str, type="csv"):
    """Lê o dataset inteiro, de um diretório, e insere todas as entradas em um dataframe

    Args:
        path (str): Caminho para o diretório

    Returns:
        Dataframe
    """
    df_list = []
    for file in os.listdir(path):
        if type == "csv":
            df_aux = pd.read_csv(f'{path}/{file}')
        elif type == "parquet":
            df_aux = pd.read_parquet(f'{path}/{file}')
        df_list.append(df_aux)
    df = pd.concat(df_list, ignore_index=True)
    return df