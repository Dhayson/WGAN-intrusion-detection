import pandas as pd
import numpy as np
import os
from numpy.random import RandomState

def GetDataset(path: str, rs: RandomState, dataset_format="csv"):
    """Lê o dataset inteiro, de um diretório, e insere todas as entradas em um dataframe

    Args:
        path (str): Caminho para o diretório

    Returns:
        Dataframe
    """
    df_list = []
    
    BENIGN = "BENIGN"
    if dataset_format == "parquet":
        BENIGN = "Benign"
        
    for file in os.listdir(path):
        #try:
        if dataset_format == "csv":
            cols = list(pd.read_csv(f'{path}/{file}', nrows=1))
            print(f"Reading file {path}/{file}")
            df_aux = pd.read_csv(f'{path}/{file}',
                usecols =[i for i in cols if not i in [
                'Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', 'SimillarHTTP']],
                parse_dates=[' Timestamp']
            )
            if mixed_dtypes := {c: dtype for c in df_aux.columns if (dtype := pd.api.types.infer_dtype(df_aux[c])).startswith("mixed")}:
                raise TypeError(f"Dataframe has one more mixed dtypes: {mixed_dtypes}")
        elif dataset_format == "parquet":
            df_aux = pd.read_parquet(f'{path}/{file}')
        df_aux.columns = df_aux.columns.str.strip()
        
        # Limitar às entradas benignas e 11342 por tipo, para não utilizar toda a RAM
        df_aux_list = []
        df_aux_ben = df_aux[df_aux["Label"] == BENIGN]
        df_aux_list.append(df_aux_ben)
        
        for kind in ["Syn", "DrDoS_UDP", "UDP-lag", "DrDoS_MSSQL", "DrDoS_NetBIOS", "DrDoS_LDAP", "UDP", "UDPLag", "MSSQL", "NetBIOS", "LDAP", "Portmap"]:
            df_aux_mal = df_aux[df_aux["Label"] == kind]
            df_aux_mal = df_aux_mal.sample(n=min(11342, len(df_aux_mal)), random_state=rs)
            df_aux_list.append(df_aux_mal)
        
        df_list.append(pd.concat(df_aux_list))
        # except:
        #     print(f"Can't read file {path}/{file}")
    df = pd.concat(df_list)
    return df