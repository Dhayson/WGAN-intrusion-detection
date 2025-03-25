import pandas as pd
import numpy as np
import os
from numpy.random import RandomState

def GetDataset(path: str, rs: RandomState, dataset_format="csv", label="both", filter=True, do_print = True):
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
            if do_print:
                print(f"Reading {path}/{file}")
            if filter:
                df_aux = pd.read_csv(f'{path}/{file}',
                    usecols =[
                    ' Label', ' Timestamp', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Flow Duration',
                    ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
                    ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max',
                    ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', 'Fwd Packets/s', ' Bwd Packets/s',
                    ' Flow IAT Max', ' Flow IAT Min',' Flow IAT Mean', ' Flow IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', ' Fwd IAT Std',
                    'Fwd IAT Total', ' Fwd IAT Mean', ' Bwd IAT Max', ' Bwd IAT Min', ' Bwd IAT Mean', ' Bwd IAT Std', 'Bwd IAT Total', 'Fwd PSH Flags',
                    ' Bwd PSH Flags', ' Fwd Header Length', ' Bwd Header Length'
                    ],
                    parse_dates=[' Timestamp']
                )
            else:
                df_aux = pd.read_csv(f'{path}/{file}', parse_dates=['Timestamp'])
            if mixed_dtypes := {c: dtype for c in df_aux.columns if (dtype := pd.api.types.infer_dtype(df_aux[c])).startswith("mixed")}:
                raise TypeError(f"Dataframe has one more mixed dtypes: {mixed_dtypes}")
        elif dataset_format == "parquet":
            df_aux = pd.read_parquet(f'{path}/{file}')
        df_aux.columns = df_aux.columns.str.strip()
        
        # Limitar às entradas benignas e 11342 por tipo, para não utilizar toda a RAM
        df_aux_list = []
        
        # Dropar na e inf
        df_aux = df_aux.replace([-np.inf, np.inf], np.nan)
        df_aux = df_aux.dropna(axis = 0)
        
        if label == "ben" or label == "both":
            df_aux_ben = df_aux[df_aux["Label"] == BENIGN]
            if len(df_aux_ben) > 0 and do_print:
                print(f"found kind {BENIGN} in {path}/{file}")
            df_aux_list.append(df_aux_ben)
        
        if label == "mal" or label == "both":
            for kind in ["Syn", "DrDoS_UDP", "UDP-lag", "DrDoS_MSSQL", "DrDoS_NetBIOS", "DrDoS_LDAP", "UDP", "UDPLag", "MSSQL", "NetBIOS", "LDAP", "Portmap"]:
                df_aux_mal = df_aux[df_aux["Label"] == kind]
                if len(df_aux_mal) > 0 and do_print:
                    print(f"found kind {kind} in {path}/{file}")
                df_aux_mal = df_aux_mal.head(n=min(11342, len(df_aux_mal)))
                df_aux_list.append(df_aux_mal)
        
        df_list.append(pd.concat(df_aux_list))
        # except:
        #     print(f"Can't read file {path}/{file}")
    df = pd.concat(df_list)
    return df