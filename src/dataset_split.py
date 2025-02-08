from numpy.random import RandomState
import pandas as pd


def SplitDataset(df_day_1: pd.DataFrame, df_day_2: pd.DataFrame, rs: RandomState, dataset_format="csv") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide entre treino, validação e teste segundo o artigo

    Args:
        df_day_1 (pd.DataFrame): Dataset do primeiro dia
        df_day_2 (pd.DataFrame): Dataset do segundo dia
    """
    BENIGN = "BENIGN"
    if dataset_format == "parquet":
        BENIGN = "Benign"
    
    # Para o treino, queremos apenas dados benignos do segundo dia
    df_train = df_day_2[df_day_2['Label'] == BENIGN].sample(frac=0.8, random_state=rs).sort_index()
    
    # Para validação, queremos os demais dados benignos e os dados malignos do segundo dia
    df_val_mal = df_day_2[df_day_2["Label"].isin(["Syn", "UDP", "UDPLag", "MSSQL", "NetBIOS", "LDAP"])]
    df_val_ben = df_day_2.drop(df_train.index)
    df_val_ben = df_val_ben[df_val_ben["Label"] == BENIGN]
    df_val = pd.concat([df_val_ben, df_val_mal])
    
    
    # Para o teste, queremos os dados benignos do primeiro dia e uma quantidade igual de dados malignos,
    # com os mesmos ataques usados anteriormente + Portmap
    # ALERTA: Dados estão rotulados de forma diferente entre os dias no dataset original, porém isso foi corrigido
    attack_types = ["Syn", "UDP", "UDPLag", "MSSQL", "NetBIOS", "LDAP", "Portmap"]
    df_test_mal = pd.DataFrame()
    for at in attack_types:
        max_entries = 8021
        if at == "Portmap":
            max_entries = 8022
            
        df_test_mal = pd.concat([
            df_test_mal,
            df_day_1[df_day_1["Label"] == at].sample(
                n=min(df_day_1[df_day_1["Label"] == at].__len__(), max_entries), 
                random_state=rs
                )
            ])
    
    df_test_ben = df_day_1[df_day_1['Label']==BENIGN].sample(
        n=min(df_day_1[df_day_1['Label']==BENIGN].__len__(), 50000, df_test_mal.__len__()), random_state=rs
    )
    
    df_test_mal = df_test_mal.sample(
        n=min(df_test_ben.__len__(), 50000), random_state=rs
    )
    df_test = pd.concat([df_test_ben, df_test_mal])
    
    
    return (df_train, df_val, df_test)
    