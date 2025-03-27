from src.import_dataset2 import GetDataset
from src.dataset_split import SplitDataset
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.random import RandomState

def DescartarDuplicatas(dataset: pd.DataFrame, do_print = False):
    initial_len = dataset.shape[0]
    dataset = dataset.drop_duplicates()
    if do_print:
        print(f'Tamanho inicial: {initial_len}, tamanho final {dataset.shape[0]} | Descartadas {initial_len - dataset.shape[0]} duplicadas')
        print()
    return dataset

# OBS: as seguintes colunas não estão presentes no dataset:
# [' Source IP', ' Source Port', ' Protocol', ' Timestamp', ' Destination IP']

def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    BENIGN = "BENIGN"
    prefix = "MachineLearningCVE/"

    mon_file = "Monday-WorkingHours.pcap_ISCX.csv"
    tue_file = "Tuesday-WorkingHours.pcap_ISCX.csv"
    wed_file = "Wednesday-workingHours.pcap_ISCX.csv"
    thu_morning_file = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    thu_afternoon_file = "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    fri_morning_file = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    fri_afternoon_file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    fri_afternoon_portscan_file = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

    # Para cada arquivo, extrair até 50k instâncias benignas e até 10k instâncias malignas de cada tipo
    
    # Monday: all benign
    df_mon_ben = GetDataset(prefix + mon_file, rs, DATASET_FORMAT, "ben")
    df_mon_ben = DescartarDuplicatas(df_mon_ben, do_print=True)
    df_mon_ben.to_csv("cicddos_2017/Day1/day1_benign.csv", encoding='utf-8', index=False)

    # Tuesday: both
    df_tue_ben = GetDataset(prefix + tue_file, rs, DATASET_FORMAT, "ben")
    df_tue_ben = DescartarDuplicatas(df_tue_ben, do_print=True)
    df_tue_ben.to_csv("cicddos_2017/Day2/day2_benign.csv", encoding='utf-8', index=False)

    df_tue_mal = GetDataset(prefix + tue_file, rs, DATASET_FORMAT, "mal")
    df_tue_mal = DescartarDuplicatas(df_tue_mal, do_print=True)
    df_tue_mal.to_csv("cicddos_2017/Day2/day2_attack.csv", encoding='utf-8', index=False)

    # Wednesday: both
    df_wed_ben = GetDataset(prefix + wed_file, rs, DATASET_FORMAT, "ben")
    df_wed_ben = DescartarDuplicatas(df_wed_ben, do_print=True)
    df_wed_ben.to_csv("cicddos_2017/Day3/day3_benign.csv", encoding='utf-8', index=False)

    df_wed_mal = GetDataset(prefix + wed_file, rs, DATASET_FORMAT, "mal")
    df_wed_mal = DescartarDuplicatas(df_wed_mal, do_print=True)
    df_wed_mal.to_csv("cicddos_2017/Day3/day3_attack.csv", encoding='utf-8', index=False)

    # Thursday: both
    df_thu_ben = GetDataset(prefix + thu_morning_file, rs, DATASET_FORMAT, "ben")
    df_thu_ben_2 = GetDataset(prefix + thu_afternoon_file, rs, DATASET_FORMAT, "ben") 
    df_thu_ben = pd.concat([df_thu_ben, df_thu_ben_2], ignore_index=True)
    df_thu_ben = DescartarDuplicatas(df_thu_ben, do_print=True)
    df_thu_ben.to_csv("cicddos_2017/Day4/day4_benign.csv", encoding='utf-8', index=False)

    df_thu_mal = GetDataset(prefix + thu_morning_file, rs, DATASET_FORMAT, "mal")
    df_thu_mal_2 = GetDataset(prefix + thu_afternoon_file, rs, DATASET_FORMAT, "mal")
    df_thu_mal = pd.concat([df_thu_mal, df_thu_mal_2], ignore_index=True)
    df_thu_mal = DescartarDuplicatas(df_thu_mal, do_print=True)
    df_thu_mal.to_csv("cicddos_2017/Day4/day4_attack.csv", encoding='utf-8', index=False)

    # Friday: both
    df_fri_ben = GetDataset(prefix + fri_morning_file, rs, DATASET_FORMAT, "ben")
    #df_fri_ben_2 = GetDataset(prefix + fri_afternoon_file, rs, DATASET_FORMAT, "ben")
    #df_fri_ben_3 = GetDataset(prefix + fri_afternoon_portscan_file, rs, DATASET_FORMAT, "ben")
    #df_fri_ben = pd.concat([df_fri_ben, df_fri_ben_2,df_fri_ben_3], ignore_index=True)
    df_fri_ben = DescartarDuplicatas(df_fri_ben, do_print=True)
    df_fri_ben.to_csv("cicddos_2017/Day5/day5_benign.csv", encoding='utf-8', index=False)

    df_fri_mal = GetDataset(prefix + fri_morning_file, rs, DATASET_FORMAT, "mal")
    df_fri_mal_2 = GetDataset(prefix + fri_afternoon_file, rs, DATASET_FORMAT, "mal")
    df_fri_mal_3 = GetDataset(prefix + fri_afternoon_portscan_file, rs, DATASET_FORMAT, "mal")
    df_fri_mal = pd.concat([df_fri_mal, df_fri_mal_2, df_fri_mal_3], ignore_index=True)
    df_fri_mal = DescartarDuplicatas(df_fri_mal, do_print=True)
    df_fri_mal.to_csv("cicddos_2017/Day5/day5_attack.csv", encoding='utf-8', index=False)

if __name__ == '__main__':
    main()