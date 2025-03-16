from src.import_dataset import GetDataset
from src.dataset_split import SplitDataset
import sys
import pandas as pd
from numpy.random import RandomState
from main import DescartarDuplicatas

# OBS: o dataset grande não cabe no repositório, mas pode ser baixado em http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/
# NOTA: esse arquivo é executado quando o dataset do dia 2 inteiro não cabe na memória. É possível o extrair em partes, por exemplo,
# deixando o arquivo TFTP.csv para ser extraído após os demais.
# 
# Além disso, pode ser necessário dividir o arquivo em vários para não consumir memória em excesso e crashar o programa. Por exemplo, com:
# $ mlr --csv split -n 1000000 TFTP.csv
# $ rm TFTP.csv
#
# Nesse caso, não estamos interessados no ataque TFTP, então essa etapa é omitida.
def main():
    RANDOM_SEED = 5
    rs = RandomState(RANDOM_SEED)
    DATASET_FORMAT = "csv"
    
    
    BENIGN = "BENIGN"
    if DATASET_FORMAT == "parquet":
        BENIGN = "Benign"
    df_day_2_ben = pd.read_csv("dataset_filtered2/Day2/day2_benign.csv")
    print("Original dataset:")
    print(df_day_2_ben['Label'].value_counts())
    
    
    df_day_2_rest_ben = GetDataset(sys.argv[1], rs, DATASET_FORMAT, "ben")
    df_day_2_ben_final = DescartarDuplicatas(df_day_2_rest_ben, do_print=True)
    
    print("Added dataset:")
    print(df_day_2_rest_ben['Label'].value_counts())
    df_day_2_ben_final = pd.concat([df_day_2_ben, df_day_2_rest_ben])
    
    df_day_2_ben_final = DescartarDuplicatas(df_day_2_ben_final, do_print=True)
    df_day_2_ben_final.to_csv("dataset_filtered2/Day2/day2_benign.csv", encoding='utf-8', index=False)
    print()
    
    
    
    

if __name__ == '__main__':
    main()