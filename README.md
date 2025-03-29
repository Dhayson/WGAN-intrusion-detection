# Como rodar:

## Treinar modelos

### Self Attention

#### Dataset CIC-DDoS-2019
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 train sa
```

#### Dataset CIC-IDS-2017
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 train sa
```

### TCN

#### Dataset CIC-DDoS-2019
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 train tcn
```

#### Dataset CIC-IDS-2017
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 train tcn
```

### LSTM

#### Dataset CIC-DDoS-2019
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 train lstm
```

#### Dataset CIC-IDS-2017
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 train lstm
```

## Executar modelos (após o treinamento)

<p>Serão criados arquivos out_curve_[conjunto].png e out_matrix_[conjunto].png, onde [conjunto] será val ou test
dependendo do que for escolhido. Outras métricas serão printadas no terminal.</p>

### Self Attention

#### Dataset CIC-DDoS-2019
##### Conjunto de validação
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 val thresh both sa
```
##### Conjunto de teste
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 test thresh both sa
```

#### Dataset CIC-IDS-2017
##### Conjunto de validação
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 val thresh both sa
```
##### Conjunto de teste
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 test thresh both sa
```


### TCN

#### Dataset CIC-DDoS-2019
##### Conjunto de validação
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 val thresh both tcn
```
##### Conjunto de teste
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 test thresh both tcn
```

#### Dataset CIC-IDS-2017
##### Conjunto de validação
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 val thresh both tcn
```
##### Conjunto de teste
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 test thresh both tcn
```

### LSTM

#### Dataset CIC-DDoS-2019
##### Conjunto de validação
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 val thresh both lstm
```
##### Conjunto de teste
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 test thresh both lstm
```

#### Dataset CIC-IDS-2017
##### Conjunto de validação
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 val thresh both lstm
```
##### Conjunto de teste
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 test thresh both lstm
```

## Realizar tunagem de parâmetros

### Self Attention

#### Dataset CIC-DDoS-2019
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 tune sa
```

#### Dataset CIC-IDS-2017
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 tune sa
```

### TCN

#### Dataset CIC-DDoS-2019
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 optuna tcn
```
#### Dataset CIC-IDS-2017
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 optuna tcn
```

### LSTM

#### Dataset CIC-DDoS-2019
```console
python3 main.py 2019 dataset_filtered/Day1 dataset_filtered/Day2 optuna lstm
```
#### Dataset CIC-IDS-2017
```console
python3 main.py 2017 dataset_alternative_filtered/Day1 dataset_alternative_filtered/Day3 dataset_alternative_filtered/Day5 optuna lstm
```

