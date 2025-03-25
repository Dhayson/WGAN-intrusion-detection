# Sistema de detecção de intrusão com WGAN e Self-Attention

## Como rodar

### Treinamento:

```console
$  python3 main.py dataset_filtered/Day1 dataset_filtered/Day2 train sa
```

### Métricas no conjunto de validação:


```console
$ python3 main.py dataset_filtered/Day1 dataset_filtered/Day2 val thresh both
```

#### Métricas dos ataques:
```console
$ python3 main.py dataset_filtered/Day1 dataset_filtered/Day2 val thresh attacks
```

### Métricas no conjunto de teste:


```console
$ python3 main.py dataset_filtered/Day1 dataset_filtered/Day2 test thresh both
```

#### Métricas dos ataques:
```console
$ python3 main.py dataset_filtered/Day1 dataset_filtered/Day2 test thresh attacks
```

