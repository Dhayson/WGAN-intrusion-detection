import numpy as np
import pandas as pd

import torch.nn as nn
import torch

from src.early_stop import EarlyStopping
from src.wgan.wgan import Generator, Discriminator, WganTrain
from src.into_dataloader import IntoDatasetNoTime

cuda = True if torch.cuda.is_available() else False

def block_mlp(in_feat, out_feat):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.ReLU(inplace=True))
    return layers

class GeneratorLinear(Generator):
    def __init__(self, data_shape, latent_dim):
        super(Generator, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        # TODO: alterar arquitetura para usar TCN e self attention
        self.model = nn.Sequential(
            *block_mlp(latent_dim, 80),
            *block_mlp(80, 80),
            *block_mlp(80, int(np.prod(data_shape))),
        )

    def forward(self, z, do_print=False):
        data = self.model(z)
        return data

class DiscriminatorLinear(Discriminator):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            *block_mlp(int(np.prod(data_shape)), 80),
            *block_mlp(80, 80),
            *block_mlp(80, 15),
            nn.Linear(15, 1)
        )

    def forward(self, data, do_print=False):
        validity = self.model(data)
        return validity

def TrainLinear(dataset_train: pd.DataFrame, lrd, lrg, epochs, dataset_val: pd.DataFrame = None, y_val: pd.Series = None, n_critic = 5, 
    clip_value = 1, latent_dim = 30, optim = torch.optim.RMSprop, wdd = 1e-2, wdg = 1e-2, early_stopping: EarlyStopping = None, dropout=0.2,
    print_each_n = 20, time_window = 40, batch_size=5, data_len=40
    ) -> tuple[GeneratorLinear, DiscriminatorLinear]:
    data_shape = (data_len,)
    # Initialize generator and discriminator
    generator = GeneratorLinear(data_shape, latent_dim)
    discriminator = DiscriminatorLinear(data_shape)
    return WganTrain(dataset_train, generator, discriminator, lrd, lrg, epochs, dataset_val, y_val, n_critic, clip_value, latent_dim, optim,
              wdd, wdg, early_stopping, dropout, print_each_n, time_window, batch_size)