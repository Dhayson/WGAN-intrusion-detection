import pandas as pd

import torch.nn as nn
import torch
import torch_optimizer

from src.early_stop import EarlyStopping
from src.self_attention import SelfAttention
from src.self_attention import PositionalEncoding

from src.wgan.wgan import Generator, Discriminator, WganTrain
from src.into_dataloader import IntoDataset

cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda else "cpu"

class BlockSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout, seq_dim = None):
       if seq_dim is None:
           seq_dim = heads
       super(BlockSelfAttention, self).__init__()
       self.pos = PositionalEncoding(embed_dim, 0)
       self.mha = SelfAttention(embed_dim, heads, dropout)
       self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # print("BLOCK SA")
        # print(" ", x.shape)
        x = self.pos(x)
        # print(" ", x.shape)
        attention = self.mha(x)
        # print(" ", attention.shape)
        z = x + attention
        # print(" ", z.shape)
        normalized = self.norm(z)
        return normalized



def block_mlp(in_feat, out_feat, leak = 0.0):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.LeakyReLU(negative_slope=leak, inplace=True))
    return layers

class GeneratorSA(Generator):
    def __init__(self, data_shape, latent_dim, heads, internal_dim, dropout=0.2, seq_dim=None):
        super(GeneratorSA, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Sequential(
            *block_mlp(latent_dim, internal_dim, leak=0.1),
        )
        self.sa = BlockSelfAttention(internal_dim, heads, dropout, seq_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(internal_dim, int(data_shape[1])),
            nn.Sigmoid(),
        )
        self.flat = nn.Flatten(0)

    def forward(self, z, do_print=False):
        # print("GENERATOR")
        # print(z.shape)
        z1 = self.fc1(z)
        # print(z1.shape)
        za = self.sa(z1)
        # print(za.shape)
        # zaf = self.flat(za)
        data = self.fc2(za)
        # print(data.shape)
        # print()
        return data

class DiscriminatorSA(Discriminator):
    def __init__(self, data_shape, time_window, heads, internal_dim, dropout=0.2, seq_dim=None):
        super(DiscriminatorSA, self).__init__()

        self.fc1 = nn.Sequential(
            *block_mlp(int(data_shape[1]), internal_dim, leak=0.1),
        )
        self.sa = BlockSelfAttention(internal_dim, heads, dropout, seq_dim)
        self.flat = nn.Flatten(1)
        self.fc2 = nn.Sequential(
            nn.Linear(internal_dim*time_window, 1)
        )

    def forward(self, data, do_print=False):
        if do_print:
            print("DISCRIMINATOR")
            print(data.shape)
        z1 = self.fc1(data)
        if do_print:
            print(z1.shape)
        za = self.sa(z1)
        if do_print:
            print(za.shape)
        zaf = self.flat(za)
        if do_print:
            print(zaf.shape)
        val = self.fc2(zaf)
        if do_print:
            print(val.shape)
            print()
        return val
    
def TrainSelfAttention(dataset_train: IntoDataset, lrd, lrg, epochs, dataset_val: IntoDataset = None, y_val: pd.Series = None, n_critic = 5, 
    clip_value = 1, latent_dim = 30, optim = torch.optim.RMSprop, wdd = 1e-2, wdg = 1e-2, early_stopping: EarlyStopping = None, dropout=0.2,
    print_each_n = 20, time_window = 40, batch_size=5, headsd=40, embedd=400, headsg=40, embedg=400, data_len=40, return_auc = False
    ) -> tuple[GeneratorSA, DiscriminatorSA]:
    assert(embedd % headsd == 0)
    assert(embedg % headsg == 0)
    data_shape = (time_window, data_len)
    # Initialize generator and discriminator
    generator = GeneratorSA(data_shape, latent_dim, headsg, embedg, dropout=dropout, seq_dim=time_window)
    discriminator = DiscriminatorSA(data_shape, time_window, headsd, embedd, dropout=dropout, seq_dim=time_window)
    
    return WganTrain(dataset_train, generator, discriminator, lrd, lrg, epochs, dataset_val, y_val, n_critic, clip_value, latent_dim, optim,
              wdd, wdg, early_stopping, dropout, print_each_n, time_window, batch_size, return_auc=return_auc)
        
def RunModelSelfAttention(dataset_train: IntoDataset, dataset_val: IntoDataset, y_val):
    generator_sa, discriminator_sa = TrainSelfAttention(dataset_train, 5e-5, 5e-5, 10, dataset_val, y_val, wdd=6e-4, wdg=9e-4, clip_value = 0.9, optim=torch_optimizer.Yogi,
                early_stopping=EarlyStopping(3, 0), latent_dim=10, batch_size=128, n_critic=5, time_window=40,
                headsd=40, embedd=240, headsg=40, embedg=240)
    torch.save(generator_sa, "GeneratorSA.torch")
    torch.save(discriminator_sa, "DiscriminatorSA.torch")