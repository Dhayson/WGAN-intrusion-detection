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
    def __init__(self, data_shape, latent_dim, heads, internal_dim, dropout=0.2, seq_dim=None, sa_layers = 1):
        super(GeneratorSA, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Sequential(
            *block_mlp(latent_dim, internal_dim, leak=0.1),
        )
        if sa_layers == 1:
            self.sa = BlockSelfAttention(internal_dim, heads, dropout, seq_dim)
        elif sa_layers == 2:
            self.sa = nn.Sequential(
                BlockSelfAttention(internal_dim, heads, dropout, seq_dim),
                BlockSelfAttention(internal_dim, heads, dropout, seq_dim)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(internal_dim, int(data_shape[1])),
            nn.Sigmoid()
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
    def __init__(self, data_shape, time_window, heads, internal_dim, dropout=0.2, seq_dim=None, sa_layers = 1):
        super(DiscriminatorSA, self).__init__()

        self.fc1 = nn.Sequential(
            *block_mlp(int(data_shape[1]), internal_dim, leak=0.1),
        )
        if sa_layers == 1:
            self.sa = BlockSelfAttention(internal_dim, heads, dropout, seq_dim)
        elif sa_layers == 2:
            self.sa = nn.Sequential(
                BlockSelfAttention(internal_dim, heads, dropout, seq_dim),
                BlockSelfAttention(internal_dim, heads, dropout, seq_dim)
            )
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
    print_each_n = 20, time_window = 40, batch_size=5, headsd=40, embedd=400, headsg=40, embedg=400, data_len=40, return_auc = False, sa_layers = 1
    ) -> tuple[GeneratorSA, DiscriminatorSA]:
    assert(embedd % headsd == 0)
    assert(embedg % headsg == 0)
    data_shape = (time_window, data_len)
    # Initialize generator and discriminator
    generator = GeneratorSA(data_shape, latent_dim, headsg, embedg, dropout=dropout, seq_dim=time_window, sa_layers=sa_layers)
    discriminator = DiscriminatorSA(data_shape, time_window, headsd, embedd, dropout=dropout, seq_dim=time_window, sa_layers=sa_layers)
    
    return WganTrain(dataset_train, generator, discriminator, lrd, lrg, epochs, dataset_val, y_val, n_critic, clip_value, latent_dim, optim,
              wdd, wdg, early_stopping, dropout, print_each_n, time_window, batch_size, return_auc=return_auc)

# Trial: 8 finished with auc score 0.9997368395421042
# Parameters: lrd:0.0007074207502579864, lrg:0.0003427041916020818, n_critic:4, clip_value:0.5036187305772312
# latent_dim:15, optim:<class 'torch.optim.adam.Adam'>, wdd:0.0017472655758194694, wdg:0.008333067108096701, dropout:0.19560173729322383
# time_window:77, batch_size:10, headsd:62, embedd:186
# headsg:24, embedg:72
def RunModelSelfAttention2019(dataset_train: IntoDataset, dataset_val: IntoDataset, y_val):
    generator_sa, discriminator_sa = TrainSelfAttention(dataset_train, lrd=0.0007074207502579864, lrg=0.0003427041916020818, epochs=50, 
                dataset_val=dataset_val, y_val=y_val, wdd=0.0017472655758194694, wdg=0.008333067108096701, clip_value = 0.5036187305772312, optim=torch.optim.Adam,
                early_stopping=EarlyStopping(15, 0), dropout=0.19560173729322383, latent_dim=15, batch_size=10, n_critic=4,
                time_window=77, headsd=62, embedd=186, headsg=24, embedg=72)
    torch.save(generator_sa, "GeneratorSA.torch")
    torch.save(discriminator_sa, "DiscriminatorSA.torch")
    

# Trial: 2 finished with auc score 0.9719520610724199
# Parameters: lrd:0.0014838446689901075, lrg:0.0004927052176484955, n_critic:7, clip_value:0.6218987845482782
# latent_dim:14, optim:<class 'torch.optim.adam.Adam'>, wdd:0.0010562643134023315, wdg:0.009849087440102025, dropout:0.25550529714759485
# time_window:80, batch_size:9, headsd:44, embedd:176
# headsg:20, embedg:40
def RunModelSelfAttention2017(dataset_train: IntoDataset, dataset_val: IntoDataset, y_val):
    generator_sa, discriminator_sa = TrainSelfAttention(dataset_train, lrd=0.0014838446689901075, lrg=0.0004927052176484955, epochs=50, 
                dataset_val=dataset_val, y_val=y_val, wdd=0.0010562643134023315, wdg=0.009849087440102025, clip_value = 0.6218987845482782, optim=torch.optim.Adam,
                early_stopping=EarlyStopping(15, 0), dropout=0.25550529714759485, latent_dim=14, batch_size=9, n_critic=7,
                time_window=80, headsd=44, embedd=176, headsg=20, embedg=40)
    torch.save(generator_sa, "GeneratorSA.torch")
    torch.save(discriminator_sa, "DiscriminatorSA.torch")

def TrainSelfAttentionGP(dataset_train: IntoDataset, lrd, lrg, epochs, dataset_val: IntoDataset = None, y_val: pd.Series = None, n_critic = 5, 
    clip_value = None, latent_dim = 30, optim = torch.optim.RMSprop, wdd = 1e-2, wdg = 1e-2, early_stopping: EarlyStopping = None, dropout=0.2,
    print_each_n = 20, time_window = 40, batch_size=5, headsd=40, embedd=400, headsg=40, embedg=400, data_len=40, return_auc = False, sa_layers = 1, lambda_penalty = 0.05
    ) -> tuple[GeneratorSA, DiscriminatorSA]:
    assert(embedd % headsd == 0)
    assert(embedg % headsg == 0)
    data_shape = (time_window, data_len)
    # Initialize generator and discriminator
    generator = GeneratorSA(data_shape, latent_dim, headsg, embedg, dropout=dropout, seq_dim=time_window, sa_layers=sa_layers)
    discriminator = DiscriminatorSA(data_shape, time_window, headsd, embedd, dropout=dropout, seq_dim=time_window, sa_layers=sa_layers)
    
    return WganTrain(dataset_train, generator, discriminator, lrd, lrg, epochs, dataset_val, y_val, n_critic, clip_value, latent_dim, optim,
              wdd, wdg, early_stopping, dropout, print_each_n, time_window, batch_size, return_auc=return_auc, lambda_penalty=lambda_penalty)

# Parameters: lrd:0.0013172236999405948, lrg:0.00042730691645367566, n_critic:6, clip_value: 0.525001342151131
# latent_dim:10, optim:<class 'torch.optim.Adam'>, wdd:0.001139442745846273, wdg:0.008386851859453001, dropout:0.15836459213389872
# time_window:79, batch_size:4, headsd:56, embedd:112
# headsg:22, embedg:44, lambda_penalty0.07531231424904279
def RunModelSelfAttentionGP2019(dataset_train: IntoDataset, dataset_val: IntoDataset, y_val):
    generator_sa, discriminator_sa = TrainSelfAttentionGP(dataset_train, lrd=0.0013172236999405948, lrg=0.00042730691645367566, epochs=50, 
                dataset_val=dataset_val, y_val=y_val, wdd=0.001139442745846273, wdg=0.008386851859453001, clip_value = 0.525001342151131, optim=torch.optim.Adam,
                early_stopping=EarlyStopping(15, 0), dropout=0.15836459213389872, latent_dim=10, batch_size=4, n_critic=6,
                time_window=79, headsd=56, embedd=112, headsg=22, embedg=44, lambda_penalty=0.07531231424904279)
    torch.save(generator_sa, "GeneratorSA.torch")
    torch.save(discriminator_sa, "DiscriminatorSA.torch")
    

# Parameters: lrd:0.001512703094318173, lrg:0.00020080466161518008, n_critic:7, clip_value:0.5442203304851734
# latent_dim:8, optim:<class 'torch.optim.Adam'>, wdd:0.0019392946765071106, wdg:0.009197830177494612, dropout:0.2986744001633296
# time_window:60, batch_size:6, headsd:72, embedd:144
# headsg:26, embedg:78, lambda_penalty:0.0765723290179939
def RunModelSelfAttentionGP2017(dataset_train: IntoDataset, dataset_val: IntoDataset, y_val):
    generator_sa, discriminator_sa = TrainSelfAttentionGP(dataset_train, lrd=0.001512703094318173, lrg=0.00020080466161518008, epochs=50, 
                dataset_val=dataset_val, y_val=y_val, wdd=0.0019392946765071106, wdg=0.009197830177494612, clip_value = 0.5442203304851734, optim=torch.optim.Adam,
                early_stopping=EarlyStopping(15, 0), dropout=0.2986744001633296, latent_dim=8, batch_size=6, n_critic=7,
                time_window=60, headsd=72, embedd=144, headsg=26, embedg=78, lambda_penalty=0.0765723290179939)
    torch.save(generator_sa, "GeneratorSA.torch")
    torch.save(discriminator_sa, "DiscriminatorSA.torch")