import numpy as np
import pandas as pd

from torch.autograd import Variable

import torch.nn as nn
import torch
import src.metrics as metrics
from sklearn.metrics import roc_auc_score
from src.early_stop import EarlyStopping
from src.self_attention import SelfAttention
from src.self_attention import PositionalEncoding
from torch.nn.functional import pad

from src.lstm import LSTM

cuda = True if torch.cuda.is_available() else False

class BlockSelfAttention(nn.Module):
    def __init__(self, input_shape, embed_dim, heads, dropout):
       super(BlockSelfAttention, self).__init__()
       self.pos = PositionalEncoding(embed_dim, 0)
       self.mha = SelfAttention(embed_dim, heads)
       self.dropout = nn.Dropout(dropout)
       self.norm = nn.LayerNorm(input_shape)

    def forward(self, x):
        #print("BLOCK SA")
        #print(" ", x.shape)
        x = self.pos(x)
        #print(" ", x.shape)
        attention = self.mha(x)
        #print(" ", attention.shape)
        drop = self.dropout(attention)
        z = x + drop
        #print(" ", z.shape)
        normalized = self.norm(z)
        return normalized



def block_mlp(in_feat, out_feat, leak = 0.0):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.LeakyReLU(negative_slope=leak, inplace=True))
    return layers

class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim, dropout=0.2):
        super(Generator, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        # TODO: alterar arquitetura para usar TCN e self attention
        self.fc1 = nn.Sequential(
            *block_mlp(latent_dim, 40, leak=0.1),
        )
        self.lstm = LSTM(40, 40)
        self.fc2 = nn.Sequential(
            nn.Linear(40, int(np.prod(data_shape))),
            nn.Sigmoid(),
        )
        self.flat = nn.Flatten(0)

    def forward(self, z):
        data = self.fc1(z)
        data = self.lstm(data)
        data = self.fc2(data)
        return data

class Discriminator(nn.Module):
    def __init__(self, data_shape, dropout=0.2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(
            *block_mlp(int(np.prod(data_shape)), 40, leak=0.1),
        )
        self.lstm = LSTM(40, 40)
        self.fc2 = nn.Sequential(
            nn.Linear(1600, 1)
        )
        self.flat = nn.Flatten(0)

    def forward(self, data):
        z1 = self.fc1(data)
        za = self.lstm(z1)
        
        za_pad = za
        if za.shape[0] != 40:
            target_size = (40, 40)
            pad_rows = target_size[0] - za.shape[0]
            za_pad = pad(za_pad, (0, 0, 0, pad_rows), value=0)

        zaf = self.flat(za_pad)
        val = self.fc2(zaf)
        return val

def Train(df_train: pd.DataFrame, lrd, lrg, epochs, df_val: pd.DataFrame = None, y_val: pd.Series = None, n_critic = 5, 
    clip_value = 1, latent_dim = 30, optim = torch.optim.RMSprop, wdd = 1e-2, wdg = 1e-2, early_stopping: EarlyStopping = None, dropout=0.2) -> tuple[Generator, Discriminator]:
    
    data_ex = df_train.iloc[0]
    # print(data_ex)
    data_shape = data_ex.shape
    # print('data_shape', data_shape)
    print_each_n = 3000
    
    # Initialize generator and discriminator
    generator = Generator(data_shape, latent_dim, dropout)
    discriminator = Discriminator(data_shape, dropout)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        
    # Optimizers
    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wdd)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wdg)
    
    Tensor: type[torch.FloatTensor] = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(epochs):
        for i, datum in df_train.iterrows():
            serial = df_train.iloc[max(0, i-39):i+1]
            # Configure input
            real_data = Variable(torch.from_numpy(serial.to_numpy()).type(Tensor))
            # print("real data shape", real_data.shape)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (min(40, i+1),latent_dim))))
            # print("latent shape", z.shape)

            # Generate a batch of images
            fake_data = generator(z).detach()
            # fake_data = fake_data.unsqueeze(0)
            # Adversarial loss
            # print('real_data shape', real_data.shape)
            # print("fake_data shape", fake_data.shape)
            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data)
            # print("disc real", disc_real)
            # print("disc fake", disc_fake)
            loss_D_true = torch.mean(disc_real)
            loss_D_fake = torch.mean(disc_fake)
            loss_D = -loss_D_true + loss_D_fake

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_data = generator(z)
                # gen_data = gen_data.unsqueeze(0)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_data))

                loss_G.backward()
                optimizer_G.step()
                if i % print_each_n == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D true loss: %f] [D fake loss: %f] [G loss: %f]"
                        % (epoch+1, epochs, batches_done % len(df_train), len(df_train), loss_D_true.item(), loss_D_fake.item(), loss_G.item())
                    )
                    # print("fake: ", fake_data)
                    # print("real: ", real_data)
            batches_done += 1
        if df_val is not None and y_val is not None:
            discriminator.eval()
            preds = discriminate(discriminator, df_val)
            best_thresh = metrics.best_validation_threshold(y_val, preds)
            thresh = best_thresh["thresholds"]
            auc_score = roc_auc_score(y_val, preds)
            print("\nValidation accuracy: ", metrics.accuracy(y_val, preds > thresh))
            print("AUC score: ", auc_score, "\n")
            # Mecanismo de early stopping
            if early_stopping is not None:
                early_stopping(auc_score, discriminator, generator)
                if early_stopping.early_stop:
                    print(f'Stopped by early stopping at epoch {epoch+1}')
                    discriminator = torch.load('checkpoint.pt', weights_only = False)
                    generator = torch.load('checkpoint2.pt', weights_only = False)
                    return generator, discriminator
            discriminator.train()
        

    
    discriminator = torch.load('checkpoint.pt', weights_only = False)
    generator = torch.load('checkpoint2.pt', weights_only = False)
    return generator, discriminator

def discriminate(discriminator: Discriminator, df: pd.DataFrame) -> list:
    Tensor: type[torch.FloatTensor] = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    scores = []
    for i, datum in df.iterrows():
        serial = df.iloc[max(0, i-39):i+1]
        data = Variable(torch.from_numpy(serial.to_numpy()).type(Tensor))
        score = discriminator(data).cpu().detach().numpy()
        print(f"\r[Validating] [Sample {i} / {len(df)}] [Score {score}]", end="")
        scores.append(-score)
    print()
    return scores