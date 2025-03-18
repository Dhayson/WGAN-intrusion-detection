import time
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
from src.into_dataloader import IntoDataset

from torch.utils.data import DataLoader

from src.lstm import LSTM

cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda else "cpu"

class BlockSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
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

class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim, heads, internal_dim, dropout=0.2):
        super(Generator, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        # TODO: alterar arquitetura para usar TCN e self attention
        self.fc1 = nn.Sequential(
            *block_mlp(latent_dim, internal_dim, leak=0.1),
        )
        self.lstm = LSTM(internal_dim, internal_dim, dropout)
        self.fc2 = nn.Sequential(
            nn.Linear(internal_dim, int(data_shape[1])),
            nn.Sigmoid(),
        )
        self.flat = nn.Flatten(0)

    def forward(self, z):
        data = self.fc1(z)
        data = self.lstm(data)
        data = self.fc2(data)
        return data

class Discriminator(nn.Module):
    def __init__(self, data_shape, time_window, heads, internal_dim, dropout=0.2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(
            *block_mlp(int(data_shape[1]), internal_dim, leak=0.1),
        )
        self.lstm = LSTM(internal_dim, internal_dim, dropout)   
        self.fc2 = nn.Sequential(
            nn.Linear(internal_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data, do_print=False):
        if do_print:
            print("DISCRIMINATOR")
            print(data.shape)
        z1 = self.fc1(data)
        if do_print:
            print(z1.shape)
        za = self.lstm(z1)
        if do_print:
            print(za.shape)
        val = self.fc2(za)
        if do_print:
            print(val.shape)
            print()
        return val

def Train(df_train: pd.DataFrame, lrd, lrg, epochs, df_val: pd.DataFrame = None, y_val: pd.Series = None, n_critic = 5, 
    clip_value = 1, latent_dim = 30, optim = torch.optim.RMSprop, wdd = 1e-2, wdg = 1e-2, early_stopping: EarlyStopping = None, dropout=0.2,
    print_each_n = 100, time_window = 40, batch_size=5, do_print = False, step_by_step = False, headsd=40, embedd=400, headsg=40, embedg=400
    ) -> tuple[Generator, Discriminator]:
    assert(embedd % headsd == 0)
    assert(embedg % headsg == 0)
    dataset_train = IntoDataset(df_train, time_window)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    data_ex = dataset_train[0]
    # print(data_ex)
    data_shape = data_ex.shape
    # print('data_shape', data_shape)
    
    # Initialize generator and discriminator
    generator = Generator(data_shape, latent_dim, headsg, embedd, dropout=dropout)
    discriminator = Discriminator(data_shape, time_window, headsd, embedg, dropout=dropout)

    if cuda:
        print("INFO: Using cuda to train")
        generator.cuda()
        discriminator.cuda()
        
    # Optimizers
    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wdd)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wdg)
    # ----------
    #  Training
    # ----------

    batches_done = 0
    i = int(-print_each_n/2)
    start_all = time.time()
    for epoch in range(epochs):
        start = time.time()
        for batch in dataloader_train:
            # Configure input
            real_data = batch.to(device)
            if do_print:
                print("real data shape", real_data.shape)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.tensor(np.random.normal(0, 1, (batch_size, time_window, latent_dim)), device=device, dtype=torch.float32)
            # print("latent shape", z.shape)

            # Generate a batch of images
            fake_data = generator(z).detach()
            if do_print:
                print("fake data shape", fake_data.shape)
            # Adversarial loss
            # print('real_data shape', real_data.shape)
            # print("fake_data shape", fake_data.shape)
            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data)
            if do_print:
                print("disc real", disc_real)
                print("disc fake", disc_fake)
            loss_D_true = torch.mean(disc_real)
            loss_D_fake = torch.mean(disc_fake)
            loss_D = -loss_D_true + loss_D_fake

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            i += 1
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
                        % (epoch+1, epochs, batches_done % len(dataloader_train), len(dataloader_train), loss_D_true.item(), loss_D_fake.item(), loss_G.item())
                    )
                    if do_print and False:
                        print("fake: ", fake_data)
                        print("real: ", real_data)
                        print()
                    if step_by_step:
                        input()
            batches_done += 1
        end = time.time()
        print(f"Epoch training time: {end-start:.3f} seconds")
        if df_val is not None and y_val is not None:
            discriminator.eval()
            preds = discriminate(discriminator, df_val, time_window)
            preds = np.mean(preds, axis=1)
            preds = np.squeeze(preds)
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
                    end_all = time.time()
                    print(f"Total Training time: {end_all-start_all:.3f} seconds")
                    discriminator = torch.load('checkpoint.pt', weights_only = False)
                    generator = torch.load('checkpoint2.pt', weights_only = False)
                    return generator, discriminator
            discriminator.train()
        
    end_all = time.time()
    print(f"Total Training time: {end_all-start_all:.3f} seconds")
    discriminator = torch.load('checkpoint.pt', weights_only = False)
    generator = torch.load('checkpoint2.pt', weights_only = False)
    return generator, discriminator

@torch.no_grad
def discriminate(discriminator: Discriminator, df: pd.DataFrame, time_window = 40, batch_size=200) -> list: 
    dataset_val = IntoDataset(df, time_window)
    dataloader_val = DataLoader(dataset_val, batch_size, shuffle=False)
    
    scores = []
    i = 0
    start = time.time()
    for batch in dataloader_val:
        data = batch.to(device)
        # print(data.shape)
        score = discriminator(data, do_print=False).cpu().detach().numpy()
        if i%100 == 0:
            print(f"\r[Validating] [Sample {i} / {len(dataloader_val)}] [Score {score[0]}]", end="")
        i+=1
        for s in score:
            scores.append(-s)
    end = time.time()
    print()
    print(f"Validation time: {end-start}")
    if batch_size == 1:
        print(f"Average detection time: {(end-start)/len(df)} seconds")
    return scores