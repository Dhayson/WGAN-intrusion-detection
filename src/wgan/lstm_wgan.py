import pandas as pd
import time
import numpy as np

import torch.nn as nn
import torch
import optuna

from src.early_stop import EarlyStopping

from src.wgan.wgan import Generator, Discriminator
from src.into_dataloader import IntoDataset
from src.lstm import LSTM
from torch.utils.data import DataLoader
from src.wgan.wgan import discriminate
from src.metrics import roc_auc_score
import src.metrics as metrics

cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda else "cpu"

def block_mlp(in_feat, out_feat, leak = 0.0):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.LeakyReLU(negative_slope=leak, inplace=True))
    return layers

class GeneratorLSTM(nn.Module):
    def __init__(self, data_shape, latent_dim, internal_dim, dropout=0.2):
        super(GeneratorLSTM, self).__init__()
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

class DiscriminatorLSTM(nn.Module):
    def __init__(self, data_shape, internal_dim, dropout=0.2):
        super(DiscriminatorLSTM, self).__init__()
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
    
def TrainLSTM(df_train: pd.DataFrame, lrd, lrg, epochs, df_val: pd.DataFrame = None, y_val: pd.Series = None, n_critic = 5, 
    clip_value = 1, latent_dim = 30, optim = torch.optim.RMSprop, wdd = 1e-2, wdg = 1e-2, early_stopping: EarlyStopping = None, dropout=0.2,
    print_each_n = 100, time_window = 40, batch_size=5, do_print = False, step_by_step = False, internal_d=400, internal_g=400, trial = None
    ) -> tuple[GeneratorLSTM, DiscriminatorLSTM]:
    dataset_train = IntoDataset(df_train, time_window)
    dataset_val = IntoDataset(df_val, time_window)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    data_ex = dataset_train[0]
    # print(data_ex)
    data_shape = data_ex.shape
    # print('data_shape', data_shape)
    
    # Initialize generator and discriminator
    generator = GeneratorLSTM(data_shape, latent_dim, internal_d, dropout=dropout)
    discriminator = DiscriminatorLSTM(data_shape, internal_d, dropout=dropout)

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
            preds = discriminate(discriminator, dataset_val, time_window)
            preds = np.mean(preds, axis=1)
            preds = np.squeeze(preds)
            best_thresh = metrics.best_validation_threshold(y_val, preds)
            thresh = best_thresh["thresholds"]
            auc_score = roc_auc_score(y_val, preds)
            print("\nValidation accuracy: ", metrics.accuracy(y_val, preds > thresh))
            print("AUC score: ", auc_score, "\n")
            # Mecanismo de early stopping
            if trial is not None:
                trial.report(auc_score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
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
