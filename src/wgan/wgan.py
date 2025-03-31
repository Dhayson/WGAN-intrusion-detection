import time
import numpy as np
import pandas as pd
import sys

import torch.nn as nn
import torch
import src.metrics as metrics
from sklearn.metrics import roc_auc_score

from src.early_stop import EarlyStopping
from src.into_dataloader import IntoDataset

from torch.utils.data import DataLoader

from src.lstm import LSTM

cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda else "cpu"

def gradient_penalty(real, fake, discriminator):
    # epsilon serve para calcular a interpolação entre os dados reais e falsos
    epsilon = torch.rand(real.shape).to(device)

    # interpolação por meio do discriminador
    interpolated_data = real*epsilon + (1 - epsilon)*fake
    interpolated_data.requires_grad_(True)
    
    interpolated_scores = discriminator(interpolated_data)

    # cálculo do gradiente
    gradient = torch.autograd.grad(
        inputs = interpolated_data,
        outputs = interpolated_scores,
        grad_outputs = torch.ones_like(interpolated_scores),
        create_graph = True,
        retain_graph = True
    )[0]

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    
    return gradient_penalty

class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__()

def WganTrain(dataset_train: IntoDataset, generator: Generator, discriminator: Discriminator, lrd, lrg, epochs, dataset_val: IntoDataset = None, y_val: pd.Series = None, n_critic = 5, 
    clip_value = 1, latent_dim = 30, optim = torch.optim.RMSprop, wdd = 1e-2, wdg = 1e-2, early_stopping: EarlyStopping = None, dropout=0.2,
    print_each_n = 20, time_window = 40, batch_size=5, do_print = False, step_by_step = False, return_auc = False, lambda_penalty = 0.05) -> tuple[Generator, Discriminator]:
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    best_auc_score = 0;
    data_ex = dataset_train[0]
    # print(data_ex)
    data_shape = data_ex.shape
    # print(data_shape)

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
            if (real_data.shape[0] < batch_size):
                z = torch.tensor(np.random.normal(0, 1, (real_data.shape[0], time_window, latent_dim)), device=device, dtype=torch.float32)
            else:
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
            
            # Gradient penalty
            penalty = gradient_penalty(real_data, fake_data, discriminator)
            loss_D += penalty * lambda_penalty

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
                        "\r[Epoch %d/%d] [Batch %d/%d] [D true loss: %f] [D fake loss: %f] [G loss: %f]"
                        % (epoch+1, epochs, batches_done % len(dataloader_train), len(dataloader_train), loss_D_true.item(), loss_D_fake.item(), loss_G.item()),
                        end=""
                    )
                    sys.stdout.flush()
                    if do_print and False:
                        print("fake: ", fake_data)
                        print("real: ", real_data)
                        print()
                    if step_by_step:
                        input()
            batches_done += 1
        end = time.time()
        print()
        print(f"Epoch training time: {end-start:.3f} seconds")
        if dataset_val is not None and y_val is not None:
            discriminator.eval()
            preds = discriminate(discriminator, dataset_val, time_window)
            best_thresh = metrics.best_validation_threshold(y_val, preds)
            thresh = best_thresh["thresholds"]
            auc_score = roc_auc_score(y_val, preds)
            if auc_score > best_auc_score:
                best_auc_score = auc_score
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
                    if return_auc:
                        return generator, discriminator, best_auc_score
                    else:
                        return generator, discriminator
            discriminator.train()
        
    end_all = time.time()
    print(f"Total Training time: {end_all-start_all:.3f} seconds")
    discriminator = torch.load('checkpoint.pt', weights_only = False)
    generator = torch.load('checkpoint2.pt', weights_only = False)
    if return_auc:
        return generator, discriminator, best_auc_score
    else:
        return generator, discriminator

@torch.no_grad
def discriminate(discriminator: Discriminator, dataset_val: IntoDataset, time_window = 40, batch_size=400, lim=None) -> list: 
    dataloader_val = DataLoader(dataset_val, batch_size, shuffle=False)
    
    scores = []
    i = 0
    start = time.time()
    for batch in dataloader_val:
        data = batch.to(device)
        # print(data.shape)
        score = discriminator(data, do_print=False).cpu().detach().numpy()
        if i%50 == 0:
            print(f"\r[Validating] [Sample {i} / {len(dataloader_val)}] [Score {np.squeeze(np.mean(score[0]))}]", end="")
            sys.stdout.flush()
        i+=1
        for s in score:
            scores.append(-s)
        if i == lim:
            break
    end = time.time()
    print()
    print(f"Validation time: {end-start}")
    if batch_size == 1:
        if lim is not None:
            print(f"Average detection time: {(end-start)/lim} seconds")
        else:
            print(f"Average detection time: {(end-start)/len(dataset_val)} seconds")
    return scores