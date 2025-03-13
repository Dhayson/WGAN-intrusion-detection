import numpy as np
import pandas as pd

from torch.autograd import Variable

import torch.nn as nn
import torch
import src.metrics as metrics

from src.lstm import LSTM

cuda = True if torch.cuda.is_available() else False

def block_mlp(in_feat, out_feat):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.ReLU(inplace=True))
    return layers

class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim):
        super(Generator, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        # TODO: alterar arquitetura para usar TCN e self attention
        self.fc1 = nn.Sequential(
            *block_mlp(latent_dim, 40, leak=0.1),
        )
        self.lstm = LSTM(40, 40, 40, 40)
        self.fc2 = nn.Sequential(
            nn.Linear(40, int(np.prod(data_shape))),
            nn.Sigmoid(),
        )
        self.flat = nn.Flatten(0)

    def forward(self, z):
        data = self.fc1(z)
        data = self.lstm(data)
        data = self.fc2(data)
        data = self.flat(data)
        return data

class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(
            *block_mlp(int(np.prod(data_shape)), 40, leak=0.1),
        )
        self.lstm = LSTM(40, 40, 40, 40)
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
    clip_value = 1, latent_dim = 30, optim = torch.optim.RMSprop, wd = 1e-2) -> tuple[Generator, Discriminator]:
    
    data_shape = df_train.loc[0].shape
    print_each_n = 3000
    
    # Initialize generator and discriminator
    generator = Generator(data_shape, latent_dim)
    discriminator = Discriminator(data_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        
    # Optimizers
    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wd)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wd)
    
    Tensor: type[torch.FloatTensor] = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(epochs):
        for i, datum in df_train.iterrows():
            # Configure input
            real_data = Variable(torch.from_numpy(datum.to_numpy()).type(Tensor))
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (latent_dim,))))

            # Generate a batch of images
            fake_data = generator(z).detach()
            # Adversarial loss
            loss_D_true = torch.mean(discriminator(real_data))
            loss_D_fake = torch.mean(discriminator(fake_data))
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
            print("\nValidation accuracy: ", metrics.accuracy(y_val, preds > thresh), "\n")
            discriminator.train()
    
    return generator, discriminator

def discriminate(discriminator: Discriminator, df: pd.DataFrame) -> list:
    Tensor: type[torch.FloatTensor] = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    results = []
    for i, val in df.iterrows():
        val0= val.to_numpy()
        val_f = Variable(torch.from_numpy(val0).type(Tensor))
        result = discriminator(val_f)
        results.append(result.detach())
    return results