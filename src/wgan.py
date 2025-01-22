import os
import numpy as np
import pandas as pd
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim):
        super(Generator, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        # TODO: alterar arquitetura para usar TCN e self attention
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 80, normalize=False),
            *block(80, 80),
            *block(80, 80),
            *block(80, 80),
            nn.Linear(80, int(np.prod(data_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        data = self.model(z)
        return data

class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(data_shape)), 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80, 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80, 1),
        )

    def forward(self, data):
        validity = self.model(data)
        return validity

def Train(df_train: pd.DataFrame, lr, epochs):
    data_shape = df_train.loc[0].shape
    latent_dim = 80
    clip_value = 0.01
    n_critic = 3
    print_each_n = 300
    
    # Initialize generator and discriminator
    generator = Generator(data_shape, latent_dim)
    discriminator = Discriminator(data_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        
    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    
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
            z = Variable(Tensor(np.random.normal(0, 1, (datum.shape[0], latent_dim))))

            # Generate a batch of images
            fake_data = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_data)) + torch.mean(discriminator(fake_data))

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
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, epochs, batches_done % len(df_train), len(df_train), loss_D.item(), loss_G.item())
                    )
            batches_done += 1
