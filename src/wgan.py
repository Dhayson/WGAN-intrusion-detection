import numpy as np
import pandas as pd

from torch.autograd import Variable

import torch.nn as nn
import torch

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
        self.model = nn.Sequential(
            *block_mlp(latent_dim, 50),
            *block_mlp(50, 70),
            *block_mlp(70, 80),
            nn.Linear(80, int(np.prod(data_shape))),
            nn.Sigmoid()
        )

    def forward(self, z):
        data = self.model(z)
        return data

class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(data_shape)), 40),
            nn.ReLU(inplace=True),
            nn.Linear(40, 15),
            nn.ReLU(inplace=True),
            nn.Linear(15, 1)
        )

    def forward(self, data):
        validity = self.model(data)
        return validity

def Train(df_train: pd.DataFrame, lr, epochs) -> tuple[Generator, Discriminator]:
    data_shape = df_train.loc[0].shape
    latent_dim = 30
    clip_value = 2
    n_critic = 3
    print_each_n = 300
    
    # Initialize generator and discriminator
    generator = Generator(data_shape, latent_dim)
    discriminator = Discriminator(data_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
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