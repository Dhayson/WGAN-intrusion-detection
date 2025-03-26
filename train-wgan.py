import torch
import torch.optim as optim
import torch.nn as nn
from wgan import Generator, Critic

# Hyperparameters
latent_dim = 100
seq_len = 50
channels = 64
hidden_dim = 128
n_tcn_blocks = 3
lr = 0.0002
weight_clip = 0.01
n_critic = 5  # W-GAN trains critic multiple times per generator step

# Initialize Models
generator = Generator(latent_dim, seq_len, channels, hidden_dim, n_tcn_blocks)
critic = Critic(seq_len, channels, hidden_dim, n_tcn_blocks)

# Optimizers (Adam as requested)
opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))


def train_wgan(generator, critic, opt_gen, opt_critic, dataloader, epochs=100):
    for epoch in range(epochs):
        for i, real_data in enumerate(dataloader):
            real_data = real_data.view(real_data.size(0), -1)  # Flatten input
            batch_size = real_data.size(0)

            # Train Critic (multiple times per generator step)
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim)
                fake_data = generator(z)

                critic_real = critic(real_data)
                critic_fake = critic(fake_data)

                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                
                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight Clipping for W-GAN
                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

            # Train Generator
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            loss_gen = -torch.mean(critic(fake_data))

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        print(f"Epoch [{epoch+1}/{epochs}], Critic Loss: {loss_critic.item():.4f}, Generator Loss: {loss_gen.item():.4f}")

# Example training call
# train_wgan(generator, critic, opt_gen, opt_critic, dataloader, epochs=100)
