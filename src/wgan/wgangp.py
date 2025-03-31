import sys
import pandas as pd
import numpy as np
import torch
import torch_optimizer
import random
import time

# Importa o módulo de métricas (certifique-se de que o caminho está correto)
import src.metrics as metrics

# ------------------------------------------------------------
# Dataset personalizado: cada amostra é uma janela temporal
class TimeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, time_window: int = 40):
        super().__init__()
        # Converte o DataFrame para array NumPy para indexação posicional
        self.data = df.to_numpy().astype(np.float32)
        self.time_window = time_window
        self.num_samples = len(self.data) - time_window + 1
        if self.num_samples < 1:
            raise ValueError(f"DataFrame com {len(df)} linhas não comporta janela de {time_window} timesteps.")
        print(f"[TimeWindowDataset] data.shape: {self.data.shape}, type: {type(self.data)}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.time_window]  # (time_window, num_features)
        return torch.tensor(seq, dtype=torch.float32)

# ------------------------------------------------------------
# Bloco TCN, Gerador e Discriminador
class TCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.norm(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNGenerator(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, num_channels, num_layers, kernel_size=3, dropout=0.2):
        super(TCNGenerator, self).__init__()
        self.input_linear = torch.nn.Linear(latent_dim, num_channels)
        self.tcn_blocks = torch.nn.ModuleList([
            TCNBlock(num_channels, num_channels, kernel_size=kernel_size,
                     dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_linear = torch.nn.Linear(num_channels, output_dim)
    
    def forward(self, z):
        # z: (batch, time_window, latent_dim)
        x = self.input_linear(z)  # (batch, time_window, num_channels)
        x = x.transpose(1, 2)     # (batch, num_channels, time_window)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)     # (batch, time_window, num_channels)
        out = self.output_linear(x)  # (batch, time_window, output_dim)
        return out

class TCNDiscriminator(torch.nn.Module):
    def __init__(self, input_dim, num_channels, num_layers, kernel_size=3, dropout=0.2):
        super(TCNDiscriminator, self).__init__()
        self.input_linear = torch.nn.Linear(input_dim, num_channels)
        self.tcn_blocks = torch.nn.ModuleList([
            TCNBlock(num_channels, num_channels, kernel_size=kernel_size,
                     dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_linear = torch.nn.Linear(num_channels, 1)
    
    def forward(self, x):
        # x: (batch, time_window, input_dim)
        x = self.input_linear(x)   # (batch, time_window, num_channels)
        x = x.transpose(1, 2)      # (batch, num_channels, time_window)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.mean(dim=2)          # Pooling global
        out = self.output_linear(x)  # (batch, 1)
        return out.squeeze(1)

# ------------------------------------------------------------
# Função de treinamento para TCN-based WGAN-GP
def TrainTCN(df_train: pd.DataFrame,
             latent_dim, output_dim, input_dim,
             lrd, lrg, epochs,
             dataset_val: pd.DataFrame = None,
             y_val=None,
             n_critic=5,
             optim=torch.optim.RMSprop,  # Pode ser Adam, por exemplo
             wdd=1e-2, wdg=1e-2,
             early_stopping=None,
             dropout=0.2,
             print_each_n=20,
             time_window=40,
             batch_size=5,
             num_channels=64,
             num_layers=3,
             do_print=False,
             step_by_step=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cria dataset de treino (janelas temporais)
    dataset_train = TimeWindowDataset(df_train, time_window)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    generator = TCNGenerator(latent_dim, output_dim, num_channels, num_layers, dropout=dropout)
    discriminator = TCNDiscriminator(input_dim, num_channels, num_layers, dropout=dropout)
    generator.to(device)
    discriminator.to(device)
    
    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wdg)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wdd)
    
    batches_done = 0
    i = -print_each_n // 2
    start_all = time.time()
    lambda_gp = 10  # Peso do gradient penalty
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        start = time.time()
        for batch in dataloader_train:
            real_data = batch.to(device)
            current_batch_size = real_data.size(0)  # Usa o tamanho real do batch
            optimizer_D.zero_grad()
            
            # Gera dados falsos usando current_batch_size
            z = torch.randn(current_batch_size, time_window, latent_dim, device=device)
            fake_data = generator(z).detach()
            
            # Saídas do discriminador para dados reais e falsos
            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data)
            
            # Perda base da WGAN
            loss_D = -torch.mean(disc_real) + torch.mean(disc_fake)
            
            # Cálculo do gradient penalty
            alpha = torch.rand(current_batch_size, 1, 1, device=device)
            interpolates = alpha * real_data + (1 - alpha) * fake_data
            interpolates.requires_grad_(True)
            disc_interpolates = discriminator(interpolates)
            grad_outputs = torch.ones_like(disc_interpolates, device=device)
            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradients = gradients.view(current_batch_size, -1)
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            # Incorpora o gradient penalty na perda do discriminador
            loss_D += gradient_penalty
            loss_D.backward()
            optimizer_D.step()
            
            i += 1
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                gen_data = generator(z)
                loss_G = -torch.mean(discriminator(gen_data))
                loss_G.backward()
                optimizer_G.step()
                if i % print_each_n == 0:
                    print(f"\r[Epoch {epoch+1}/{epochs}] [Batch {batches_done % len(dataloader_train)}/{len(dataloader_train)}] "
                          f"[D real: {disc_real.mean().item():.6f}] [D fake: {disc_fake.mean().item():.6f}] "
                          f"[G loss: {loss_G.item():.6f}]", end="")
                    if do_print and step_by_step:
                        input("Pressione Enter para continuar...")
            batches_done += 1
        end = time.time()
        print()
        print(f"Tempo de treinamento da Epoch {epoch+1}: {end - start:.3f} segundos")
        
        # Validação (se dataset_val e y_val forem fornecidos)
        if dataset_val is not None and y_val is not None:
            discriminator.eval()
            val_dataset = TimeWindowDataset(dataset_val, time_window)
            preds = discriminate(discriminator, val_dataset, time_window=time_window, batch_size=400, device=device)
            # Ajusta os rótulos para que correspondam ao rótulo da última linha de cada janela
            adjusted_y_val = y_val[time_window - 1:]
            try:
                auc = metrics.roc_auc_score(adjusted_y_val, preds)
            except Exception as e:
                auc = 0.0
                print("Erro ao calcular AUC:", e)
            print(f"Epoch {epoch+1} - Validation AUC: {auc:.6f}")
            
            if early_stopping is not None:
                early_stopping(auc, discriminator, generator)
                if early_stopping.early_stop:
                    print(f"Treinamento interrompido pelo early stopping na Epoch {epoch+1}")
                    generator = torch.load(early_stopping.path2, map_location=device)
                    discriminator = torch.load(early_stopping.path, map_location=device)
                    return generator, discriminator
            discriminator.train()
    
    end_all = time.time()
    print(f"Tempo total de treinamento: {end_all - start_all:.3f} segundos")
    if early_stopping is not None:
        generator = torch.load(early_stopping.path2, map_location=device)
        discriminator = torch.load(early_stopping.path, map_location=device)
    return generator, discriminator

# ------------------------------------------------------------
# Função de inferência: cria janelas e retorna scores
@torch.no_grad()
def discriminate(discriminator: torch.nn.Module, dataset_val: torch.utils.data.Dataset, time_window=40, batch_size=400, device="cpu"):
    loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    scores = []
    start = time.time()
    i = 0
    for batch in loader:
        batch = batch.to(device)
        out = discriminator(batch).cpu().numpy()
        scores.extend([-s for s in out])
        if i % 50 == 0:
            print(f"\r[Validating] [Sample {i} / {len(loader)}] [Score {np.squeeze(np.mean(out[0]))}]", end="")
        i += 1
    end = time.time()
    print(f"Validation time: {end-start}")
    if batch_size == 1:
        print(f"Average detection time: {(end-start)/len(dataset_val)} seconds")
    return scores 
