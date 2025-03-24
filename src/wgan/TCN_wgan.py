import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# Flag global para utilização de CUDA
cuda = torch.cuda.is_available()

class CausalConv1d(nn.Module):
    """
    Convolução causal 1D: garante que cada saída no tempo t utilize apenas dados do passado.
    Aplica padding manualmente apenas à esquerda.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        pad = (self.dilation * (self.kernel_size - 1), 0)  # Padding somente à esquerda
        x = F.pad(x, pad)
        return self.conv(x)

class TCNBlock(nn.Module):
    """
    Bloco TCN: inclui convolução causal dilatada, ReLU, BatchNorm, Dropout e conexão residual.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# --- Novas definições considerando janelas temporais ---
# Suponha que cada janela tenha tamanho T (número de instantes consecutivos) e cada instante tenha F features.
# Assim, cada amostra terá shape (F, T).

class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim, tcn_channels=32, kernel_size=3, num_tcn_layers=1):
        """
        data_shape: (T, F) onde T é o tamanho da janela (passos de tempo) e F é o número de features.
        """
        super(Generator, self).__init__()
        self.T = data_shape[0]  # janela temporal
        self.F = data_shape[1]  # número de features (canais)
        self.tcn_channels = tcn_channels

        # Expande o vetor latente para formar uma sequência de tamanho T (com tcn_channels canais)
        self.fc = nn.Linear(latent_dim, tcn_channels * self.T)
        tcn_blocks = []
        for i in range(num_tcn_layers):
            tcn_blocks.append(
                TCNBlock(tcn_channels, tcn_channels,
                         kernel_size=kernel_size,
                         dilation=2**i,
                         dropout=0.2)
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        # Altera o número de canais para F (número de features)
        self.out_conv = nn.Conv1d(tcn_channels, self.F, kernel_size=1)
    
    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.fc(z)  # (batch, tcn_channels * T)
        x = x.view(-1, self.tcn_channels, self.T)  # (batch, tcn_channels, T)
        x = self.tcn(x)  # (batch, tcn_channels, T)
        x = self.out_conv(x)  # (batch, F, T)
        return x

class Discriminator(nn.Module):
    def __init__(self, data_shape, tcn_channels=32, kernel_size=3, num_tcn_layers=1):
        """
        data_shape: (T, F)
        """
        super(Discriminator, self).__init__()
        self.T = data_shape[0]
        self.F = data_shape[1]
        # Projeta de F canais para tcn_channels
        self.input_conv = nn.Conv1d(self.F, tcn_channels, kernel_size=1)
        tcn_blocks = []
        for i in range(num_tcn_layers):
            tcn_blocks.append(
                TCNBlock(tcn_channels, tcn_channels,
                         kernel_size=kernel_size,
                         dilation=2**i,
                         dropout=0.2)
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        # Achata para um escalar: tcn_channels * T
        self.out_fc = nn.Linear(tcn_channels * self.T, 1)
    
    def forward(self, x):
        # x: (batch, F, T)
        x = self.input_conv(x)  # (batch, tcn_channels, T)
        x = self.tcn(x)         # (batch, tcn_channels, T)
        x = x.view(x.size(0), -1)  # Flatten para (batch, tcn_channels * T)
        validity = self.out_fc(x)
        return validity

# Função auxiliar para criar janelas temporais a partir do DataFrame.
def create_windows(df, window_size):
    data = df.to_numpy()  # shape: (N, F)
    windows = []
    for i in range(len(data) - window_size + 1):
        # Cada janela é composta por 'window_size' instantes consecutivos.
        window = data[i : i + window_size]  # shape: (window_size, F)
        windows.append(window)
    return np.array(windows)  # shape: (num_windows, window_size, F)

# --- Função de treinamento modificada ---
def Train(df_train, lrd, lrg, epochs, window_size=50, batch_size=32,
          df_val=None, y_val=None, n_critic=5, clip_value=1, latent_dim=30,
          optim=torch.optim.RMSprop, wd=1e-2):
    """
    Treinamento da WGAN com TCN utilizando janelas temporais.
    df_train: DataFrame com os dados de treinamento (linhas ordenadas cronologicamente).
    window_size: número de instantes em cada janela.
    """
    # Supondo que as colunas do df_train são as features (excluindo timestamp etc.)
    num_features = df_train.shape[1]
    # data_shape agora é (T, F)
    data_shape = (window_size, num_features)
    print_each_n = 3000

    # Inicializa gerador e discriminador com a nova estrutura
    generator = Generator(data_shape, latent_dim, tcn_channels=32, kernel_size=3, num_tcn_layers=1)
    discriminator = Discriminator(data_shape, tcn_channels=32, kernel_size=3, num_tcn_layers=1)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wd)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wd)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Cria as janelas temporais a partir do df_train
    windows = create_windows(df_train, window_size)  # shape: (num_windows, window_size, num_features)
    num_windows = windows.shape[0]

    batches_done = 0
    for epoch in range(epochs):
        # Embaralha as janelas a cada época
        indices = np.random.permutation(num_windows)
        for i in range(0, num_windows, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_windows = windows[batch_indices]  # shape: (batch, window_size, num_features)
            # Converte para tensor e reordena para (batch, F, T)
            real_data = torch.from_numpy(batch_windows).float().permute(0, 2, 1)
            real_data = Variable(real_data).type(Tensor)
            
            optimizer_D.zero_grad()

            # Amostra ruído para gerar dados falsos (batch_size x latent_dim)
            z = Variable(Tensor(np.random.normal(0, 1, (real_data.size(0), latent_dim))))
            fake_data = generator(z).detach()  # fake_data shape: (batch, F, T)

            loss_D_true = torch.mean(discriminator(real_data))
            loss_D_fake = torch.mean(discriminator(fake_data))
            loss_D = -loss_D_true + loss_D_fake

            loss_D.backward()
            optimizer_D.step()

            # Clipagem dos pesos do discriminador
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if batches_done % n_critic == 0:
                optimizer_G.zero_grad()
                gen_data = generator(z)
                loss_G = -torch.mean(discriminator(gen_data))
                loss_G.backward()
                optimizer_G.step()

                if batches_done % print_each_n == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D true loss: %f] [D fake loss: %f] [G loss: %f]" %
                          (epoch+1, epochs, batches_done % num_windows, num_windows,
                           loss_D_true.item(), loss_D_fake.item(), loss_G.item()))
            batches_done += 1
    return generator, discriminator

def discriminate(discriminator, df, window_size=50):
    """
    Calcula os escores do discriminador para cada janela temporal gerada a partir do DataFrame.
    Assumindo que df é um DataFrame ordenado cronologicamente e que você já pré-processou os dados.
    """
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    windows = create_windows(df, window_size)  # Função para criar janelas
    results = []
    for i in range(len(windows)):
        # Cada janela: (window_size, num_features)
        # Reordena para (1, num_features, window_size) para o discriminador
        window = torch.from_numpy(windows[i]).float().permute(1, 0).unsqueeze(0)
        window = window.type(Tensor)
        output = discriminator(window)
        results.append(output.item())
    return results
