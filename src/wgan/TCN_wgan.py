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

class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim, tcn_channels=32, kernel_size=3, num_tcn_layers=1):
        """
        Transforma um vetor latente em uma amostra com a mesma dimensão dos dados reais.
        """
        super(Generator, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        self.num_features = int(np.prod(data_shape))
        self.tcn_channels = tcn_channels

        # Expande o vetor latente e reinterpreta como sequência
        self.fc = nn.Linear(latent_dim, tcn_channels * self.num_features)
        tcn_blocks = []
        for i in range(num_tcn_layers):
            tcn_blocks.append(
                TCNBlock(tcn_channels, tcn_channels,
                         kernel_size=kernel_size,
                         dilation=2**i,
                         dropout=0.2)
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        self.out_conv = nn.Conv1d(tcn_channels, 1, kernel_size=1)
    
    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.fc(z)  # (batch, tcn_channels * num_features)
        x = x.view(-1, self.tcn_channels, self.num_features)  # Reinterpreta para (batch, tcn_channels, num_features)
        x = self.tcn(x)
        x = self.out_conv(x)  # Reduz os canais para 1
        x = x.view(-1, self.num_features)  # Achata a saída para (batch, num_features)
        return x

class Discriminator(nn.Module):
    def __init__(self, data_shape, tcn_channels=32, kernel_size=3, num_tcn_layers=1):
        """
        Recebe uma amostra (vetor) e a interpreta como uma sequência, retornando um escalar que representa a validade.
        """
        super(Discriminator, self).__init__()
        self.num_features = int(np.prod(data_shape))
        self.input_conv = nn.Conv1d(1, tcn_channels, kernel_size=1)
        tcn_blocks = []
        for i in range(num_tcn_layers):
            tcn_blocks.append(
                TCNBlock(tcn_channels, tcn_channels,
                         kernel_size=kernel_size,
                         dilation=2**i,
                         dropout=0.2)
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        self.out_fc = nn.Linear(tcn_channels * self.num_features, 1)
    
    def forward(self, x):
        # x: (batch, num_features)
        x = x.view(-1, 1, self.num_features)
        x = self.input_conv(x)
        x = self.tcn(x)
        x = x.view(-1, x.size(1) * x.size(2))
        validity = self.out_fc(x)
        return validity

def Train(df_train, lrd, lrg, epochs, df_val=None, y_val=None, n_critic=5, 
          clip_value=1, latent_dim=30, optim=torch.optim.RMSprop, wd=1e-2):
    """
    Função de treinamento da WGAN com TCN.
    
    Parâmetros:
      - df_train: DataFrame com os dados de treinamento.
      - lrd, lrg: taxas de aprendizado para o discriminador e gerador.
      - epochs: número de épocas de treinamento.
      - df_val, y_val: (opcionais) dados e rótulos de validação.
      - n_critic: número de atualizações do discriminador para cada atualização do gerador.
      - clip_value: valor de clipagem para os pesos do discriminador.
      - latent_dim: dimensão do vetor latente.
      - optim: otimizador utilizado.
      - wd: weight decay.
    """
    data_shape = df_train.iloc[0].shape
    print_each_n = 3000

    # Inicializa o gerador e o discriminador com parâmetros padrão (você pode ajustar conforme necessário)
    generator = Generator(data_shape, latent_dim, tcn_channels=32, kernel_size=3, num_tcn_layers=1)
    discriminator = Discriminator(data_shape, tcn_channels=32, kernel_size=3, num_tcn_layers=1)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = optim(generator.parameters(), lr=lrg, weight_decay=wd)
    optimizer_D = optim(discriminator.parameters(), lr=lrd, weight_decay=wd)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    batches_done = 0
    for epoch in range(epochs):
        for i, datum in df_train.iterrows():
            real_data = Variable(torch.from_numpy(datum.to_numpy()).type(Tensor))
            optimizer_D.zero_grad()

            # Amostra ruído para gerar dados falsos
            z = Variable(Tensor(np.random.normal(0, 1, (latent_dim,))))
            fake_data = generator(z).detach()
            loss_D_true = torch.mean(discriminator(real_data))
            loss_D_fake = torch.mean(discriminator(fake_data))
            loss_D = -loss_D_true + loss_D_fake

            loss_D.backward()
            optimizer_D.step()

            # Clipagem dos pesos do discriminador
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if i % n_critic == 0:
                optimizer_G.zero_grad()
                gen_data = generator(z)
                loss_G = -torch.mean(discriminator(gen_data))
                loss_G.backward()
                optimizer_G.step()

                if i % print_each_n == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D true loss: %f] [D fake loss: %f] [G loss: %f]" %
                          (epoch+1, epochs, batches_done % len(df_train), len(df_train),
                           loss_D_true.item(), loss_D_fake.item(), loss_G.item()))
            batches_done += 1
    return generator, discriminator

def discriminate(discriminator, df):
    """
    Calcula os escores do discriminador para cada amostra de um DataFrame.
    """
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    results = []
    for i, row in df.iterrows():
        row_np = row.to_numpy()
        row_tensor = torch.from_numpy(row_np).type(Tensor).unsqueeze(0)
        output = discriminator(row_tensor)
        results.append(output.item())
    return results
