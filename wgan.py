import torch
import torch.nn as nn
from tcn import TCNBlock

class Critic(nn.Module):
    def __init__(self, seq_len, channels, hidden_dim, n_tcn_blocks):
        super(Critic, self).__init__()

        # Fully Connected Input Layer
        self.fc_in = nn.Linear(seq_len * channels, seq_len * channels)
        
        # TCN Blocks
        tcn_layers = []
        for i in range(n_tcn_blocks):
            tcn_layers.append(TCNBlock(channels, channels, dilation=2**i))
        self.tcn = nn.Sequential(*tcn_layers)
        
        # Fully Connected Output Layer
        self.fc_out = nn.Linear(seq_len * channels, 1)  # Output single scalar value
    
    def forward(self, x):
        x = self.fc_in(x).view(x.size(0), -1, 1)  # Reshape to (batch, channels, seq_len)
        x = self.tcn(x)  # Process with TCN blocks
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_out(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, channels, hidden_dim, n_tcn_blocks):
        super(Generator, self).__init__()

        # Fully Connected Input Layer
        self.fc_in = nn.Linear(latent_dim, seq_len * channels)
        
        # TCN Blocks
        tcn_layers = []
        for i in range(n_tcn_blocks):
            tcn_layers.append(TCNBlock(channels, channels, dilation=2**i))
        self.tcn = nn.Sequential(*tcn_layers)
        
        # Fully Connected Output Layer
        self.fc_out = nn.Linear(seq_len * channels, seq_len * channels)
    
    def forward(self, z):
        x = self.fc_in(z).view(z.size(0), -1, 1)  # Reshape to (batch, channels, seq_len)
        x = self.tcn(x)  # Pass through TCN blocks
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_out(x)