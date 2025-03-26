import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, dropout=0.25):
        super(TCNBlock, self).__init__()
        
        # Ensure causal padding
        padding = (kernel_size - 1) * dilation  # Padding to maintain causality
        
        # Causal Dilated Convolution (WeightNorm applied)
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        
        # Normalization (BatchNorm)
        self.norm = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 1x1 Convolution for skip connection if dimensions don't match
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        res = self.residual(x)  # Skip connection

        # Convolution, then cropping to maintain causality
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # Crop to keep causality
        
        out = self.norm(out)  # Normalization
        out = F.relu(out)     # Activation
        out = self.dropout(out)  # Dropout
        
        return out + res  # Add residual connection