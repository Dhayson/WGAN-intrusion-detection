import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            num_layers=3,
            batch_first=False
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out