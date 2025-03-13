import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, num_layers=1, batch_first=True)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out