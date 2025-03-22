import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad

class IntoDataset(Dataset):
    def __init__(self, dataframe, time_window):
        self.dataframe = dataframe
        self.time_window = time_window

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Supondo que a última coluna seja o rótulo
        time_window = self.time_window
        x = self.dataframe.iloc[max(0, idx-self.time_window):idx+1].to_numpy()
        
        x_pad = torch.tensor(x, dtype=torch.float32)
        if x.shape[0] != time_window:
            # Realiza padding nos primeiros pacotes
            target_size = (time_window, time_window)
            pad_rows = target_size[0] - x.shape[0]
            x_pad = pad(x_pad, (0, 0, 0, pad_rows), value=0)
        return x_pad

class IntoDatasetNoTime(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx].to_numpy()
        x = torch.tensor(x, dtype=torch.float32)
        return x