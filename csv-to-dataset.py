import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)  # Load CSV file
        self.data = torch.tensor(self.data.values, dtype=torch.float32)  # Convert to Tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Return one row as input
