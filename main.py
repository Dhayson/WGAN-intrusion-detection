import torch
import torch.optim as optim
import torch.nn as nn
from csv-to-dataset import CSVDataset

# Example: Load CSV file
csv_file = "data.csv"
dataset = CSVDataset(csv_file)

batch_size = 64  # Adjust as needed

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example: Iterate over the data
for batch in dataloader:
    print(batch.shape)  # Should be (batch_size, num_features)
    break

# TRAIN WGAN