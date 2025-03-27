import torch

cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda else "cpu"

class MeanNormalizeTensor:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)

    @torch.no_grad
    def __call__(self, tensor):
        tensor = tensor.to(device)
        return (tensor - self.mean) / (self.std + 1e-9)

class MinMaxNormalizeTensor:
    def __init__(self, max, min):
        self.max = torch.tensor(max, device=device)
        self.min = torch.tensor(min, device=device)
    
    @torch.no_grad
    def __call__(self, tensor):
        tensor = tensor.to(device)
        return (tensor - self.min) / (self.max - self.min + 1e-9)