# Implementação do Early Stopping
import torch
import numpy as np

class EarlyStopping:
  def __init__(self, patience=7, delta=0, verbose=True, path='checkpoint.pt', path2='checkpoint2.pt'):
      self.patience = patience
      self.delta = delta
      self.verbose = verbose
      self.counter = 0
      self.early_stop = False
      self.val_max_acc = -0
      self.path = path
      self.path2 = path2

  def __call__(self, val_acc, model, model2 = None):
    if val_acc > self.val_max_acc + self.delta:   # Caso a acc da validação reduza, vamos salvar o modelo e nova acc mínima
      self.save_checkpoint(val_acc, model, model2)
      self.counter = 0
    else:                                           # Caso a acc da validação NÃO reduza, vamos incrementar o contador da paciencia
      self.counter += 1
      print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Current validation acc: {val_acc:.5f}')
      if self.counter >= self.patience:
          self.early_stop = True

  def save_checkpoint(self, val_acc, model, model2 = None):
    if self.verbose:
        print(f'Validation accuracy increased ({self.val_max_acc:.5f} --> {val_acc:.5f}).  Saving model ...')
    torch.save(model, self.path)
    if model2 is not None:
        torch.save(model2, self.path2)
    self.val_max_acc = val_acc