import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_diabetes

class DiabetesDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X, self.y = load_diabetes(retutn_X_y=True)
        
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx]), torch.tensor(self.y[idx]))
    
    def __len__(self):
        return (len(self.X))
    
dataset = DiabetesDataset()
print(dataset[10])
print(len(dataset))