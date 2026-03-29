import torch.nn as nn
from src.mamba import Mamba, MambaConfig


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers, seq_len):
        super().__init__()
        self.config = MambaConfig(d_model=hidden, n_layers=layers)
        self.embedding = nn.Linear(in_dim, hidden)
        self.mamba = Mamba(self.config)
        self.head = nn.Sequential(
            nn.Linear(hidden * seq_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba(x)
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        return x