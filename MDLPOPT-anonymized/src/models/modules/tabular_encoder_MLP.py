import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, X, X_mask=None):
        # X: (B, T, F)
        B, T, F = X.shape
        X_flat = X.reshape(B * T, F)  # treat sequence steps independently
        encoded_flat = self.mlp(X_flat)
        encoded = encoded_flat.reshape(B, T, -1)
        # average over time dimension
        return encoded.mean(dim=1)  # (B, hidden_dim)