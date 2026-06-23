import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, n_classes=2, dropout=0.1):
        super().__init__()
        # Adding a Dropout layer to prevent memorization
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, encoded):
        # encoded: (B, T, H) or (B, H)
        if encoded.dim() == 3:
            # Using the first token (baseline) as the representation
            encoded = encoded[:, 0]
        
        # Apply dropout before the final linear layer
        encoded = self.dropout(encoded)
        return self.fc(encoded)