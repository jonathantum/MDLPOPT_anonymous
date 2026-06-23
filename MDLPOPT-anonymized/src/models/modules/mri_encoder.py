import torch
import torch.nn as nn
from torchvision import models
from torch.utils.checkpoint import checkpoint  

class MRIEncoder(nn.Module):
    def __init__(self, model_dim=256, dropout=0.7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveMaxPool2d((1, 1))        
        self.projection = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout) 
        )

    def forward(self, x):
        if self.training and x.requires_grad:
            x = checkpoint(self.features, x, use_reentrant=False)
        else:
            x = self.features(x)        
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.projection(x)
    
class SliceAttention(nn.Module):
    def __init__(self, model_dim, dropout=0.2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.Tanh(),
            # Dropout here prevents the attention mechanism from 
            # over-focusing on specific noisy slice features.
            nn.Dropout(dropout), 
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Supports:
        - Single: [Slices, model_dim] -> [model_dim]
        - Batched: [B, Slices, model_dim] -> [B, model_dim]
        """
        is_batched = x.dim() == 3
        slice_dim = 1 if is_batched else 0
        
        # Calculate attention scores
        weights = self.attention(x)
        
        # Softmax across the Slices dimension
        weights = torch.softmax(weights, dim=slice_dim)
        
        # Weighted sum: (B, Slices, model_dim) * (B, Slices, 1)
        return torch.sum(x * weights, dim=slice_dim)