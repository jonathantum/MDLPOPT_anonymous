import torch
import torch.nn as nn

class MLPFusion(nn.Module):
    def __init__(self, input_dim=256, num_years=3, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_years = num_years
        
        # Flattened dimension: T * D (e.g., 3 * 256 = 768)
        flattened_dim = input_dim * num_years
        
        self.mlp = nn.Sequential(
            nn.Linear(flattened_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        """
        x: (B, T, D) -> Sequential image features
        mask: (B, T) -> Ignored by standard MLP, but we can zero out masked features
        """
        B, T, D = x.shape
        
        # 1. Zero out masked features so they don't contribute to the flat vector
        if mask is not None:
            # mask is True for padding, so we flip it
            x = x * (~mask.unsqueeze(-1))
            
        # 2. Flatten: (B, T, D) -> (B, T*D)
        # Note: We enforce the sequence length to match num_years
        x_flat = x[:, :self.num_years, :].reshape(B, -1)
        
        # 3. Pass through MLP
        # Output: (B, D)
        return self.mlp(x_flat)