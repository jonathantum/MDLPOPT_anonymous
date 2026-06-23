import torch
import torch.nn as nn
from timm.layers import DropPath
class CustomTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        # DropPath drops entire residual blocks during training
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        # 1. Attention Block (Pre-LN)
        x_norm = self.norm1(x)
        # attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask)
        # x = x + self.drop_path(attn_out)
        attn_out, weights = self.attn(x_norm, x_norm, x_norm, 
                                       key_padding_mask=src_key_padding_mask,
                                       average_attn_weights=True)
        x = x + self.drop_path(attn_out)
        # 2. MLP Block (Pre-LN)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights

class MidFusionTransformer(nn.Module):
    def __init__(self, input_dim=256, num_heads=8, num_layers=2, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 25, input_dim)) 
        
        # Using our custom layers instead of standard nn.TransformerEncoder
        self.layers = nn.ModuleList([
            CustomTransformerLayer(input_dim, num_heads, dropout, drop_path)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, tokens, masks=None):
        B, N, D = tokens.shape
        last_weights = None

        x = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1)
        x = x + self.pos_embed[:, :N+1, :]
        
        if masks is not None:
            cls_mask = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
            src_key_padding_mask = torch.cat([cls_mask, masks], dim=1)
        else:
            src_key_padding_mask = None
        for layer in self.layers:
            x,last_weights = layer(x, src_key_padding_mask=src_key_padding_mask)
            
        x = self.norm(x) # Final norm for Pre-LN architecture
        return x[:, 0, :], last_weights # Joint Latent Representation