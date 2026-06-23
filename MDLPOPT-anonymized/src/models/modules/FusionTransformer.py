import torch
import torch.nn as nn
class TrajectoryTransformerHub(nn.Module):
    """The 'Brain' that accepts any number of year-tokens from any modality."""
    def __init__(self, model_dim=256, n_heads=8, n_layers=4, n_classes=3):
        super().__init__()
        self.model_dim = model_dim
        
        # Positional Encoding for Years (Year 0, 1, 2...)
        self.year_embedding = nn.Embedding(10, model_dim) 
        
        # Modality Encoding (Tabular vs Image) - helps transformer distinguish source
        self.mod_embedding = nn.Embedding(2, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Linear(model_dim, n_classes)

    def forward(self, token_list, year_indices, mod_indices):
        """
        token_list: List of tensors [(B, D), (B, D), ...]
        year_indices: List of ints [0, 1, 2...] corresponding to the tokens
        mod_indices: List of ints [0=Tab, 1=Img]
        """
        # 1. Stack tokens: (B, num_tokens, D)
        x = torch.stack(token_list, dim=1)
        B, T, D = x.shape
        
        # 2. Add Year and Modality context
        for i in range(T):
            year_idx = torch.full((B,), year_indices[i], device=x.device)
            mod_idx = torch.full((B,), mod_indices[i], device=x.device)
            x[:, i, :] += self.year_embedding(year_idx) + self.mod_embedding(mod_idx)

        # 3. Self-Attention (Every year looks at every other year/modality)
        fused = self.transformer(x)
        
        # 4. Global Average Pool across tokens or take the last year
        summary = fused.mean(dim=1) 
        return self.classifier(summary)