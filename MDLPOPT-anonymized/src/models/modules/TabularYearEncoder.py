import torch
import torch.nn as nn
class TabularYearEncoder(nn.Module):
    def __init__(self, input_dim=67, model_dim=256, dropout=0.3, nominal_idx=[50, 51, 52]):
        super().__init__()
        self.input_dim = input_dim
        
        # 1. Setup for Full Feature Mode (F=67)
        if input_dim == 67:
            self.cont_binary_idx = [i for i in range(input_dim) if i not in nominal_idx]
            self.race_emb = nn.Embedding(10, 16, padding_idx=0)
            self.sex_emb = nn.Embedding(5, 8, padding_idx=0)
            self.cohort_emb = nn.Embedding(10, 8, padding_idx=0)
            
            # 64 (others) + 16 + 8 + 8 = 96
            self.embedding_concat_dim = len(self.cont_binary_idx) + 16 + 8 + 8
        else:
            self.embedding_concat_dim = input_dim

        # 2. DOUBLE THE INPUT DIMENSION
        # We take the features (embedding_concat_dim) AND the original mask (input_dim)
        layer1_input = self.embedding_concat_dim + input_dim
        
        self.net = nn.Sequential(
            nn.Linear(layer1_input, 512), # Now 96 + 67 = 163
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, T, F = x.shape
        x_flat = x.reshape(B * T, F)
        mask_flat = mask.reshape(B * T, F).float() # [B*T, 67]
        
        # Process Embeddings
        if F == 67:
            race = self.race_emb(x_flat[:, 50].long())
            sex = self.sex_emb(x_flat[:, 51].long())
            cohort = self.cohort_emb(x_flat[:, 52].long())
            others = x_flat[:, self.cont_binary_idx]
            features_processed = torch.cat([others, race, sex, cohort], dim=-1) # [B*T, 96]
        else:
            features_processed = x_flat

        # 3. CONCATENATE FEATURES WITH THE MASK
        # This tells the model explicitly which features are real and which are noise
        combined = torch.cat([features_processed, mask_flat], dim=-1) # [B*T, 163]
        
        out = self.net(combined)
        
        # Final visit-level gate (optional but keeps sequence clean)
        year_mask = (mask_flat.any(dim=-1, keepdim=True))
        out = out * year_mask

        return out.view(B, T, -1)