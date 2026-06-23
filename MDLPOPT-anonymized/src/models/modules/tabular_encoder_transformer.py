import torch
import torch.nn as nn


class TabularTransformerEncoder(nn.Module):
    """
    Transformer encoder for tabular time-series:
    X: (B, T_in, F)
    -> (B, T_in, D)
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        max_timesteps: int = 20,
        feature_types: list | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim

        # ---------- feature projection ----------
        self.input_proj = nn.Linear(input_dim, model_dim)

        # ---------- temporal embeddings ----------
        self.time_emb = nn.Embedding(max_timesteps, model_dim)

        # ---------- feature-type embeddings ----------
        # feature_types: list of length F with values in {0,1,2,...}
        # e.g. 0=continuous, 1=binary, 2=ordinal, 3=categorical
        if feature_types is not None:
            self.register_buffer(
                "feature_type_ids",
                torch.tensor(feature_types, dtype=torch.long),
            )
            self.feature_type_emb = nn.Embedding(
                int(max(feature_types)) + 1, model_dim
            )
        else:
            self.feature_type_ids = None
            self.feature_type_emb = None

        # ---------- transformer ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, X_mask=None):
        """
        X: (B, T, F)
        X_mask: (B, T, F) or None
        """
        B, T, F = X.shape

        # project features
        H = self.input_proj(X)  # (B, T, D)

        # add temporal embeddings
        t_idx = torch.arange(T, device=X.device)
        H = H + self.time_emb(t_idx)[None, :, :]

        # add feature-type embeddings (shared across time)
        if self.feature_type_emb is not None:
            ft_emb = self.feature_type_emb(self.feature_type_ids)  # (F, D)
            ft_emb = ft_emb.mean(dim=0)  # collapse feature axis
            H = H + ft_emb[None, None, :]

        H = self.dropout(H)

        # optional attention mask (if entire timestep missing)
        attn_mask = None
        if X_mask is not None:
            # mask timestep if *all* features missing
            attn_mask = ~(X_mask.sum(dim=-1) > 0)  # (B, T)

        H = self.encoder(H, src_key_padding_mask=attn_mask)

        return H  # (B, T, D)
