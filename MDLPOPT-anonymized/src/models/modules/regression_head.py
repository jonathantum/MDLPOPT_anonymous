import torch
import torch.nn as nn
import torch.nn.functional as F
class RegressionHead(nn.Module):
    def __init__(self, input_dim, T_out, hidden_dim=128):
        """
        Args:
            input_dim: dimension of encoder output
            T_out: number of future timesteps
        """
        super().__init__()
        self.T_out = T_out
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for previous step
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoded, prev_step=None):
        """
        encoded: (B, 1, hidden)
        prev_step: (B, 1)
        Returns: (B, 1)
        """
        B = encoded.shape[0]
        x = encoded[:, -1, :]  # last timestep
        if prev_step is None:
            prev_step = torch.zeros(B, 1, device=x.device)
        x = torch.cat([x, prev_step], dim=-1)
        out = self.mlp(x)
        return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class RegressionHead(nn.Module):
#     """
#     Maps:
#       (B, D)       -> (B, T_out)
#       (B, T_in, D) -> (B, T_out)
#     """

#     def __init__(self, input_dim: int, T_out: int):
#         super().__init__()
#         self.T_out = T_out

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.ReLU(),
#             nn.Linear(input_dim, 1),
#         )

#     def forward(self, H):

#         if H.ndim == 2:
#             # -------- pooled encoder case --------
#             # H: (B, D)
#             y = self.net(H)            # (B, 1)
#             y = y.expand(-1, self.T_out)  # (B, T_out)

#         elif H.ndim == 3:
#             # -------- sequence encoder case --------
#             # H: (B, T_in, D)
#             y = self.net(H).squeeze(-1)  # (B, T_in)

#             if y.shape[1] != self.T_out:
#                 y = F.interpolate(
#                     y.unsqueeze(1),       # (B, 1, T_in)
#                     size=self.T_out,
#                     mode="linear",
#                     align_corners=False,
#                 ).squeeze(1)               # (B, T_out)

#         else:
#             raise RuntimeError(f"Unsupported input shape: {H.shape}")

#         return y
