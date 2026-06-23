import torch
import torch.nn as nn
class DualHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.progression_head = nn.Linear(input_dim, 2) # Binary: p_np
        self.state_head = nn.Linear(input_dim, 5)       # KL Grades: 0, 1, 2, 3, 4


    def forward(self, fused_features, flat_features):
        # fused_features: [B, D] (Output from MLPFusion)
        # flat_features: [B*T, D] (Raw features from Image Encoder)
        
        # 1. Main Task: Progression (Binary)
        # This uses the fused "difference" features
        p_logits = self.progression_head(fused_features) # [B, 2]
        
        # 2. Aux Task: State (KL Grade 0-4)
        # This uses the flat features because we want a prediction for every image
        s_logits = self.state_head(flat_features)   # [B*T, 5]
        
        return p_logits, s_logits