import torch
import torch.nn as nn
import torchvision.models as models

class XRayResNetEncoder(nn.Module):
    def __init__(self, model_dim: int = 256, pretrained: bool = True, backbone: str = "resnet18"):
        super().__init__()
        
        if backbone == "resnet18":
            self.cnn = models.resnet18(weights="DEFAULT" if pretrained else None)
        elif backbone == "resnet34":
            self.cnn = models.resnet34(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError("Use resnet18 or resnet34.")

        # Extract features and ensure pooling is adaptive for 256x256
        self.features = nn.Sequential(*list(self.cnn.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # This handles 256x256 automatically
        
        num_ftrs = self.cnn.fc.in_features
        
        # 2. Improved Head for 256-dim features
        self.projection = nn.Sequential(
            nn.Linear(num_ftrs, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Dropout(0.4) 
        )
        # num_ftrs = self.cnn.fc.in_features
        
        # # IMPROVED HEAD: BatchNorm and higher Dropout to stop overfitting
        # self.cnn.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, model_dim),
        #     nn.BatchNorm1d(model_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.4) 
        # )

    # def forward(self, images, mask=None):
    #         # images: (B, T, C, 256, 256)
    #         B, T, C, H, W = images.shape
            
    #         # Flatten B and T: (B*T, C, H, W)
    #         x = images.reshape(B * T, C, H, W)        
            
    #         # If using a mask, we can avoid processing zero-padded images 
    #         # but for simplicity, we process all and mask the output.
    #         x = self.features(x)   # (B*T, 512, 8, 8) for 256x256 input
    #         x = self.pool(x)       # (B*T, 512, 1, 1)
    #         x = torch.flatten(x, 1) # (B*T, 512)
            
    #         # Apply projection head
    #         features = self.projection(x) # (B*T, model_dim)
            
    #         # Reshape back to (B, T, model_dim)
    #         out = features.view(B, T, -1)
            
    #         if mask is not None:
    #             # mask: (B, T) -> (B, T, 1)
    #             out = out * mask.unsqueeze(-1)
                
    #         return out
    def forward(self, images, mask=None):
        # 1. Flexible Unpacking
        shape = images.shape
        if len(shape) == 5:
            # If input is (B, T, C, H, W)
            B, T, C, H, W = shape
            x = images.reshape(B * T, C, H, W)
        else:
            # If input is already flattened (B*T, C, H, W)
            x = images
            # We don't know B or T here, so we handle it at the end
            
        # 2. Process through CNN
        x = self.features(x)   
        x = self.pool(x)       
        x = torch.flatten(x, 1) 
        
        # 3. Apply projection head
        features = self.projection(x) # Shape: (N, model_dim)
        
        # 4. Reshape back ONLY if we started with 5D
        if len(shape) == 5:
            out = features.view(B, T, -1)
            if mask is not None:
                out = out * mask.unsqueeze(-1)
            return out
        
        # Otherwise return the flat features (N, model_dim)
        return features