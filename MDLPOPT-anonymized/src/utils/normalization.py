import torch

class StandardScaler:
    def __init__(self, eps=1e-6):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, x, mask=None):
        """
        x: (N, T, F)
        mask: (N, T, F) or None
        """
        if mask is None:
            mask = torch.ones_like(x)

        valid = mask > 0
        x_valid = x[valid]

        self.mean = x_valid.mean(dim=0)
        self.std = x_valid.std(dim=0)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def fit_transform(self, x, mask=None):
        self.fit(x, mask)
        return self.transform(x)
