"""Dataset definitions for PyTorch.
"""
from typing import List, Optional
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    # If torch is not installed, define a minimal placeholder base class for typing
    Dataset = object


class OAIDataset(Dataset):
    """Simple PyTorch Dataset scaffold for clinical + image samples.

    Inputs/Outputs contract:
    - init inputs: metadata (DataFrame or path), image_dir (str/Path), transforms (callable)
    - __getitem__ returns (x, y) where x is multimodal dict and y is label/value
    - __len__ returns number of samples
    """

    def __init__(self, metadata, image_dir: Optional[str] = None, transforms=None):
        self.metadata = metadata
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # implement loading logic: clinical features + image loading
        row = self.metadata.iloc[idx]
        sample = {
            'clinical': row.to_dict(),
            'image': None,
        }
        label = row.get('label') if 'label' in row else None
        return sample, label
