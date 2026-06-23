"""Transforms and augmentations for imaging data.

Provide wrappers over torchvision or nibabel pipelines.
"""

try:
    from torchvision import transforms as T
except Exception:
    T = None


def get_xray_transforms(train: bool = True):
    if T is None:
        return None
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
    else:
        return T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
