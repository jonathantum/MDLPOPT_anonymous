from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import time
from torchvision.transforms import v2
# ============================================================
# Dataset
# ============================================================
import os
from pathlib import Path

class OAIDataset(Dataset):
    def __init__(self, X, X_mask, y, y_mask, xray_paths, xray_mask, ids, 
                    mri_sequences=None, # Update this name here
                    modalities=["tabular"], 
                    xray_base_dir=None, 
                    transform=None, stride=4):
        self.X = X
        self.X_mask = X_mask
        self.y = y
        self.y_mask = y_mask
        self.xray_paths = np.asarray(xray_paths)
        self.xray_mask = xray_mask
        self.ids = np.asarray(ids)
        
        # This now stores the dictionary of sequence data
        self.mri_sequences_data = mri_sequences or {}
        
        self.modalities = modalities
        self.xray_base_dir = Path(xray_base_dir) if xray_base_dir else None
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.stride = stride

    def __getitem__(self, idx):
        knee_id = self.ids[idx]
        batch = {
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "y_mask": torch.tensor(self.y_mask[idx], dtype=torch.bool),
            "id": knee_id
        }

        # Determine training mode for random offset
        is_train = self.transform and hasattr(self.transform, 'transforms') and len(self.transform.transforms) > 3
        offset = 0
        if is_train and self.stride > 1:
            offset = torch.randint(0, self.stride, (1,)).item()        
        # Enhanced Jitter: Randomly dropout slices to prevent anchor-point overfitting
        slice_dropout_prob = 0.1 if is_train else 0.0

        # 1. Tabular Data
        if "tabular" in self.modalities:
            batch["tabular"] = torch.tensor(self.X[idx], dtype=torch.float32)
            batch["tabular_mask"] = torch.tensor(self.X_mask[idx], dtype=torch.bool)

        # 2. X-ray Data
        if "xray" in self.modalities:
            batch["xray_images"], batch["xray_mask"] = self._load_xrays(idx)

        # 3. MRI Data (Hybrid 2D/3D Logic)
        mri_keys = [m for m in self.modalities if m.startswith("mri_")]
        
        base_depth = 160 
        target_size = (256, 256)
        roi_size = (192, 192) 
        
        # Determine target depth based on stride mode
        if self.stride == -1:
            global_target_depth = 16
        else:
            global_target_depth = (base_depth // self.stride)
        
        for seq_key in mri_keys:
            if seq_key in self.mri_sequences_data:
                paths_matrix = self.mri_sequences_data[seq_key]["paths"]
                mask_matrix = self.mri_sequences_data[seq_key]["mask"]
                
                temporal_volumes = []
                for t in range(paths_matrix.shape[1]):
                    path = paths_matrix[idx, t]
                    mask_val = mask_matrix[idx, t]
                    final_vol = torch.zeros((global_target_depth, *target_size))

                    if path and not mask_val and Path(path).exists():
                        try:
                            vol_mmap = np.load(path, mmap_mode='r')
                            d_orig = vol_mmap.shape[0] if vol_mmap.ndim == 3 else 1
                            
                            if d_orig > 1:
                                if self.stride == -1:
                                    # --- NEW: Center 16 Slices Logic ---
                                    center_idx = d_orig // 2
                                    start_idx = max(0, center_idx - 8)
                                    end_idx = min(d_orig, start_idx + 16)
                                    indices = np.arange(start_idx, end_idx)
                                else:
                                    # Standard Strided Logic
                                    start_idx = max(0, (d_orig - base_depth) // 2)
                                    indices = np.arange(start_idx + offset, start_idx + base_depth, self.stride)
                                    indices = indices[indices < d_orig]
                            else:
                                indices = np.array([0])

                            if len(indices) > 0:
                                vol_data = np.array(vol_mmap[indices])
                                if vol_data.ndim == 2: vol_data = vol_data[np.newaxis, ...]

                                # ... (Rest of your spatial jitter and resizing logic remains the same) ...
                                h, w = vol_data.shape[-2:]
                                if is_train:
                                    crop_h, crop_w = int(roi_size[0] * 0.9), int(roi_size[1] * 0.9)
                                    i_start = torch.randint(0, h - crop_h + 1, (1,)).item()
                                    j_start = torch.randint(0, w - crop_w + 1, (1,)).item()
                                    vol_cropped = vol_data[:, i_start:i_start+crop_h, j_start:j_start+crop_w]
                                else:
                                    th, tw = min(roi_size[0], h), min(roi_size[1], w)
                                    i_start, j_start = (h - th) // 2, (w - tw) // 2
                                    vol_cropped = vol_data[:, i_start:i_start+th, j_start:j_start+tw]

                                img_tensor = torch.from_numpy(vol_cropped.copy()).float()
                                if img_tensor.shape[-2:] != target_size:
                                    img_tensor = TF.resize(img_tensor, target_size, antialias=True)

                                # Normalization
                                p99 = torch.quantile(img_tensor, 0.99)
                                img_tensor = torch.clamp(img_tensor, 0, p99) / (p99 + 1e-6)

                                # --- Enhanced Slice Dropout (Only if not in fixed 16-slice mode) ---
                                if self.stride != -1 and is_train and torch.rand(1).item() < slice_dropout_prob:
                                    num_slices = img_tensor.shape[0]
                                    drop_mask = torch.rand(num_slices) > 0.15
                                    img_tensor = img_tensor * drop_mask.view(-1, 1, 1)

                                num_to_fill = min(img_tensor.shape[0], global_target_depth)
                                final_vol[:num_to_fill] = img_tensor[:num_to_fill]
                            
                            temporal_volumes.append(final_vol)

                        except Exception as e:
                            temporal_volumes.append(final_vol) 
                    else:
                        temporal_volumes.append(final_vol) 

                batch[seq_key] = torch.stack(temporal_volumes) 
                batch[f"{seq_key}_mask"] = torch.tensor(mask_matrix[idx], dtype=torch.bool)
        return batch
    def _load_xrays(self, idx):
        paths = self.xray_paths[idx]
        imgs = []
        masks = []
        for p in paths:
            img_tensor = torch.zeros((3, 256, 256))
            mask_val = True
            if p and str(p) != "nan":
                fname = os.path.basename(str(p).strip())
                knee_id = fname.split('_')[0]
                full_path = self.xray_base_dir / knee_id / fname
                try:
                    img = Image.open(full_path).convert('RGB')
                    img_tensor = self.transform(img)
                    mask_val = False
                except: pass
            imgs.append(img_tensor)
            masks.append(mask_val)
        return torch.stack(imgs), torch.tensor(masks, dtype=torch.bool)

    def __len__(self):
        return self.X.shape[0]

# ============================================================
# Pipeline
# ============================================================
class OAIPipeline:
    def __init__(self, X, X_mask, y, y_mask, xray_paths, xray_mask, ids, 
                 mri_paths=None, mri_mask=None, save_dir="processed_data", seed=42):
        self.X = X
        self.X_mask = X_mask
        print("X mask shape:", self.X_mask.shape)
        self.y = y
        self.y_mask = y_mask
        self.xray_paths = xray_paths
        self.xray_mask = xray_mask
        self.ids = np.asarray(ids)
        print(f"OAIPipeline initialized with {len(self.ids)} samples.")
        self.mri_paths = mri_paths
        self.mri_mask = mri_mask
        self.mri_registry = {} # Dictionary to store different MRI sequences
        # SANITY CHECK: Ensure all arrays are the same length
        lengths = [len(self.X), len(self.y), len(self.xray_paths), len(self.ids)]
        if len(set(lengths)) > 1:
            raise ValueError(f"Array length mismatch! X:{len(self.X)}, y:{len(self.y)}, "
                             f"paths:{len(self.xray_paths)}, ids:{len(self.ids)}")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(seed)
        self.patient_ids = np.array([pid[:-1] for pid in self.ids])
        print("length patient_ids", len(self.patient_ids))
        self.unique_patients = np.unique(self.patient_ids)
        print(f"Found {len(self.unique_patients)} unique patients.")
    def add_mri_sequence(self, name, paths, mask):
            """Register a specific MRI sequence (e.g., 'mri_COR_IW_TSE')"""
            self.mri_registry[name] = {
                "paths": np.asarray(paths),
                "mask": np.asarray(mask)
            }
            print(f"✅ Registered sequence: {name}")
    def split(self, train_frac=0.7, val_frac=0.15, test_frac=0.15):
        patients = self.unique_patients.copy()
        np.random.shuffle(patients)
        n = len(patients)
        n_train, n_val = int(n * train_frac), int(n * val_frac)
        
        self.train_patients = patients[:n_train]
        self.val_patients = patients[n_train:n_train + n_val]
        self.test_patients = patients[n_train + n_val:]
        
        self.train_idx = np.where(np.isin(self.patient_ids, self.train_patients))[0]
        self.val_idx = np.where(np.isin(self.patient_ids, self.val_patients))[0]
        self.test_idx = np.where(np.isin(self.patient_ids, self.test_patients))[0]
        print(f"✅ Split: Train={len(self.train_idx)}, Val={len(self.val_idx)}, Test={len(self.test_idx)}")

    def build_datasets(self, modalities=["tabular"], xray_base_dir=None, stride=4):
        self.modalities = modalities
        self.xray_base_dir = xray_base_dir 
        self.stride = stride

        # We define the helper function to take idxs and transform
        def create_ds(idxs, transform):
            # Create a dictionary of MRI data filtered for these specific indices
            split_mri = {
                name: {
                    "paths": data["paths"][idxs],
                    "mask": data["mask"][idxs]
                } for name, data in self.mri_registry.items()
            }

            return OAIDataset(
                X=self.X[idxs], 
                X_mask=self.X_mask[idxs], 
                y=self.y[idxs], 
                y_mask=self.y_mask[idxs],
                xray_paths=self.xray_paths[idxs], 
                xray_mask=self.xray_mask[idxs], 
                ids=self.ids[idxs],
                mri_sequences=split_mri, # This is the registry dict
                modalities=modalities,
                transform=transform,
                xray_base_dir=self.xray_base_dir,
                stride=self.stride
            )
        
        # Corrected calls: passing idxs first, then transform
        self.train_ds = create_ds(self.train_idx, self.get_train_transforms())
        self.val_ds = create_ds(self.val_idx, self.get_val_transforms())
        self.test_ds = create_ds(self.test_idx, self.get_val_transforms())
    def build_dataloaders(self, batch_size=8, num_workers=4, shuffle=False):
        # num_workers > 0 is essential for fast image loading!
        #self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def run(self, modalities=["tabular"], batch_size=8, num_workers=4, xray_base_dir=None, mri_base_dir=None, stride=8):
            self.split()
            self.build_datasets(modalities=modalities, xray_base_dir=xray_base_dir, stride=stride)
            self.build_dataloaders(batch_size=batch_size, num_workers=num_workers)
            return self
    
    @staticmethod
    def get_train_transforms(img_size=256):
        """
        Augmentations for Training: 
        Forces the model to learn features, not noise.
        """
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # RandomHorizontalFlip effectively doubles your training data
            transforms.RandomHorizontalFlip(p=0.5),
            # Elastic Transform: Mimics tissue compression/different anatomy
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            # Small rotations handle patient positioning variance
            transforms.RandomRotation(degrees=10),
            # ColorJitter handles brightness/contrast differences in X-ray machines
            transforms.ColorJitter(brightness=0.1, contrast=0.1),

            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            # ImageNet normalization for the pretrained ResNet
            # Add Noise: Mimics scanner grain
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
            
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    @staticmethod
    def get_val_transforms(img_size=256):
        """
        Standard transforms for Val/Test:
        No randomness here to ensure consistent evaluation.
        """
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    def _build_indices(self):
        def idx_for(pats):
            return np.where(np.isin(self.patient_ids, pats))[0]

        self.train_idx = idx_for(self.train_patients)
        self.val_idx = idx_for(self.val_patients)
        self.test_idx = idx_for(self.test_patients)

    def _check_no_leakage(self):
        def pats(idxs):
            return set(self.patient_ids[idxs])

        assert pats(self.train_idx).isdisjoint(pats(self.val_idx))
        assert pats(self.train_idx).isdisjoint(pats(self.test_idx))
        assert pats(self.val_idx).isdisjoint(pats(self.test_idx))

        print("✅ No patient leakage detected")

    def get_balanced_train_loader(self, trainer, batch_size=16, num_workers=4):
        """
        Calculates weights based on WSI labels and returns a balanced DataLoader.
        """
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
        
        # 1. Get training labels using the trainer's logic
        y_train = torch.tensor(self.y[self.train_idx])
        y_mask_train = torch.tensor(self.y_mask[self.train_idx])
        
        labels, mask = trainer.derive_wsi_labels(y_train, y_mask_train)
        
        # 2. Filter labels to only include those where mask is True (valid sequences)
        valid_labels = labels[mask].cpu().numpy().astype(int)
        
        # 3. Calculate Weights: Higher weight for minority classes
        class_counts = np.bincount(valid_labels, minlength=3)
        # Avoid division by zero
        class_weights = 1.0 / (class_counts + 1e-6)
        
        # 4. Map each VALID sample to its specific class weight
        sample_weights = np.array([class_weights[t] for t in valid_labels])
        
        # 5. Create Sampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"✅ Balanced sampler created with class counts: {class_counts}")
        print(f"   Class weights: {class_weights}")
        # 6. We must use a Subset of the training dataset that matches the 'mask' 
        # length if derive_wsi_labels filtered out any invalid subjects.
        from torch.utils.data import Subset
        valid_indices = np.arange(len(self.train_ds))[mask.cpu().numpy()]
        train_subset = Subset(self.train_ds, valid_indices)

        return DataLoader(
            train_subset, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=num_workers,
            pin_memory=True
        )
    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------
    def save(self):
        np.save(self.save_dir / "X.npy", self.X)
        np.save(self.save_dir / "X_mask.npy", self.X_mask)
        np.save(self.save_dir / "y.npy", self.y)
        np.save(self.save_dir / "y_mask.npy", self.y_mask)
        np.save(self.save_dir / "xray_paths.npy", self.xray_paths)
        np.save(self.save_dir / "xray_mask.npy", self.xray_mask)
        np.save(self.save_dir / "ids.npy", self.ids)

        splits = {
            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
            "train_patients": self.train_patients,
            "val_patients": self.val_patients,
            "test_patients": self.test_patients,
        }

        with open(self.save_dir / "splits.pkl", "wb") as f:
            pickle.dump(splits, f)

        print(f"Saved processed data to {self.save_dir}")

    # --------------------------------------------------------
    # ANALYSIS / SANITY CHECK
    # --------------------------------------------------------
    def analyze(self, show_examples=2):
        print("\n=== SHAPES ===")
        print("X:", self.X.shape)
        print("X_mask:", self.X_mask.shape)
        print("y:", self.y.shape)
        print("y_mask:", self.y_mask.shape)

        print("\n=== SPLITS (samples) ===")
        print("Train:", len(self.train_idx))
        print("Val:", len(self.val_idx))
        print("Test:", len(self.test_idx))

        print("\n=== SAMPLE IDS ===")
        print("Train:", self.ids[self.train_idx][:show_examples])
        print("Val:", self.ids[self.val_idx][:show_examples])
        print("Test:", self.ids[self.test_idx][:show_examples])

        if hasattr(self, "train_loader"):
            batch = next(iter(self.train_loader))
            print("\n=== BATCH CHECK ===")
            print("Batch X:", batch["X"].shape)
            print("Batch y:", batch["y"].shape)

