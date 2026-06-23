from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
# ============================================================
# Dataset
# ============================================================
import os
from pathlib import Path

class OAIDataset(Dataset):
    def __init__(self, X, X_mask, y, y_mask, xray_paths, xray_mask, ids, 
                 modalities=["tabular"], 
                 xray_base_dir=None, 
                 transform=None):
        self.X = X
        self.X_mask = X_mask
        self.y = y
        self.y_mask = y_mask
        self.xray_paths = np.asarray(xray_paths)
        self.xray_mask = xray_mask
        self.ids = np.asarray(ids)
        self.modalities = modalities
        
        # Ensure xray_base_dir is a Path object
        self.xray_base_dir = Path(xray_base_dir)
        print(f"📂 OAIDataset re-anchoring to: {self.xray_base_dir}")

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if "xray" in self.modalities:
            count = 0
            sample_size = min(500, len(self.xray_paths)) 
            for i in range(sample_size):
                for p in self.xray_paths[i]:
                    if p and str(p) != 'nan':
                        # --- THE FIX ---
                        # Get only the filename (e.g., '9000099L_V00_...jpg')
                        # This removes the incorrect '/home/anonymous/...' prefix
                        fname = os.path.basename(str(p).strip())
                        
                        # Extract the folder (the knee_id, e.g., '9000099L')
                        knee_id = fname.split('_')[0]
                        
                        # Correctly join: /mnt/nfs/... + 9000099L + filename
                        full_p = self.xray_base_dir / knee_id / fname
                        
                        if full_p.exists():
                            count += 1
            print(f"✅ Path check complete: {count} images found using re-anchored paths.")

    def __getitem__(self, idx):
        batch = {
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "y_mask": torch.tensor(self.y_mask[idx], dtype=torch.bool),
            "id": self.ids[idx]
        }

        if "xray" in self.modalities:
            paths = self.xray_paths[idx]
            imgs = []
            actual_valid_mask = [] 
            
            for p in paths:
                img_tensor = None

                if p and str(p) != "nan":
                    fname = os.path.basename(str(p).strip())
                    knee_id = fname.split('_')[0]
                    full_path = self.xray_base_dir / knee_id / fname
                    
                    try:
                        img = Image.open(full_path).convert('RGB')
                        # Apply transform ONLY to real images
                        img_tensor = self.transform(img)
                        is_loaded = True
                    except Exception:
                        img_tensor = None
                
                if img_tensor is None:
                    # Placeholder MUST be zeros and is_loaded MUST be False
                    img_tensor = torch.zeros((3, 256, 256))
                    mask_value = True
                else:
                    mask_value = False

                imgs.append(img_tensor)
                actual_valid_mask.append(mask_value)
            
            batch["xray_images"] = torch.stack(imgs)
            # This overrides your self.xray_mask[idx] with what actually happened
            # Set dtype to bool. True = MASK OUT (Ignore), False = KEEP
            batch["xray_mask"] = torch.tensor(actual_valid_mask, dtype=torch.bool)
            
        return batch
    def __len__(self):
        return self.X.shape[0]


# ============================================================
# Pipeline
# ============================================================
class OAIPipeline:
    def __init__(self, X, X_mask, y, y_mask, xray_paths=None, xray_mask=None, ids=None, save_dir="processed_data", seed=42):
        self.X = X
        self.X_mask = X_mask
        self.y = y
        self.y_mask = y_mask
        self.xray_paths = xray_paths
        self.xray_mask = xray_mask
        self.ids = np.asarray(ids)
        # SANITY CHECK: Ensure all arrays are the same length
        lengths = [len(self.X), len(self.y), len(self.xray_paths), len(self.ids)]
        if len(set(lengths)) > 1:
            raise ValueError(f"Array length mismatch! X:{len(self.X)}, y:{len(self.y)}, "
                             f"paths:{len(self.xray_paths)}, ids:{len(self.ids)}")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(seed)
        self.patient_ids = np.array([pid[:-1] for pid in self.ids])
        self.unique_patients = np.unique(self.patient_ids)

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

    def build_datasets(self, modalities=["tabular"], xray_base_dir=None):
            self.modalities = modalities
            self.xray_base_dir = xray_base_dir 

            def create_ds(idxs, transform=None):
                return OAIDataset(
                    self.X[idxs], self.X_mask[idxs], self.y[idxs], self.y_mask[idxs],
                    self.xray_paths[idxs], self.xray_mask[idxs], self.ids[idxs],
                    modalities=modalities, 
                    transform=transform,
                    xray_base_dir=self.xray_base_dir
                )
            
            # Apply the specific transforms here
            self.train_ds = create_ds(self.train_idx, transform=self.get_train_transforms())
            self.val_ds = create_ds(self.val_idx, transform=self.get_val_transforms())
            self.test_ds = create_ds(self.test_idx, transform=self.get_val_transforms())

    def build_dataloaders(self, batch_size=8, num_workers=4, shuffle=False):
        # num_workers > 0 is essential for fast image loading!
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def run(self, modalities=["tabular"], batch_size=8, num_workers=4, xray_base_dir=None):
            self.split()
            self.build_datasets(modalities=modalities, xray_base_dir=xray_base_dir)
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
            # Small rotations handle patient positioning variance
            transforms.RandomRotation(degrees=10),
            # ColorJitter handles brightness/contrast differences in X-ray machines
            transforms.ColorJitter(brightness=0.1, contrast=0.1),

            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            # ImageNet normalization for the pretrained ResNet
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

