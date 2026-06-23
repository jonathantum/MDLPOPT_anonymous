import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------- Project Bootstrap ----------
logging.basicConfig(level=logging.INFO)

def get_project_root():
    env = os.environ.get("THESIS_PROJECT_ROOT")
    if env: return Path(env).resolve()
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "src").exists(): return parent
    raise RuntimeError("Project root not found")

PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Modular Imports
from models.modules.mri_encoder import MRIEncoder 
from models.modules.MidFusionTransformer import MidFusionTransformer
from models.modules.classification_head import ClassificationHead
from training.losses import FocalLoss
from training.trainer import Trainer
from utils.oai_ds_utils import OAIPipeline
from models.baseline.baseline_tabular_only_transformer_classification_2 import evaluate_classification

# ---------- Sampling Logic ----------

def create_weighted_sampler(labels):
    """Oversamples minority classes to prevent the model from ignoring 'Increasing' pain."""
    valid_mask = labels != -1
    clean_labels = labels[valid_mask]
    
    class_sample_count = np.array([len(np.where(clean_labels == t)[0]) for t in np.unique(clean_labels)])
    weight = 1. / class_sample_count
    
    samples_weight = np.zeros(len(labels))
    samples_weight[valid_mask] = np.array([weight[int(t)] for t in clean_labels])
    
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))
    return sampler

# ---------- MRI Temporal Model ----------

class MRITemporalModel(nn.Module):
    def __init__(self, mri_encoder, fusion_module, head):
        super().__init__()
        self.mri_encoder = mri_encoder
        self.fusion_module = fusion_module
        self.head = head

    def forward(self, **kwargs):
        # 1. Identify MRI inputs dynamically
        mri_keys = [k for k in kwargs.keys() if k.startswith("mri_") and not k.endswith("_mask")]
        if not mri_keys:
            raise ValueError(f"Model received no MRI inputs. Keys: {list(kwargs.keys())}")
        
        mri_key = mri_keys[0]
        mri_data = kwargs[mri_key] # Shape: [B, T, C, H, W]
        mri_mask = kwargs.get(f"{mri_key}_mask", None) # Shape: [B, T]

        B, T, C, H, W = mri_data.shape
        
        # 2. Flatten time for the CNN Encoder
        flat_mri = mri_data.reshape(B * T, C, H, W)
        mri_feats = self.mri_encoder(flat_mri) # (B*T, 256)
        
        # 3. Reshape back for Transformer: (B, T, 256)
        mri_tokens = mri_feats.reshape(B, T, -1)
        
        # 4. Temporal Fusion (Transformer)
        # combined_masks should be (B, T) where True = masked/ignored
        fused_vector = self.fusion_module(mri_tokens, mri_mask)
        
        # 5. Classification Head
        return self.head(fused_vector)

# ---------- Main Execution ----------

def single_run_main(oai_pipeline, device="cuda", input_years=3, target_years=7):
    # 1. Build Components
    mri_encoder = MRIEncoder(model_dim=256)
    fusion_layer = MidFusionTransformer(input_dim=256, num_layers=2)
    head = ClassificationHead(input_dim=256, n_classes=3)
    
    model = MRITemporalModel(mri_encoder, fusion_layer, head).to(device)

    # 2. Calculate Weights for Sampler
    # Use a dummy trainer to get trajectory labels from the raw Y sequence
    temp_trainer = Trainer(model=model, optimizer=None, loss_fn=None, device=device, 
                           input_years=input_years, target_years=target_years, 
                           task="classification", label_aggregation="womac_trajectory")

    y_train_raw = torch.tensor(oai_pipeline.y[oai_pipeline.train_idx])
    y_train_mask = torch.tensor(oai_pipeline.y_mask[oai_pipeline.train_idx])
    train_labels, _ = temp_trainer.derive_womac_trajectory_labels(y_train_raw, y_train_mask)

    # 3. Create Balanced DataLoaders
    sampler = create_weighted_sampler(train_labels.numpy())
    # Note: sampler and shuffle=True are mutually exclusive
    balanced_train_loader = DataLoader(oai_pipeline.train_ds, batch_size=16, sampler=sampler)

    # --- STAGE 1: WARM-UP (Freeze Encoder) ---
    print("\n❄️ STAGE 1: Training Head & Transformer (Encoder Frozen)")
    for param in model.mri_encoder.parameters():
        param.requires_grad = False
    
    optimizer_v1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = FocalLoss(gamma=3.0) # Increased gamma to push for minority classes
    
    trainer = Trainer(model=model, optimizer=optimizer_v1, loss_fn=loss_fn, device=device,
                      task="classification", label_aggregation="womac_trajectory",
                      input_years=input_years, target_years=target_years)

    trainer.fit(balanced_train_loader, oai_pipeline.val_loader, epochs=5, patience=2)

    # --- STAGE 2: FINE-TUNING (Unfreeze All) ---
    print("\n🔥 STAGE 2: Full Model Fine-tuning (Encoder Unfrozen)")
    for param in model.mri_encoder.parameters():
        param.requires_grad = True
    
    # Use a much smaller learning rate for fine-tuning
    optimizer_v2 = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    trainer.optimizer = optimizer_v2
    
    trainer.fit(balanced_train_loader, oai_pipeline.val_loader, epochs=45, patience=10, min_delta=0.001)
    
    # 5. Final Evaluation
    return trainer.evaluate(oai_pipeline.test_loader)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data Paths
    data_dir = Path("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_target_no_surgery")
    
    X = np.load(data_dir / "X.npy")
    X_mask = np.load(data_dir / "X_mask.npy")
    y = np.load(data_dir / "y.npy").squeeze(-1)
    y_mask = np.load(data_dir / "y_mask.npy").squeeze(-1)
    ids = np.load(data_dir / "ids.npy", allow_pickle=True)
    
    # MRI Paths & Masks
    mri_paths = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/images/mri/SAG_3D_DESS/mri_paths.npy", allow_pickle=True)
    mri_mask = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/images/mri/SAG_3D_DESS/mri_mask.npy")

    # Pipeline setup
    # Xray paths are passed as None/dummy if only training MRI
    dummy_xray = np.zeros((len(X), 0)) # Empty array with matching length
    pipeline = OAIPipeline(
        X=X, X_mask=X_mask, y=y, y_mask=y_mask, 
        xray_paths=dummy_xray, xray_mask=dummy_xray, 
        ids=ids, 
        mri_paths=mri_paths, mri_mask=mri_mask
    )

    pipeline.run(modalities=["mri_SAG_3D_DESS"], batch_size=16)

    # Start Run
    loss, y_pred, y_probs, y_true = single_run_main(pipeline, device=device, input_years=1, target_years=7)
    
    # Performance Report
    evaluate_classification(y_true, y_pred, y_probs)

if __name__ == "__main__":
    main()