import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

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

# Modular Imports (Ensure these files exist in your src/models/modules/)
from models.modules.TabularYearEncoder import TabularYearEncoder 
from models.modules.MidFusionTransformer import MidFusionTransformer
from models.modules.classification_head import ClassificationHead
from training.losses import FocalLoss
from training.trainer import Trainer
from utils.oai_ds_utils import OAIPipeline

# ---------- Sampling Logic to Fix Mode Collapse ----------

def get_stratified_samplers(y_labels, n_splits=5):
    """Returns indices for Stratified K-Fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(skf.split(np.arange(len(y_labels)), y_labels))

def create_weighted_sampler(labels):
    """Oversamples minority classes (Improving/Worsening) to prevent mean collapse."""
    # Handle ignore_index if labels contains -1
    valid_mask = labels != -1
    clean_labels = labels[valid_mask]
    
    class_sample_count = np.array([len(np.where(clean_labels == t)[0]) for t in np.unique(clean_labels)])
    weight = 1. / class_sample_count
    
    # Assign weight 0 to invalid samples
    samples_weight = np.zeros(len(labels))
    samples_weight[valid_mask] = np.array([weight[t] for t in clean_labels])
    
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))
    return sampler

# ---------- Modular Model Wrapper ----------

class MultimodalTemporalModel(nn.Module):
    def __init__(self, tab_encoder, fusion_module, head):
        super().__init__()
        self.tab_encoder = tab_encoder
        self.fusion_module = fusion_module
        self.head = head

    def forward(self, tabular, tabular_mask=None, **kwargs):
            B, T, F = tabular.shape
            
            # 1. Feature-Wise Masking for the Encoder
            flat_tab = tabular.reshape(B * T, F)
            if tabular_mask is not None:
                #flat_mask = tabular_mask.reshape(B * T, F).float()
                # New mask-aware encoder call
                flat_features = self.tab_encoder(tabular, mask=tabular_mask)
            else:
                flat_features = self.tab_encoder(tabular)
            
            # 2. Reshape back to temporal sequence (B, T, 256)
            #temporal_features = flat_features.reshape(B, T, -1)
            
            # 3. COLLAPSE MASK FOR TRANSFORMER
            # Transformer needs (B, T) where True = "Ignore this year"
            if tabular_mask is not None:
                # any(dim=-1) checks if ANY feature is present. 
                # We invert it with ~ so that if NO features are present, it is True (masked).
                # We take the mask from one feature index or check all features.
                year_exists = tabular_mask.any(dim=-1) # (B, T)
                transformer_mask = ~year_exists        # (B, T)
            else:
                transformer_mask = None

            # 4. Fusion
            # Pass the 2D transformer_mask instead of the 3D tabular_mask
            fused_vector = self.fusion_module(flat_features, transformer_mask)
            
            # 5. Classification
            return self.head(fused_vector)

# ---------- Evaluation Logic ----------

def print_confidence_stats(y_pred_probs, y_true):
    if hasattr(y_pred_probs, 'detach'): y_pred_probs = y_pred_probs.detach().cpu().numpy()
    if hasattr(y_true, 'detach'): y_true = y_true.detach().cpu().numpy()

    valid_mask = (y_true >= 0) & (y_true <= 2)
    y_true_clean = y_true[valid_mask]
    y_probs_clean = y_pred_probs[valid_mask]

    avg_probs = y_probs_clean.mean(axis=0)
    print("\n🔮 AVERAGE PREDICTION CONFIDENCE:")
    class_names = ["Low-Stable (0)", "Increasing (1)", "High-Persistent (2)"]
    for i, name in enumerate(class_names):
        if i < len(avg_probs):
            print(f"  {name}: {avg_probs[i]:.4f}")

def evaluate_classification(y_true, y_pred, y_probs, class_names=None):
    mask = (y_true >= 0) & (y_true <= 2)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if class_names is None: class_names = ["Low-Stable", "Increasing", "High-Persistent"]

    print("\n" + "=" * 80 + "\n📊 CLASSIFICATION METRICS\n" + "=" * 80)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f} | Balanced Acc: {balanced_accuracy_score(y_true, y_pred):.4f}")
    
    print("\n📉 Confusion Matrix (counts):")
    print(confusion_matrix(y_true, y_pred))
    
    print("\n📋 Detailed Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print_confidence_stats(y_probs, y_true)

# ---------- Main Execution ----------

def single_run_main(oai_pipeline, device="cuda", wsi_threshold=1.0, input_years=3, target_years=1):
    # 1. Modular Components
    tab_encoder = TabularYearEncoder(input_dim=67, model_dim=256)
    fusion_layer = MidFusionTransformer(input_dim=256, num_layers=2)
    head = ClassificationHead(input_dim=256, n_classes=3)
    
    model = MultimodalTemporalModel(tab_encoder, fusion_layer, head).to(device)

    # 2. Derive Labels for Weighted Sampler
    temp_trainer = Trainer(model=model, optimizer=None, loss_fn=None, device=device, 
                           input_years=input_years, target_years=target_years, task="classification", label_aggregation="womac_trajectory")

    # We extract training labels to build the sampler
    y_train_raw = torch.tensor(oai_pipeline.y[oai_pipeline.train_idx])
    y_train_mask = torch.tensor(oai_pipeline.y_mask[oai_pipeline.train_idx])
    train_labels, train_mask = temp_trainer.derive_womac_trajectory_labels(y_train_raw, y_train_mask)

    # 3. Create Balanced Loader
    sampler = create_weighted_sampler(train_labels.numpy())
    balanced_train_loader = DataLoader(oai_pipeline.train_ds, batch_size=32, sampler=sampler)

    # 4. Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = FocalLoss(gamma=2.0) # Helps with hard-to-classify progressors
    
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
                      task="classification", label_aggregation="womac_trajectory",
                      input_years=input_years, target_years=target_years)

    trainer.fit(balanced_train_loader, oai_pipeline.val_loader, epochs=100, patience=20, min_delta=0.0001)
    
    # 5. Final Evaluation
    loss, y_pred, y_probs, y_true = trainer.evaluate(oai_pipeline.test_loader)
    return loss, y_pred, y_probs, y_true

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = Path("/mnt/nfs/homedirs/anonymous/NDA_processed/images/xray")
    
    # Load raw data pointers
    X = np.load("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_target_no_surgery/X.npy")
    X_mask = np.load("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_target_no_surgery/X_mask.npy")
    y = np.load("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_target_no_surgery/y.npy").squeeze(-1)
    y_mask = np.load("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_target_no_surgery/y_mask.npy").squeeze(-1)
    xray_paths = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_paths.npy", allow_pickle=True)[:,:,0]
    xray_mask = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_mask.npy")[:,:,0]
    ids = np.load("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_target_no_surgery/ids.npy", allow_pickle=True)

    pipeline = OAIPipeline(X, X_mask, y, y_mask, xray_paths, xray_mask, ids)
    pipeline.run(modalities=["tabular"], batch_size=16, xray_base_dir=base_path)
    #verify_dataset(pipeline.train_ds)
    loss, y_pred, y_probs, y_true = single_run_main(pipeline, device=device, input_years=3, target_years=7)
    evaluate_classification(y_true, y_pred, y_probs)

if __name__ == "__main__":
    main()