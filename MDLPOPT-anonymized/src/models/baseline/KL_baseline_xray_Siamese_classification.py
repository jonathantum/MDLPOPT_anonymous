from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import os

# ---------- project bootstrap ----------
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

from models.modules.xray_encoder import XRayResNetEncoder
from models.modules.MidFusionTransformer import MidFusionTransformer
from models.modules.FusionMLP import MLPFusion
from models.modules.classification_head import ClassificationHead
from models.modules.dual_task_head import DualHead
from training.losses import FocalLoss
from training.trainer import Trainer
from utils.oai_ds_utils import OAIPipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------- Modular Model Wrapper ----------

class MultimodalTemporalModel(nn.Module):
    def __init__(self, image_encoder, fusion_module, head):
        super().__init__()
        self.image_encoder = image_encoder
        self.fusion_module = fusion_module
        self.head = head

        self.diff_norm = nn.LayerNorm(256)

    # def forward(self, xray_images, xray_mask, **kwargs):
    #     # xray_images = kwargs.get("xray_images") # (B, T, C, H, W)
    #     # xray_mask = kwargs.get("xray_mask")     # (B, T)
        
    #     # 1. Feature Extraction (Weight Sharing across T)
    #     # Returns (B, T, 256)
    #     img_features = self.image_encoder(xray_images, xray_mask)
        
    #     # 2. Temporal Fusion (Transformer)
    #     # Returns (B, 256) via the CLS token
    #     fused_vector = self.fusion_module(img_features, xray_mask)
        
    #     # 3. Prediction
    #     return self.head(fused_vector, img_features)
    def forward(self, xray_images, xray_mask=None, **kwargs):
        # xray_images shape: [B, T, C, H, W] (e.g., [16, 3, 3, 256, 256])
        B, T, C, H, W = xray_images.shape
        
        # 1. Encode all timepoints using the shared encoder
        # Reshape to [B*T, C, H, W] to process all images at once
        flat_images = xray_images.reshape(B * T, C, H, W)
        flat_features = self.image_encoder(flat_images) # [B*T, 256]
        
        # 2. Reshape back to [B, T, 256]
        features = flat_features.reshape(B, T, -1)
        
        # 3. TIULPIN DIFFERENCE LOGIC
        # Assume T=3 (Year 0, Year 1, Year 2)
        f0 = features[:, 0, :]
        f1 = features[:, 1, :]
        f2 = features[:, 2, :]
        
        diff1 = f1 - f0  # Change from Year 0 to 1
        diff2 = f2 - f1  # Change from Year 1 to 2
        
        diff1 = self.diff_norm(diff1) 
        diff2 = self.diff_norm(diff2)
        # Option A: Stack differences and the latest state
        # This forces the fusion layer to look at the MOVEMENT
        temporal_input = torch.stack([diff1, diff2, f2], dim=1) # [B, 3, 256]
        
        # 4. Pass to Fusion (MLP or Transformer)
        fused_features = self.fusion_module(temporal_input)
        
        # 5. Get Logits
        p_logits, s_logits = self.head(fused_features, flat_features)
        return p_logits, s_logits
# ---------- Evaluation Logic ----------

def evaluate_wsi_classification(y_true, y_pred, y_probs, class_names=None):
    """
    y_true, y_pred: (N,) integer class labels
    """
    # Filter out the ignore_index (usually -1 or -100)
    mask = (y_true >= 0) & (y_true <= 2)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if class_names is None:
        class_names = ["Improving", "Stable", "Worsening"]

    print("\n" + "=" * 80)
    print("📊 W / S / I CLASSIFICATION METRICS")
    print("=" * 80)

    # ---------- basic metrics ----------
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")

    # ---------- class counts ----------
    print("\n🔢 Class distribution (TRUE):")
    for i, name in enumerate(class_names):
        print(f"  {name:10s}: {(y_true == i).sum():5d}")

    print("\n🔢 Class distribution (PREDICTED):")
    for i, name in enumerate(class_names):
        print(f"  {name:10s}: {(y_pred == i).sum():5d}")

    # ---------- confusion matrix ----------
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    print("\n📉 Confusion Matrix (counts):")
    print(cm)

    print("\n📉 Confusion Matrix (row-normalized):")
    np.set_printoptions(precision=3, suppress=True)
    print(cm_norm)

    # ---------- detailed report ----------
    print("\n📋 Per-class Precision / Recall / F1:")
    print("y_true shape:", y_true.shape, "y_pred shape:", y_pred.shape)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0,1,2],
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )
    print_confidence_stats(y_probs, y_true)
    print("=" * 80)

def print_confidence_stats(y_pred_probs, y_true):
    if hasattr(y_pred_probs, 'detach'): y_pred_probs = y_pred_probs.detach().cpu().numpy()
    if hasattr(y_true, 'detach'): y_true = y_true.detach().cpu().numpy()

    # CRITICAL: If y_true was pre-filtered (1102) and y_probs wasn't (1125),
    # we take the last N elements of y_probs to match.
    if len(y_pred_probs) != len(y_true):
        print(f"⚠️ Alignment mismatch: Probs({len(y_pred_probs)}) vs True({len(y_true)}). Aligning...")
        y_pred_probs = y_pred_probs[:len(y_true)] 

    # Now they are both 1102. Masking will work.
    valid_mask = (y_true >= 0) & (y_true <= 2)
    y_true_clean = y_true[valid_mask]
    y_probs_clean = y_pred_probs[valid_mask]

    avg_probs = y_probs_clean.mean(axis=0)
    print("\n🔮 AVERAGE PREDICTION CONFIDENCE:")
    if len(avg_probs) == 2:
        print(f"   Non-Progression: {avg_probs[0]:.4f} | Progression: {avg_probs[1]:.4f}")
    else:
        print(f"   Improving: {avg_probs[0]:.4f} | Stable: {avg_probs[1]:.4f} | Worsening: {avg_probs[2]:.4f}")
    
    for c in range(len(avg_probs)):
        mask_c = (y_true_clean == c)
        if mask_c.any():
            conf = y_probs_clean[mask_c, c].mean()
            print(f"   Avg confidence for True Class {c}: {conf:.4f}")  
# ---------- Main Execution ----------

def single_run_main(oai_pipeline, device="cuda", wsi_threshold=1.0, input_years=3, target_years=1, label_aggregation="year_to_year"):
    # 1. Architecture
    image_encoder = XRayResNetEncoder(model_dim=256, pretrained=True, backbone="resnet18")
    #fusion_layer = MidFusionTransformer(input_dim=256, num_layers=2)
    fusion_layer = MLPFusion(input_dim=256, num_years=input_years, dropout=0.5)
    head = DualHead(input_dim=256)
    model = MultimodalTemporalModel(image_encoder, fusion_layer, head).to(device)

# --- 2. Optimizer with Differential LR ---
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
        {"params": model.fusion_module.parameters(), "lr": 5e-4, "weight_decay": 1e-2},
        {"params": model.head.parameters(), "lr": 1e-4, "weight_decay": 1e-2}
    ])
    print("✅ Optimizer params:", optimizer)
    
    # 3. Initialize Optimizer with these groups
    # --- 3. Scheduler ---
    # T_0: Number of iterations for the first restart (e.g., 10 epochs)
    # T_mult: Factor by which T_0 increases after a restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=1, 
        eta_min=1e-6
    )
    trainer = Trainer(
        model=model, optimizer=optimizer, loss_fn=None,
        device=device, task="p_np_aux", wsi_threshold=wsi_threshold,
        input_years=input_years, target_years=target_years,
        label_aggregation=label_aggregation
    )

    # 3. NOW call the balanced loader
    print("⚖️ Building balanced training loader...")
    # train_loader_balanced = oai_pipeline.get_balanced_train_loader(
    #     trainer=trainer, 
    #     batch_size=16, 
    #     num_workers=4
    # )
    # Punish missing 'Worsening' (Class 2) and 'Improving' (Class 0) much harder
    # Order: [Improving, Stable, Worsening]
    #weights = torch.tensor([5.0, 0.1, 8.0], dtype=torch.float32).to(device)

    # 4. Update trainer with the real loss weights
    y_train = torch.tensor(oai_pipeline.y[oai_pipeline.train_idx])
    # 1. Get the labels (0=Non-Prog, 1=Prog)
    labels, mask = trainer.derive_progression_labels(y_train, torch.tensor(oai_pipeline.y_mask[oai_pipeline.train_idx]))

    # 2. Count only classes 0 and 1
    counts = np.bincount(labels[mask].cpu().numpy().astype(int), minlength=2)

    # 3. Calculate weights and fix the .to(device) syntax
    # counts[0] is Non-Prog, counts[1] is Prog
    weights = torch.tensor(len(labels[mask]) / (2.0 * counts), dtype=torch.float32).to(device) 
    #weights = torch.tensor([1.0, 12.0], dtype=torch.float32).to(device)
    print(f"⚖️ Binary Weights: Non-Prog={weights[0]:.2f}, Prog={weights[1]:.2f}")    
    #trainer.loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1, ignore_index=-100)
    loss_fn = FocalLoss(alpha=weights[1].item(), gamma=1.5, reduction='mean')
    # weights[1].item()
    # FocalLoss(label_smoothing=0.1)
    # 5. Train for more than 1 epoch to see actual results
    #train_loader_balanced
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        scheduler= None,
        loss_fn=loss_fn, # Trainer.step handles its own CE logic
        device=device, 
        task="p_np_aux", 
        class_weights=weights, # <--- ADD THIS LINE
        wsi_threshold=wsi_threshold,
        input_years=input_years, 
        target_years=target_years,
        label_aggregation=label_aggregation
    )
    trainer.fit(oai_pipeline.train_loader, oai_pipeline.val_loader, epochs=50, patience=25, min_delta=0.0001)
    
    # 6. Eval
    loss, y_pred, y_probs, y_true = trainer.evaluate(oai_pipeline.test_loader)

    return loss, y_pred, y_probs, y_true

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = Path("/mnt/nfs/homedirs/anonymous/NDA_processed/images/xray")
    
    # Load raw data pointers
    X = np.load("data/processed/all_years/2_or_less_missing_target_visits/KL_target/X.npy")
    X_mask = np.load("data/processed/all_years/2_or_less_missing_target_visits/KL_target/X_mask.npy")
    y = np.load("data/processed/all_years/2_or_less_missing_target_visits/KL_target/y.npy").squeeze(-1)
    y_mask = np.load("data/processed/all_years/2_or_less_missing_target_visits/KL_target/y_mask.npy").squeeze(-1)
    xray_paths = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_paths.npy", allow_pickle=True)[:,:,0]
    xray_mask = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_mask.npy")[:,:,0]
    ids = np.load("data/processed/all_years/2_or_less_missing_target_visits/ids.npy", allow_pickle=True)

    pipeline = OAIPipeline(X, X_mask, y, y_mask, xray_paths, xray_mask, ids)
    pipeline.run(modalities=["xray"], batch_size=16, xray_base_dir=base_path)
    #verify_dataset(pipeline.train_ds)
    loss, y_pred, y_probs, y_true = single_run_main(pipeline, device=device, wsi_threshold=0.3, input_years=3, target_years=3, label_aggregation="max_delta")
    evaluate_wsi_classification(y_true, y_pred, y_probs, class_names=["Non-Progression", "Progression"])

def verify_dataset(dataset):
    print("\n🔍 DATASET DIAGNOSTIC 🔍")
    for i in range(5):
        sample = dataset[i]
        xray_seq = sample['xray_images'] # (T, C, H, W)
        mask_seq = sample['xray_mask']   # (T)
        
        for t in range(xray_seq.shape[0]):
            img = xray_seq[t]
            is_padded = mask_seq[t]
            
            # Statistics
            mean = img.mean().item()
            std = img.std().item()
            
            status = "✅ DATA FOUND"
            if std < 1e-4:
                status = "🚨 EMPTY/CONSTANT TENSOR"
            if is_padded:
                status = "⚪ PADDING (EXPECTED)"
                
            print(f"Sample {i}, Time {t} | Mean: {mean:.3f} | Std: {std:.3f} | {status}")

# Usage:
# verify_dataset(pipeline.train_ds)
if __name__ == "__main__":
    main()