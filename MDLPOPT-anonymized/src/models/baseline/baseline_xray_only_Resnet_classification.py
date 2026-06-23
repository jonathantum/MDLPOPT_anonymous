# ---------- imports ----------
import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import os
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# ---------- project bootstrap ----------
logging.basicConfig(level=logging.INFO)

def get_project_root():
    env = os.environ.get("THESIS_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Project root not found")

PROJECT_ROOT = get_project_root()
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
from models.modules.tabular_encoder_transformer import TabularTransformerEncoder
from models.modules.regression_head import RegressionHead
from models.modules.classification_head import ClassificationHead
from training.trainer import Trainer
from training.losses import masked_mse, weighted_masked_mae, weighted_masked_huber
from utils.oai_ds_utils import OAIPipeline


# ---------- continuous feature indices ----------
cont_idx = [0, 2, 3, 4, 11, 12, 14, 15, 17, 18, 20, 51, 52, 53, 54, 55, 56, 57, 58]
# class TabularModel(nn.Module):
#     def __init__(self, encoder, head):
#         super().__init__()
#         self.encoder = encoder
#         self.head = head

#     def forward(self, X, X_mask):
#         encoded = self.encoder(X, X_mask)  # encoder uses both X and X_mask
#         out = self.head(encoded)           # head just takes the encoded output
#         return out
class XRayModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, **kwargs):
        # 1. Pull what you need safely
        xray_images = kwargs.get("xray_images")
        xray_mask = kwargs.get("xray_mask")
        
        # 2. Safety check: Since this is an X-ray model, 
        # it MUST have xray_images
        if xray_images is None:
            raise ValueError(f"XRayModel received keys {list(kwargs.keys())} "
                             f"but 'xray_images' is missing!")

        # 3. Process
        # Encode images into latent vectors: (B, T, 256)
        encoded = self.encoder(xray_images, xray_mask)
        
        # Pass to classification head
        out = self.head(encoded) 
        return out
# ---------- New Module Imports ----------
from models.modules.xray_encoder import XRayResNetEncoder

def load_data(processed_dir):
    X = np.load(f"{processed_dir}/X.npy")
    X_mask = np.load(f"{processed_dir}/X_mask.npy")
    y = np.load(f"{processed_dir}/y.npy").squeeze(-1)
    y_mask = np.load(f"{processed_dir}/y_mask.npy").squeeze(-1)
    
    # Load the X-ray paths and masks we built
    xray_paths = np.load(f"/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_paths.npy", allow_pickle=True)
    xray_mask = np.load(f"/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_mask.npy")
    
    ids = np.load(f"{processed_dir}/ids.npy", allow_pickle=True)
    return X, X_mask, y, y_mask, xray_paths, xray_mask, ids
# def subsample(X, X_mask, y, y_mask, ids, frac=0.1, seed=0):
#     rng = np.random.default_rng(seed)
#     N = len(X)
#     idx = rng.choice(N, int(frac * N), replace=False)
#     return X[idx], X_mask[idx], y[idx], y_mask[idx], ids[idx]
def subsample_loader(loader, max_samples):
    dataset = loader.dataset
    n = len(dataset)
    if n == 0:
        return loader  # leave empty splits alone
    idx = torch.randperm(n)[:min(max_samples, n)]
    return torch.utils.data.DataLoader(
        Subset(dataset, idx),
        batch_size=loader.batch_size,
        shuffle=True,
        )
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

def evaluate_wsi_classification(y_true, y_pred, class_names=None):
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

    print("=" * 80)



def single_run_main(oai_pipeline, trainloader, val_loader, test_loader, device="cpu", 
                    wsi_threshold=1, input_years=1, target_years=1, 
                    label_aggregation="mean"):

    # 1. Initialize Model Components first
    encoder = XRayResNetEncoder(model_dim=256, pretrained=True)
    head = ClassificationHead(input_dim=256, n_classes=3)
    model = XRayModel(encoder, head).to(device)

    # 2. Use a "temp_trainer" to derive labels for weights
    # We pass the real model here so .to(device) doesn't fail
    temp_trainer = Trainer(
        model=model, 
        optimizer=None, 
        loss_fn=None, 
        device=device,
        task="wsi", 
        wsi_threshold=wsi_threshold,
        input_years=input_years, 
        target_years=target_years,
        label_aggregation=label_aggregation
    )

    y_train = torch.tensor(oai_pipeline.y[oai_pipeline.train_idx])
    y_mask_train = torch.tensor(oai_pipeline.y_mask[oai_pipeline.train_idx])
    
    # Extract training labels to calculate weights
    labels, mask = temp_trainer.derive_wsi_labels(y_train, y_mask_train)
    valid_labels = labels[mask].numpy().astype(int)
    
    class_counts = np.bincount(valid_labels, minlength=3)
    # Total / (Classes * Count)
    weights = len(valid_labels) / (3.0 * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"⚖️ Training Class Counts: {class_counts}")
    print(f"⚖️ Class Weights: {weights}")

    # 3. Initialize REAL Optimizer and Trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights_tensor), # <--- Weighting applied
        device=device,
        class_weights=weights_tensor,
        task="wsi",
        wsi_threshold=wsi_threshold,
        input_years=input_years,
        target_years=target_years,
        label_aggregation=label_aggregation
    )

    # 4. Train for more than 1 epoch!

    trainer.fit(
        train_loader=trainloader,
        val_loader=val_loader,
        epochs=100,
        patience=10,
    )

    # 6. Test
    loss, y_pred, y_true = trainer.evaluate(test_loader)
    
    evaluate_wsi_classification(y_true=y_true, y_pred=y_pred)
    
    return loss, y_pred, y_true, balanced_accuracy_score(y_true, y_pred)# ---------- main ----------
from itertools import product
from sklearn.metrics import balanced_accuracy_score

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Standardize to the /home/ path which is common across nodes
    base_path = Path("/home/anonymous/NDA_processed/images/xray")
    
    # Verify accessibility immediately
    if not base_path.exists():
        print(f"❌ ERROR: base_path {base_path} is NOT accessible!")
        # Fallback to the other path if the first fails
        base_path = Path("/mnt/nfs/homedirs/anonymous/NDA_processed/images/xray")
        print(f"📂 Fallback to base_path: {base_path}")
    
    print(f"📂 Using base_path: {base_path}")
    # Load all data
    X, X_mask, y, y_mask, xray_paths, xray_mask, ids = load_data(
        "data/processed/all_years/2_or_less_missing_target_visits/"
    )

    # Select Bilateral images
    xray_paths_bilateral = xray_paths[:, :, 0]
    xray_mask_bilateral = xray_mask[:, :, 0]


    


    oai_pipeline = OAIPipeline(
        X=X, X_mask=X_mask, y=y, y_mask=y_mask, 
        xray_paths=xray_paths_bilateral, 
        xray_mask=xray_mask_bilateral, 
        ids=ids,
    )
    oai_pipeline.run(
        modalities=["xray"], 
        batch_size=16,
        num_workers=4,
        xray_base_dir=base_path 
    )

    single_run_main(
        oai_pipeline=oai_pipeline, # Pass this first as expected by your new logic
        trainloader=oai_pipeline.train_loader,
        val_loader=oai_pipeline.val_loader,
        test_loader=oai_pipeline.test_loader,
        device=device,
        wsi_threshold=1.0,
        input_years=3,
        target_years=1,
        label_aggregation="year_to_year"
    )
if __name__ == "__main__":
    main()
