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
class TabularModel(nn.Module):
    def __init__(self, encoder, head, autoreg=False, teacher_forcing_ratio=0.1):
        """
        Args:
            encoder: nn.Module (Transformer or MLP)
            head: RegressionHead
            autoreg: if True, use auto-regressive decoding
            teacher_forcing_ratio: probability of using ground truth during training
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.autoreg = autoreg
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, X, X_mask=None, y=None, y_mask=None):
        """
        X: (B, T_in, F)
        y: (B, T_out) optional, used for teacher forcing
        """
        B, T_in, F = X.shape
        device = X.device
        T_out = 3  # y.shape[1] if y is not None else self.head.T_out

        # encode baseline / history
        encoded = self.encoder(X, X_mask)  # Transformer: (B,T,H), MLP: (B,H)
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)  # make it (B,1,H) for auto-regression

        # ---- standard all-at-once prediction ----
        if not self.autoreg:
            return self.head(encoded)

        # ---- auto-regressive decoding ----
        T_context = 2  # first two WOMAC years are inputs
        preds = []
        prev = None

        for t in range(T_out):

            # --- CONTEXT STEPS (do NOT learn from them) ---
            if t < T_context:
                # always feed ground truth, never model output
                inp = y[:, t:t+1] if y is not None else torch.zeros(B, 1, device=device)
                
                # optional: skip prediction entirely
                out_t = self.head(encoded, prev_step=inp)
                out_t = self.head(encoded, prev_step=inp)
                if prev is not None:
                    out_t = prev + out_t  # add residual

                preds.append(out_t)
                prev = inp  # IMPORTANT: use GT, not prediction
                continue

            # --- FUTURE STEPS (real learning starts here) ---
            if y is not None and np.random.rand() < self.teacher_forcing_ratio:
                inp = y[:, t:t+1]
            else:
                inp = prev

            out_t = self.head(encoded, prev_step=inp)
            preds.append(out_t)
            prev = out_t

        return torch.cat(preds, dim=1)  # (B, T_out)
# ---------- data loader ----------
def load_data(processed_dir):
    X = np.load(f"{processed_dir}/X.npy")
    X_mask = np.load(f"{processed_dir}/X_mask.npy")
    y = np.load(f"{processed_dir}/y.npy").squeeze(-1)
    y_mask = np.load(f"{processed_dir}/y_mask.npy").squeeze(-1)
    ids = np.load(f"{processed_dir}/ids.npy", allow_pickle=True)
    return X, X_mask, y, y_mask, ids

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

def single_run_main(trainloader,
                    val_loader,
                    test_loader,
                    X,
                    device="cpu",
                    F=None,
                    T_out=None,
                    wsi_threshold=1,
                    input_years=1,
                    target_years=1,
                    label_aggregation="mean"):

    # ---------- build model ----------
    encoder = TabularTransformerEncoder(
        input_dim=F,
        model_dim=256,
        num_layers=4,
        num_heads=8,
        max_timesteps=X.shape[1],
    )

    head = ClassificationHead(input_dim=256, n_classes=3)
    #head = RegressionHead(input_dim=256, T_out=target_years)

    # ---------- optimizer ----------
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # ---------- trainer ----------
    trainer = Trainer(
        model=TabularModel(encoder, head, autoreg=False),
        optimizer=optimizer,
        loss_fn=None,              
        device=device,
        cont_idx=cont_idx,
        task="wsi",
        wsi_threshold=wsi_threshold,
        input_years=input_years,
        target_years=target_years,
        label_aggregation=label_aggregation
    )


    # compute stats for continuous features
    trainer.set_cont_stats(X)
    # trainer.set_y_stats(y)

    # ---------- train ----------
    trainer.fit(
        train_loader=trainloader,
        val_loader=val_loader,
        epochs=50,
        patience=20,
        min_delta=1e-4,
    )

    # ---------- test ----------
    loss, y_pred, y_true = trainer.evaluate(test_loader)

#    print("Accuracy:", accuracy_score(y_true, y_pred))
 #   print("Balanced Acc:", balanced_accuracy_score(y_true, y_pred))
   # from sklearn.metrics import confusion_matrix
  #  print(confusion_matrix(y_true, y_pred))


    evaluate_wsi_classification(
        y_true=y_true,
        y_pred=y_pred,
        class_names=["Improving", "Stable", "Worsening"],
    )
    return loss, y_pred, y_true, balanced_accuracy_score(y_true, y_pred)
# ---------- main ----------
from itertools import product
from sklearn.metrics import balanced_accuracy_score

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load data
    X, X_mask, y, y_mask, ids = load_data(
        "data/processed/all_years/2_or_less_missing_target_visits/"
    )

    pipeline = OAIPipeline(X=X, X_mask=X_mask, y=y, y_mask=y_mask, ids=ids)
    pipeline.run(batch_size=64, shuffle=True, num_workers=4)
    # pipeline.train_loader = subsample_loader(pipeline.train_loader, max_samples=32)
    # pipeline.val_loader = subsample_loader(pipeline.val_loader, max_samples=16)
    # pipeline.test_loader = subsample_loader(pipeline.test_loader, max_samples=16)
    print("train_loader size:", len(pipeline.train_loader.dataset))
    print("val_loader size:", len(pipeline.val_loader.dataset))
    print("test_loader size:", len(pipeline.test_loader.dataset))

    F = X.shape[-1]
    T_out = y.shape[1]

    # Experiment parameters
    input_years_list = [3] # list(range(1, 4)) # [2] #
    target_years_list = list(range(1, 9))  # will filter in loop  [1] # 
    aggregation_list = ["year_to_year"] #,"mean",  "baseline_to_last", "last_year_to_whole"]

    results = []

    for i in input_years_list:
        for t in  range(1, 11 - i): # and target_years_list:  # ensures we don't go past T
            for l in aggregation_list:
                #print(f"\n🔹 Running experiment: Input Years ={i}, Target Years={t}, Label Agg={l}")
                model_name = f"Trainer_input_{i}_target_{t}_label_{l}"
                print(f"\n🔹 Running experiment: {model_name}")

                loss, y_pred, y_true, bal_acc = single_run_main(
                    trainloader=pipeline.train_loader,
                    val_loader=pipeline.val_loader,
                    test_loader=pipeline.test_loader,
                    X=X,
                    device=device,
                    F=F,
                    T_out=T_out,
                    wsi_threshold=3.0,
                    input_years=i,
                    target_years=t,
                    label_aggregation=l,
                )

                results.append({
                    "model_name": model_name,
                    "loss": loss,
                    "balanced_accuracy": bal_acc,
                    "y_true": y_true,
                    "y_pred": y_pred
                })

    # Optionally, print summary
    print("\n✅ All experiments completed. Summary:")
    for r in results:
        print(f"{r['model_name']}: Balanced Acc = {r['balanced_accuracy']:.4f}, Loss = {r['loss']:.4f}")
    print("Highest Balanced Accuracy achieved:", max(r['balanced_accuracy'] for r in results))
    print("Lowest Loss achieved:", min(r['loss'] for r in results))
    print("Best model by Balanced Accuracy:", max(results, key=lambda x: x['balanced_accuracy'])['model_name'])

if __name__ == "__main__":
    main()
