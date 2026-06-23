import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import os
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Subset


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

# ---------- imports ----------
from models.modules.tabular_encoder_transformer import TabularTransformerEncoder
from models.modules.regression_head import RegressionHead
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
        T_out = 4  # y.shape[1] if y is not None else self.head.T_out

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
# ---------- main ----------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load data
    X, X_mask, y, y_mask, ids = load_data("data/processed/dataset_2_or_less_missing/small_dataset_10")  # full_dataset small_dataset_10 small_dataset_50 small_dataset_100
    # X, X_mask, y, y_mask, ids = subsample(X, X_mask, y, y_mask, ids, frac=0.05, seed=42)
    pipeline = OAIPipeline(X=X, X_mask=X_mask, y=y, y_mask=y_mask, ids=ids)
    pipeline.run(batch_size=64)
    pipeline.train_loader = pipeline.train_loader # subsample_loader(pipeline.train_loader, 256)
    pipeline.val_loader    = pipeline.val_loader # subsample_loader(pipeline.val_loader, 32)
    pipeline.test_loader  = pipeline.test_loader # subsample_loader(pipeline.test_loader, 32)
    print("train_loader size:", len(pipeline.train_loader.dataset))
    print("val_loader size:", len(pipeline.val_loader.dataset))
    print("test_loader size:", len(pipeline.test_loader.dataset))
    F = X.shape[-1]
    T_out = y.shape[1]

    # ---------- build model ----------
    encoder = TabularTransformerEncoder(
        input_dim=F,
        model_dim=256,
        num_layers=4,
        num_heads=8,
        max_timesteps=X.shape[1],
    )

    head = RegressionHead(
        input_dim=256,
        T_out=T_out,
    )

    # ---------- optimizer ----------
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # ---------- trainer ----------
    trainer = Trainer(
        model=TabularModel(encoder, head, autoreg=True, teacher_forcing_ratio=0.3),
        optimizer=optimizer,
        loss_fn=weighted_masked_huber, #weighted_masked_mae,
        device=device,
        cont_idx=cont_idx
    )

    # compute stats for continuous features
    trainer.set_cont_stats(X)
    trainer.set_y_stats(y)

    # ---------- train ----------
    trainer.fit(
        train_loader=pipeline.train_loader,
        val_loader=pipeline.val_loader,
        epochs=50,
        patience=20,
        min_delta=1e-4,
    )

    # ---------- test ----------
    test_loss, y_pred = trainer.evaluate(pipeline.test_loader)
    print(f"✅ Test Masked {trainer.loss_fn} Loss: {test_loss:.4f}")

    # ===== CORRECT METRICS: FUTURE ONLY =====

    # collect y_true from the SAME loader
    y_true = np.concatenate(
        [batch["y"] for batch in pipeline.test_loader],
        axis=0
    )  # (B_test, T_out)

    # model predicts ONLY future (t >= 2)
    y_true_fut = y_true[:, 2:]        # (B_test, T_out-2)
    y_pred_fut = y_pred               # (B_test, T_out-2)
    t=y_pred_fut.shape[1]
    y_true_fut = y_true_fut[:, :t]    # align shapes if needed
    # sanity check (MANDATORY)
    assert y_true_fut.shape == y_pred_fut.shape, (
        f"Shape mismatch: {y_true_fut.shape} vs {y_pred_fut.shape}"
    )

    # flatten ONCE
    y_true_flat = y_true_fut.reshape(-1)
    y_pred_flat = y_pred_fut.reshape(-1)

    # metrics
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
    r2 = r2_score(y_true_flat, y_pred_flat)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")



    # save predictions
    np.save("data/processed/test_predictions_future.npy", y_pred_fut)
    print("Test predictions saved to data/processed/test_predictions_future.npy")
    y_true_flat = y_true_fut.reshape(-1)
    y_pred_flat = y_pred_fut.reshape(-1)

    print(
        "True stats:",
        "min", np.min(y_true_flat),
        "mean", np.mean(y_true_flat),
        "max", np.max(y_true_flat),
    )
    print(
        "Pred stats:",
        "min", np.min(y_pred_flat),
        "mean", np.mean(y_pred_flat),
        "max", np.max(y_pred_flat),
    )


    print("True stats:", " y_true min ",np.min(y_true), "y_true mean", np.mean(y_true), "y_true max ", np.max(y_true))
    print("Pred stats:", " y_pred min ", np.min(y_pred), "y_pred mean", np.mean(y_pred), "y_pred max ", np.max(y_pred))

    trainer.visualize_batch(
        pipeline.test_loader,
        n_examples=5,
        plot=True, denorm_y=False
        )

    # plt.hist(y_true, bins=30, alpha=0.5, label="true")
    # plt.hist(y_pred, bins=30, alpha=0.5, label="pred")
    # plt.legend()
    # plt.title("WOMAC distribution")
    # plt.show()

    for t in [2, 4, 6]:
        print(f"MAE >= {t}:", trainer.conditional_mae(y_true_fut, y_pred_fut, t))


    # y_true MUST be (B_test, T_out) here
    # If you flattened it earlier, reconstruct it from the loader

    y_test = np.concatenate(
        [batch["y"] for batch in pipeline.test_loader],
        axis=0
    )  # shape: (B_test, T_out)

    y_train = np.concatenate(
        [batch["y"] for batch in pipeline.train_loader],
        axis=0
    )  # shape: (B_train, T_out)

    mean_y = y_train.mean(axis=0)  # (T_out,)

    B_test = y_test.shape[0]
    y_baseline = np.tile(mean_y[None, :], (B_test, 1))  # (B_test, T_out)

    # slice future only
    y_test_fut = y_test[:, 2:]
    y_baseline_fut = y_baseline[:, 2:]

    baseline_mae = mean_absolute_error(
        y_test_fut.reshape(-1),
        y_baseline_fut.reshape(-1)
    )

    baseline_rmse = np.sqrt(
        np.mean((y_test_fut - y_baseline_fut) ** 2)
    )

    print(f"📉 Baseline MAE : {baseline_mae:.4f}")
    print(f"📉 Baseline RMSE: {baseline_rmse:.4f}")



    print(
        "Target percentiles:",
        np.percentile(y_true_flat, [50, 80, 90, 95, 99]),
    )



if __name__ == "__main__":
    main()
