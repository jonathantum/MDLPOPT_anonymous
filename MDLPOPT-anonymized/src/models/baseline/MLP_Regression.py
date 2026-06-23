# src/models/mlp_regressor.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
import logging
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "MDLPOPT", "data", "processed")

from MDLPOPT.data.utils.oai_ds_utils import OAIDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------
# Utility: Normalize X inside dataset
# ---------------------------------------------------------
def normalize_dataset(ds):
    X = ds.X
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    ds.X = (X - mean) / std
    return ds


# ---------------------------------------------------------
# Model: Simple MLP Regressor
# ---------------------------------------------------------
class MLPRegressor(nn.Module):
    """Standard feed-forward MLP that predicts 1 continuous WOMAC score."""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h

        layers.append(nn.Linear(prev, 1))  # 1 regression output
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # [batch]


# ---------------------------------------------------------
# Training + Evaluation
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader=None, lr=1e-3, epochs=100, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOPPING = 10

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.float().to(device)  # WOMAC target is continuous

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            total += X.size(0)

        train_loss = running_loss / total
        logging.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}")

        # validation
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device)
            logging.info(f"  Val Loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_mlp_regressor.pth")
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING:
                logging.info("Early stopping triggered.")
                break

    logging.info("Loading best model checkpoint...")
    model.load_state_dict(torch.load("best_mlp_regressor.pth"))
    return model


def evaluate_model(model, loader, device='cpu'):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.float().to(device)
            pred = model(X)

            loss = criterion(pred, y)
            total_loss += loss.item() * X.size(0)
            total += X.size(0)

    return total_loss / total


# ---------------------------------------------------------
# Load datasets + run
# ---------------------------------------------------------
if __name__ == "__main__":
    save_dir = "MDLPOPT/data/processed"

    def load_dataset(name):
        path = os.path.join(DATA_DIR, f"{name}_dataset.pkl")
        print("Loading:", path)
        with open(path, "rb") as f:
            return pickle.load(f)

    # load
    train_dataset = normalize_dataset(load_dataset("train"))
    val_dataset = normalize_dataset(load_dataset("val"))
    test_dataset = normalize_dataset(load_dataset("test"))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = train_dataset.X.shape[1]

    model = MLPRegressor(input_dim=input_dim)
    model = train_model(model, train_loader, val_loader, epochs=100, device='cpu')

    test_loss = evaluate_model(model, test_loader)
    logging.info(f"Test MSE Loss: {test_loss:.4f}")
