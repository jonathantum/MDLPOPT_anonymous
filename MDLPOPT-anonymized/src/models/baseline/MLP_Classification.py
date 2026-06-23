# src/models/mlp_classifier.py

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
# Utility: Normalize X inside the dataset ONCE
# ---------------------------------------------------------
def normalize_dataset(ds):
    X = ds.X
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    ds.X = (X - mean) / std
    return ds


# ---------------------------------------------------------
# Model: Two-branch MLP (left/right)
# ---------------------------------------------------------
class MLPClassifier(nn.Module):
    """
    Two-output MLP: predicts two independent multi-class labels (left + right)
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], output_dims=(10, 10), dropout=0.3):
        super().__init__()

        def make_branch():
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]
                prev = h
            layers.append(nn.Linear(prev, output_dims[0]))  # placeholder, overwritten below
            return layers

        # left
        left_layers = make_branch()
        left_layers[-1] = nn.Linear(hidden_dims[-1], output_dims[0])
        self.network_left = nn.Sequential(*left_layers)

        # right
        right_layers = make_branch()
        right_layers[-1] = nn.Linear(hidden_dims[-1], output_dims[1])
        self.network_right = nn.Sequential(*right_layers)

    def forward(self, x):
        return self.network_left(x), self.network_right(x)


# ---------------------------------------------------------
# Training + Evaluation
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader=None, lr=1e-3, epochs=100, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR scheduler: lower LR when val loss stagnates
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOPPING = 10

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        correct = [0, 0]
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y_left = y[:, 0].to(device).long()
            y_right = y[:, 1].to(device).long()

            optimizer.zero_grad()
            out_left, out_right = model(X)

            loss_left = criterion(out_left, y_left)
            loss_right = criterion(out_right, y_right)
            loss = loss_left + loss_right

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            correct[0] += (out_left.argmax(dim=1) == y_left).sum().item()
            correct[1] += (out_right.argmax(dim=1) == y_right).sum().item()
            total += X.size(0)

        train_loss = running_loss / total
        train_acc = [correct[0] / total, correct[1] / total]

        logging.info(
            f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | "
            f"Acc L: {train_acc[0]:.4f} | Acc R: {train_acc[1]:.4f}"
        )

        # validation
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, device)
            logging.info(
                f"  Val Loss: {val_loss:.4f} | "
                f"Val Acc L: {val_acc[0]:.4f} | Val Acc R: {val_acc[1]:.4f}"
            )

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_mlp_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING:
                logging.info("Early stopping triggered.")
                break

    logging.info("Loading best model checkpoint...")
    model.load_state_dict(torch.load("best_mlp_model.pth"))
    return model


def evaluate_model(model, loader, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    correct = [0, 0]
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y_left = y[:, 0].to(device).long()
            y_right = y[:, 1].to(device).long()

            out_left, out_right = model(X)
            loss = criterion(out_left, y_left) + criterion(out_right, y_right)
            running_loss += loss.item() * X.size(0)

            correct[0] += (out_left.argmax(dim=1) == y_left).sum().item()
            correct[1] += (out_right.argmax(dim=1) == y_right).sum().item()
            total += X.size(0)

    return running_loss / total, [correct[0] / total, correct[1] / total]


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
    output_dims = (
        len(torch.unique(train_dataset.y[:, 0])),
        len(torch.unique(train_dataset.y[:, 1])),
    )

    model = MLPClassifier(input_dim=input_dim, output_dims=output_dims)
    model = train_model(model, train_loader, val_loader, epochs=100, device='cpu')

    test_loss, test_acc = evaluate_model(model, test_loader)
    logging.info(f"Test Loss: {test_loss:.4f} | Test Acc left: {test_acc[0]:.4f} | Test Acc right: {test_acc[1]:.4f}")

