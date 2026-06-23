import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

class OAIDataset(Dataset):
    """
    Simple PyTorch Dataset wrapping a tensor of features (X) and optional labels (y)
    """
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.has_labels = y is not None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_labels:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class OAIDSPipeline:
    """
    A modular pipeline to convert DataFrames into tensors, Datasets, and DataLoaders.
    Patient-wise splitting is supported.
    Intermediate steps are saved to disk for later reuse.
    """
    def __init__(self, save_dir: str = "./data/processed/tensors", seed: int = 42):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.seed = seed
        self.splits = {}
        self.tensors = {}
        self.datasets = {}
        self.dataloaders = {}
     
    # ------------------------
    # Step 1: Patient-wise split
    # ------------------------
    def split_patients(self, df: pd.DataFrame, patient_id_col="ID", test_size=0.2, val_size=0.1):
        patients = df[patient_id_col].unique()
        trainval_patients, test_patients = train_test_split(
            patients, test_size=test_size, random_state=self.seed
        )
        train_patients, val_patients = train_test_split(
            trainval_patients, test_size=val_size / (1 - test_size), random_state=self.seed
        )

        self.splits = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients
        }

        # Save splits
        with open(os.path.join(self.save_dir, "patient_splits.pkl"), "wb") as f:
            pickle.dump(self.splits, f)

        return self.splits

    # ------------------------
    # Step 2: Build tensors
    # ------------------------
    def build_sequential_tensors(
    self,
    df: pd.DataFrame,
    feature_cols=["VXXWOMKPL"],   # list of columns measured per visit
    feature_visits=[1,3],             # how many visits/years to include
    label_col="trajectory_cluster_left",
    patient_id_col="ID",
    visit_col="Visit"
    ):
        """
        Build train/val/test tensors using only the first N visits of each patient.
        Each patient becomes one row with shape (n_features * n_visits).

        df must contain: ID, Visit, features, and labels (one label per patient).
        """

        # ---- 1) Restrict to first feature_visits visits ----
        df_feat = df[df[visit_col].isin(feature_visits)].copy()
        df_feat = df_feat.sort_values([patient_id_col, visit_col])
        # ---- 2) Pivot so each patient has fixed-length feature vector ----

        pivoted = (
            df_feat
            .pivot(index=patient_id_col, columns=visit_col, values=['VXX' + col for col in feature_cols if 'VXX' + col in df.columns])
            .sort_index()
        )

        # Flatten MultiIndex columns
        pivoted.columns = [re.sub('VXX','', f"V0{v}{feat}") for (v, feat) in pivoted.columns]

        # ---- 3) Attach labels (labels must be patient-level!) ----
        labels = df[[patient_id_col]+ label_col].drop_duplicates().set_index(patient_id_col)
        pivoted = pivoted.join(labels, how="inner")

        # ---- 4) For each split, build tensors ----
        self.tensors = {}

        for split_name, patient_ids in self.splits.items():
            # select only patients in this split
            df_split = pivoted.loc[pivoted.index.isin(patient_ids)]

            # X has only the flattened feature columns
            X_np = df_split[pivoted.columns.difference(label_col)].values

            # Ensure numeric type
            X_np = X_np.astype(np.float32)

            # Optionally, fill NaNs with 0 (or some other strategy)
            X_np = np.nan_to_num(X_np, nan=0.0)

            y_np = df_split[label_col].values
            y_np = y_np.astype(np.int64)  # for class labels

            # Convert to tensors
            X = torch.tensor(X_np, dtype=torch.float32)
            y = torch.tensor(y_np, dtype=torch.long)


            self.tensors[split_name] = {"X": X, "y": y}

            # Save
            split_path = os.path.join(self.save_dir, f"{split_name}_tensors.pt")
            torch.save(self.tensors[split_name], split_path)

        return self.tensors


    # ------------------------
    # Step 3: Build Datasets
    # ------------------------
    def build_datasets(self):
        for split_name, tensor_dict in self.tensors.items():
            dataset = OAIDataset(tensor_dict["X"], tensor_dict["y"])
            self.datasets[split_name] = dataset

            # Save dataset
            split_path = os.path.join(self.save_dir, f"{split_name}_dataset.pkl")
            with open(split_path, "wb") as f:
                pickle.dump(dataset, f)

        return self.datasets

    # ------------------------
    # Step 4: Build DataLoaders
    # ------------------------
    def build_dataloaders(self, batch_size=32, shuffle=True):
        for split_name, dataset in self.datasets.items():
            # Test/val usually not shuffled
            is_shuffle = shuffle if split_name == "train" else False
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle)
            self.dataloaders[split_name] = loader

        return self.dataloaders

    # ------------------------
    # Step 5: Full pipeline
    # ------------------------
    def run_pipeline(self, df, feature_cols=None, label_col=None, patient_id_col="ID",
                     test_size=0.2, val_size=0.1, batch_size=32):
        self.split_patients(df=df, patient_id_col=patient_id_col, test_size=test_size, val_size=val_size)
        self.build_sequential_tensors(
            df=df, feature_cols=feature_cols, label_col=label_col, patient_id_col=patient_id_col
        )
        self.build_datasets()
        self.build_dataloaders(batch_size=batch_size)
        print("Pipeline finished. All tensors, datasets, and dataloaders saved.")
        return self.dataloaders
