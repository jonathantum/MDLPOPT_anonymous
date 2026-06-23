# MDLPOPT — Multimodal Deep Learning for Pain Outcome Prediction

A research pipeline for predicting knee osteoarthritis pain trajectories using the [OAI dataset](https://nda.nih.gov/oai/), combining **tabular clinical data** and **X-ray images** in a multimodal deep learning framework.

---

## Project Overview

The goal is to classify patients into pain trajectory clusters (e.g. stable, improving, worsening) over ~10 clinical visits, using:
- **Tabular features** (clinical assessments, KL grades, demographics) — longitudinal `[N, T, F]` tensors
- **X-ray images** (bilateral PA knee, full limb views) — path tensors `[N, T, I]`
- **Target labels** — WOMAC pain trajectory clusters `[N, 1]`

---

## Repository Structure

```
MDLPOPT/
│
├── data/
│   ├── OAI_Dataset/                    # Raw OAI data (not versioned)
│   ├── processed/
│   │   └── all_years/
│   │       └── 2_or_less_missing_target_visits/
│   │           └── Womac_3_cluster_with_surgery/
│   │               ├── X.npy           # Input features [N, T, F]
│   │               ├── X_mask.npy      # Missingness mask [N, T, F]
│   │               ├── y.npy           # Cluster labels [N, 1]
│   │               ├── y_mask.npy      # Label mask [N, 1]
│   │               ├── ids.npy         # Knee IDs [N]
│   │               └── splits.pkl      # Train/val/test indices
│   └── utils/
│       ├── full_oai_df_utils.py        # OAIDataProcessor: raw → clean DataFrames
│       └── oai_xray_utils.py           # OAIImageTensorBuilder: xray path tensors
│
├── src/
│   ├── models/
│   │   ├── baseline/                   # Tabular-only and image-only baselines
│   │   ├── multimodal_model/
│   │   │   └── multimodal_model.py     # Main model: wires together all modules
│   │   └── modules/                    # Reusable building blocks
│   │       ├── tabular_encoder_MLP.py          # MLP encoder for tabular features
│   │       ├── xray_encoder.py                 # ResNet-based X-ray encoder
│   │       ├── mri_encoder.py                  # MRI encoder (future modality)
│   │       ├── TabularYearEncoder.py           # Encodes visit/time index
│   │       ├── FusionTransformer.py            # Late-fusion transformer
│   │       ├── MidFusionTransformer.py         # Mid-fusion transformer
│   │       ├── classification_head.py          # Standard classification head
│   │       └── cluster_classification_head.py  # Head for trajectory cluster output
│   ├── training/
│   │   ├── trainer.py                  # Training loop
│   │   └── losses.py                   # Loss functions (MSE, MAE, Huber, FocalLoss...)
│   ├── utils/
│   │   └── oai_ds_utils.py             # OAIDataset + OAIPipeline (splits, loaders)
│   └── clustering/
│       └── clustering_pain_trajectories.py  # DTW/k-means trajectory clustering
│
├── notebooks/
│   ├── exploration_clinical.ipynb      # EDA on clinical features
│   ├── new_process_tabular.ipynb       # Interactive tabular pipeline (cell-by-cell)
│   ├── build_clusters.ipynb            # FPCA / DTW clustering experiments
│   └── evaluate_models.ipynb           # Metrics and plots for trained models
│
├── new_process_tabular_data.py         # CLI: raw OAI -> tensors + splits
├── process_xray.py                     # CLI: builds xray_paths.npy / xray_mask.npy
└── run_gpu.sh                          # SLURM job script for GPU cluster
```

---

## Pipeline — Step by Step

### 1. Process X-ray Images
Builds `[N, T, I]` path and mask tensors from the raw image metadata.

```bash
python process_xray.py
# Output: NDA_processed/tensors/xray_paths.npy, xray_mask.npy
```

Internally uses `OAIImageTensorBuilder` (`data/utils/oai_xray_utils.py`), which:
- Filters metadata to IDs present in your tabular cohort
- Maps visit codes (`V00`-`V11`) to time indices 0-9
- Maps scan types to image-type indices (bilateral PA=0, full limb=1)

### 2. Process Tabular Data & Build Tensors
The main data pipeline. Runs `OAIDataProcessor` then `OAIPipeline`.

```bash
python new_process_tabular_data.py \
  --input-path data/OAI_Dataset \
  --input-path-new data/processed/all_years/2_or_less_missing_target_visits \
  --output-path-full data/processed/all_years/2_or_less_missing_target_visits/Womac_3_cluster_with_surgery
```

**What it does (in order):**

| Step | What happens |
|------|-------------|
| 1 | `OAIDataProcessor` loads/cleans clinical, KXR, outcomes, and enrollee tables |
| 2 | Removes patients with >2 missing WOMAC visits |
| 3 | Builds input tensor `X [N, T, F]` and WOMAC target tensor |
| 4 | Clusters pain trajectories -> assigns integer class label per patient |
| 5 | `OAIPipeline` does patient-level train/val/test split (no leakage) |
| 6 | Saves `X.npy`, `X_mask.npy`, `y.npy`, `y_mask.npy`, `ids.npy`, `splits.pkl` |

### 3. Train a Model

```bash
# Locally
python src/models/multimodal_model/multimodal_model.py

# On SLURM
sbatch run_gpu.sh
```

`run_gpu.sh` sets up the conda env, WandB dirs, and runs the script defined by `SCRIPT_TO_RUN` at the top of the file — that is the only variable you normally need to change.

---

## Key Classes

### `OAIDataProcessor` (`data/utils/full_oai_df_utils.py`)
Handles all raw -> clean DataFrame logic. Key methods:
- `create_clinical_dataframe()` / `create_kxr_dataframe()` / `create_outcomes_dataframe()`
- `clean_all_visits()` — fills static variables, handles missingness
- `remove_patients_with_x_missing_entries()` — cohort filtering
- `build_input_tensor()` / `build_womac_target_tensor()` — produces DataFrames for tensors
- `finalize_target_as_labels()` — collapses trajectory scores to cluster labels
- `save_tensors()` / `save_processed_data()`

### `OAIPipeline` (`src/utils/oai_ds_utils.py`)
Handles splits, datasets, and dataloaders:
- `split()` — patient-level 70/15/15 split
- `build_datasets()` — wraps `OAIDataset` for each split with appropriate augmentations
- `build_dataloaders()` — returns train/val/test `DataLoader`
- `get_balanced_train_loader()` — `WeightedRandomSampler` for class imbalance
- `save()` — persists all arrays and `splits.pkl`

### `OAIDataset` (`src/utils/oai_ds_utils.py`)
PyTorch `Dataset`. Returns a dict per sample:
```python
{
  "y":           Tensor[1],            # class label
  "y_mask":      Tensor[1, bool],      # always True for classification
  "id":          str,                  # knee ID e.g. "9000099L"
  # if modalities includes "xray":
  "xray_images": Tensor[T, 3, 256, 256],
  "xray_mask":   Tensor[T, bool],      # True = missing (mask out)
}
```
> **Note:** tabular features `X` / `X_mask` are passed directly into the model from stored tensors in the training loop, not through the batch dict.

### Model Modules (`src/models/modules/`)

| Module | Role |
|--------|------|
| `tabular_encoder_MLP.py` | Encodes `[N, T, F]` tabular features per timestep |
| `TabularYearEncoder.py` | Adds visit/time index as a learned embedding |
| `xray_encoder.py` | ResNet backbone for X-ray images |
| `mri_encoder.py` | Encoder for MRI (future modality) |
| `FusionTransformer.py` | Late fusion: combines modality tokens via transformer |
| `MidFusionTransformer.py` | Mid fusion: fuses intermediate representations |
| `classification_head.py` | Maps fused representation -> class logits |
| `cluster_classification_head.py` | Head specific to trajectory cluster output |

### Loss Functions (`src/training/losses.py`)

| Function | Use case |
|----------|----------|
| `masked_mse` | Regression with missingness |
| `masked_mae` / `masked_huber` | Robust regression |
| `weighted_masked_mae` | Upweights high-pain tail |
| `masked_quantile_loss` | Asymmetric errors |
| `FocalLoss` | Classification with class imbalance (with label smoothing) |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `exploration_clinical.ipynb` | First-pass EDA on raw clinical features |
| `new_process_tabular.ipynb` | Interactive run of the tabular pipeline (same as the CLI but cell-by-cell) |
| `build_clusters.ipynb` | DTW / FPCA trajectory clustering experiments to pick number of clusters |
| `evaluate_models.ipynb` | Load trained model checkpoints and compute metrics / plots |

---

## Data Notes

- Cohort: knees with 2 or fewer missing WOMAC visits across 10 OAI visits (V00-V11)
- Each knee is treated as an independent sample (`ID` = patient ID + side suffix `L`/`R`)
- Patient-level splits prevent data leakage between train/val/test
- TKR (total knee replacement) is handled as a terminal event (capped at 20.0 before clustering)
- X-rays are re-anchored at load time: only the filename is stored in the path tensor; the base directory is passed at runtime

---

## Environment

```bash
conda activate MDLPOPT

# Key dependencies
torch, torchvision
numpy, pandas, scikit-learn
Pillow
wandb
```

Set `THESIS_PROJECT_ROOT` if running outside the repo root:
```bash
export THESIS_PROJECT_ROOT=/path/to/MDLPOPT
```

---

## WandB

The SLURM script exports the API key and redirects WandB directories to `/mnt/nfs/...` to avoid permission issues on the cluster home dir. If running locally, just `wandb login` once.