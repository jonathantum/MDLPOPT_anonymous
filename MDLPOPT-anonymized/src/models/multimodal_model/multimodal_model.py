import os
os.environ['LD_PRELOAD'] = ""
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
import signal
# ---------- Project Bootstrap ----------
logging.basicConfig(level=logging.INFO)
cont_idx = [0, 2, 3, 4, 11, 12, 14, 15, 17, 18, 20,  53, 56, 57, 58]
binary_idx = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 63, 64]
ordinal_idx = [1, 5, 6, 7, 8, 9, 10, 13, 16, 19, 21, 22, 23, 54, 55, 65, 66]
nominal_idx = [50, 51, 52] # RACE, SEX, COHORT
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
from models.modules.TabularYearEncoder import TabularYearEncoder
from models.modules.mri_encoder import MRIEncoder, SliceAttention
# Note: Using MRIEncoder as a base for X-ray as well, or you can import a specific XrayEncoder
from models.modules.MidFusionTransformer import MidFusionTransformer
from models.modules.classification_head import ClassificationHead
from training.losses import FocalLoss
from training.trainer import Trainer, evaluate_classification, create_weighted_sampler, rank_oai_features, plot_patient_attention
from utils.oai_ds_utils import OAIPipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


os.environ["WANDB_DIR"] = "/mnt/nfs/homedirs/anonymous/MDLPOPT/wandb"
# ---------- Full Multimodal Model ----------

class MultimodalTemporalModel(nn.Module):
    def __init__(self, tab_encoder, mri_dess_encoder, mri_iw_encoder, xray_encoder, fusion_module, head, model_dim):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "tabular": tab_encoder,
            "xray": xray_encoder,
            "mri_SAG_3D_DESS": mri_dess_encoder
            #"mri_COR_IW_TSE": mri_iw_encoder,
        })
        self.mri_attention = SliceAttention(model_dim=model_dim)
        self.fusion = fusion_module
        self.head = head
    def forward(self, **kwargs):
        all_tokens = []
        all_masks = []
        
        # Determine Batch Size
        first_input = next(iter(kwargs.values()))
        B = first_input.shape[0]
        
        for key, encoder in self.encoders.items():
            if key in kwargs:
                data = kwargs[key]
                mask = kwargs.get(f"{key}_mask")

                # 1. ENCODING
                # if data.dim() == 5:  # MRI [B, T, Slices, H, W]
                #     B_val, T, S, H, W = data.shape
                #     # Process MRI through Slice Attention
                #     # (Using your existing sub-batching logic for memory)
                #     mri_visits = []
                #     for t in range(T):
                #         vol = data[:, t].reshape(-1, 1, H, W).repeat(1, 3, 1, 1) # [B*S, 3, H, W]
                #         chunk_size = 16
                #         feats = []
                #         for i in range(0, vol.shape[0], chunk_size):
                #             feats.append(encoder(vol[i:i+chunk_size]))
                #         # Aggregate slices back to [B, 256]
                #         visit_feats = torch.cat(feats, dim=0).reshape(B, S, -1)
                #         mri_visits.append(self.mri_attention(visit_feats))
                #     tokens = torch.stack(mri_visits, dim=1) # [B, T, 256]
                if data.dim() == 5:  # MRI [B, T, S, H, W]
                    B_val, T, S, H, W = data.shape
                    mri_visits = []
                    for t in range(T):
                        # 1. Prepare 2.5D stacks
                        vol = data[:, t] # [B, S, H, W]
                        # Pad with replicate to keep 3 channels for first and last slices
                        padded = torch.nn.functional.pad(vol, (0, 0, 0, 0, 1, 1), mode='replicate')
                        
                        # Create [B, S, 3, H, W]
                        stack = torch.stack([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=2)
                        # Flatten to [B*S, 3, H, W]
                        vol_input = stack.reshape(-1, 3, H, W)
                        
                        # 2. Process through encoder (using your existing sub-batching)
                        chunk_size = 16
                        feats = []
                        for i in range(0, vol_input.shape[0], chunk_size):
                            feats.append(encoder(vol_input[i:i+chunk_size]))
                        
                        visit_feats = torch.cat(feats, dim=0).reshape(B, S, -1)
                        mri_visits.append(self.mri_attention(visit_feats))
                    tokens = torch.stack(mri_visits, dim=1) # [B, T, 256]
                else: 
                    tokens = encoder(data, mask) # [B, T, 256]

                # 2. MASKING
                if mask is not None:
                    if mask.dim() == 3: mask = ~mask.any(dim=-1)
                    mask = mask.bool()
                else:
                    mask = torch.zeros((B, tokens.shape[1]), device=tokens.device, dtype=torch.bool)

                # 3. MODALITY-SPECIFIC DELTA + NOISE GATE
                # delta = tokens[:, -1, :] - tokens[:, 0, :] 
                
                # # Apply Noise Gate to Tabular only to prevent false 'Increasing' labels
                # if key == "tabular":
                #     delta = torch.where(torch.abs(delta) < 0.05, torch.zeros_like(delta), delta)
                
                # delta_mask = mask[:, 0] | mask[:, -1]
                
                # # Append Delta
                # tokens = torch.cat([tokens, delta.unsqueeze(1)], dim=1)
                # mask = torch.cat([mask, delta_mask.unsqueeze(1)], dim=1)
                if tokens.shape[1] > 1:
                    delta = tokens[:, -1, :] - tokens[:, 0, :] 
                    
                    # Apply Noise Gate to Tabular
                    if key == "tabular":
                        delta = torch.where(torch.abs(delta) < 0.05, torch.zeros_like(delta), delta)
                    
                    delta_mask = mask[:, 0] | mask[:, -1]
                    
                    # Append Delta only if we actually have a temporal sequence
                    tokens = torch.cat([tokens, delta.unsqueeze(1)], dim=1)
                    mask = torch.cat([mask, delta_mask.unsqueeze(1)], dim=1)
                else:
                    # T = 1: Do nothing. The sequence stays length 1.
                    pass
                all_tokens.append(tokens)
                all_masks.append(mask)
        # 4. FUSION & HEAD
        combined_tokens = torch.cat(all_tokens, dim=1) 
        combined_masks = torch.cat(all_masks, dim=1)
        # fused = self.fusion(combined_tokens, combined_masks)
        # return self.head(fused)
        # Capture weights here
        fused_representation, attn_weights = self.fusion(combined_tokens, combined_masks)
        
        logits = self.head(fused_representation)
        
        # If in training mode, just return logits. If in eval, return weights too.
        if self.training:
            return logits
        else:
            return logits, attn_weights
# class GracefulInterrupt:
#     def __init__(self):
#         self.stop_now = False
#         signal.signal(signal.SIGINT, self.exit_gracefully)
#         signal.signal(signal.SIGTERM, self.exit_gracefully)

#     def exit_gracefully(self, signum, frame):
#         print("\n[!] Termination signal received. Will finish current epoch and jump to testing...")
#         self.stop_now = True
# ---------- Training Execution ----------


def single_run_main(oai_pipeline, device="cuda", input_years=3, target_years=7, batch_size=8, modalities=None, num_classes=4):
    # parameters
    lr = 1e-3
    mri_lr = 1e-4
    weight_decay = 0.1
    gamma = 1.5
    batch_size = batch_size
    epochs = 30
    patience = 10
    delta = 0.01
    task = "classification"
    label_aggregation = "womac_clsuters_4" #"KL_trajectory" #"womac_trajectory"
    monitor_metric =  "val/balanced_acc"
    num_layers_transformer = 2
    model_dim = 128
    tab_enc_dropout = 0.5 
    xray_enc_dropout = 0.5
    mri_enc_dropout = 0.5
    slice_attn_dropout = 0.2
    transformer_dropout = 0.3
    pred_head_dropout = 0.5
    top_n_features = 1

    # Initialize Encoders
    tab_enc = TabularYearEncoder(input_dim=top_n_features, model_dim=model_dim, dropout=tab_enc_dropout)
    dess_enc = MRIEncoder(model_dim=model_dim, dropout=mri_enc_dropout) 
    iw_enc = MRIEncoder(model_dim=model_dim, dropout=slice_attn_dropout)
    xray_enc = MRIEncoder(model_dim=model_dim, dropout=xray_enc_dropout) # Reusing MRI architecture for X-ray
    
    fusion = MidFusionTransformer(input_dim=model_dim, num_layers=num_layers_transformer, dropout=transformer_dropout)
    head = ClassificationHead(input_dim=model_dim, n_classes=num_classes, dropout=pred_head_dropout)
    
    model = MultimodalTemporalModel(tab_enc, dess_enc, iw_enc, xray_enc, fusion, head, model_dim=model_dim).to(device)
    # for param in model.encoders["mri_SAG_3D_DESS"].parameters():
    #     param.requires_grad = False

    mri_params = []
    for key in ["xray","mri_SAG_3D_DESS" ]:
        if key in model.encoders:
            mri_params.extend(list(model.encoders[key].parameters()))
    # Identify parameter IDs to avoid overlap
    mri_param_ids = set(map(id, mri_params))
    other_params = [p for p in model.parameters() if id(p) not in mri_param_ids]
    # --- Optimizer & Scheduler Block ---
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr},            # Fusion & Head
        {'params': mri_params, 'lr': mri_lr}           # MRI/X-ray slow fine-tuning
    ], weight_decay=weight_decay)

    temp_trainer = Trainer(model=model, device=device, input_years=input_years, 
                           target_years=target_years, task=task, 
                           label_aggregation=label_aggregation, optimizer=optimizer, main_loss_fn=None)
    # 1. Get class counts from your derived labels
    y_train_raw = torch.tensor(oai_pipeline.y[oai_pipeline.train_idx])
    y_train_mask = torch.tensor(oai_pipeline.y_mask[oai_pipeline.train_idx])
    train_labels, _ = temp_trainer.derive_labels(y_train_raw, y_train_mask)
    # 2. Compute Weights
    # Count samples per class (0, 1)
    unique, counts = torch.unique(train_labels[train_labels != -1], return_counts=True)
    print(f"DEBUG: Train Class Counts: {dict(zip(unique.tolist(), counts.tolist()))}")
    class_counts = torch.bincount(train_labels[train_labels != -1])
    alpha_weights = 1.0 / torch.pow(class_counts.float(), 0.35)
    alpha_weights = alpha_weights / alpha_weights.sum() # Normalize to sum to 1
    # if num_classes == 3:
    #     alpha_weights = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float32)
    # elif num_classes == 4:
    #     alpha_weights = torch.tensor([0.225, 0.225, 0.275, 0.275], dtype=torch.float32) # Manually set based on class distribution and importance
    # elif num_classes == 2:
    #     alpha_weights = torch.tensor([0.30, 0.70], dtype=torch.float32)
    print(f"Computed Focal Alpha Weights: {alpha_weights}")

    # 3. Initialize FocalLoss with these weights
    loss_fn = FocalLoss(gamma=gamma, alpha=alpha_weights.to(device), smoothing=0.1, reduction='mean')

    # alpha_weights = torch.tensor(alpha_weights).to(device).half()
    # loss_fn = nn.CrossEntropyLoss(weight=alpha_weights.to(device))

    # Sampler Logic

    
    
    if num_classes == 3:
        target_ratios = {
            0: 0.10,  # Low Stable
            1: 0.40,  # Worsening
            2: 0.50   # High Persistent
        }
    elif num_classes == 4:
        target_ratios = {
            0: 0.50,  # Low-Stable (Increase this from 0.40)
            1: 0.15,  # Improving
            2: 0.20,  # Worsening (Slightly lower weight to reduce False Positives)
            3: 0.15   # High-Persistent
        }
    elif num_classes == 2:
        target_ratios = {
            0: 0.40,  # Non-Progressor
            1: 0.60   # Progressor
        }
        print(f"Target Ratios for Weighted Sampling: {target_ratios}")
    sampler =  create_weighted_sampler(train_labels.numpy(), target_ratios=target_ratios)


    train_loader = DataLoader(oai_pipeline.train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True)
    # --- Main Scheduler Only ---
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-7
    )

    trainer = Trainer(model=model, optimizer=optimizer, main_loss_fn=loss_fn, 
                      device=device, task=task, label_aggregation=label_aggregation,
                      input_years=input_years, target_years=target_years, cont_idx=cont_idx, 
                      binary_idx=binary_idx, ordinal_idx=ordinal_idx, nominal_idx=nominal_idx,
                       main_scheduler=main_scheduler, top_n_features=top_n_features, num_classes=num_classes, modalities=modalities)


    wandb.log({


        "learning_rate": lr,
        "mri_learning_rate": mri_lr,
        "weight_decay": weight_decay,
        "focal_gamma": gamma,
        "epochs": epochs,
        "patience": patience,
        "delta": delta,
        "model_architecture": "ResNet18+Transformer",
        "modalities":  oai_pipeline.modalities,
        "task": task,
        "label_aggregation": label_aggregation,
        "loss_function": loss_fn.__class__.__name__,
        "smoothing": loss_fn.smoothing if isinstance(loss_fn, FocalLoss) else "N/A",
        "alpha_weights": alpha_weights.tolist(),
        "optimizer": optimizer.__class__.__name__,
        "Encoders": {
            "tabular": tab_enc.__class__.__name__,
            "xray": xray_enc.__class__.__name__,
            "mri_SAG_3D_DESS": dess_enc.__class__.__name__,
#            "mri_COR_IW_TSE": iw_enc.__class__.__name__,
        },
        "fusion_module": fusion.__class__.__name__,
        "classification_head": head.__class__.__name__,
        "monitor_metric": monitor_metric,
        "sampler": sampler.__class__.__name__ if sampler is not None else "None",
        "num_layers_transformer": num_layers_transformer,
        "model_dim": model_dim,
        "tab_enc_dropout": tab_enc_dropout,
        "xray_enc_dropout": xray_enc_dropout,
        "mri_enc_dropout": mri_enc_dropout,
        "transformer_dropout": transformer_dropout,
        "pred_head_dropout": pred_head_dropout,
                
    })
    # # Freeze the heavy lifting
    # for param in dess_enc.parameters():
    #     param.requires_grad = False

    # # Only the Attention layer needs to learn how to weight slices
    # for param in model.mri_attention.parameters():
    #     param.requires_grad = True
    #trainer.set_cont_stats(oai_pipeline.X[oai_pipeline.train_idx])
    trainer.set_stats(oai_pipeline.X[oai_pipeline.train_idx], y_train_raw, y_train_mask)
    #interrupter = GracefulInterrupt()

    # We need to modify the trainer.fit to check for interrupter.stop_now
    # If you can't edit Trainer.fit, you can do this:
    try:
        # Pass the interrupter to the trainer or check it in a loop
        trainer.fit(
            train_loader, 
            oai_pipeline.val_loader, 
            epochs=epochs, 
            patience=patience, 
            min_delta=delta, 
            monitor_metric=monitor_metric,
            #interrupter=interrupter,
            main_scheduler=main_scheduler
        )
    except KeyboardInterrupt:
        print("\n[!] Manual Ctrl+C detected. Proceeding to evaluation...")
    return trainer.evaluate(oai_pipeline.test_loader)

def main():
    print("Starting Multimodal Temporal Model Training...")
    print("Change comment: new baseline multimodal test to see if gpu can take it")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_target_no_surgery")
    root2 = Path("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/KL_TKR_target_low_and high_stable_KL_removed")
    root3 = Path("/mnt/nfs/homedirs/anonymous/MDLPOPT/data/processed/all_years/2_or_less_missing_target_visits/Womac_4_cluster_with_surgery_3")
    xray_base = Path("/mnt/nfs/homedirs/anonymous/NDA_processed/images/xray")
    print("Loading Dataset root", root3)
    # Load Main Tensors
    X, X_mask = np.load(root3 / "X.npy"), np.load(root3 / "X_mask.npy")
    y, y_mask = np.load(root3 / "y.npy").squeeze(), np.load(root3 / "y_mask.npy").squeeze()
    ids = np.load(root3 / "ids.npy", allow_pickle=True)
    
    # Load Image Paths
    xray_paths = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_paths.npy", allow_pickle=True)[:,:,0]
    xray_mask = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/tensors/xray_mask.npy")[:,:,0]
    
    dess_paths = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/images/mri/SAG_3D_DESS/full_vol/mri_paths.npy", allow_pickle=True)
    dess_mask = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/images/mri/SAG_3D_DESS/full_vol/mri_mask.npy")
    
    #cor_paths = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/images/mri/COR_IW_TSE/full_vol/mri_paths.npy", allow_pickle=True)
    #cor_mask = np.load("/mnt/nfs/homedirs/anonymous/NDA_processed/images/mri/COR_IW_TSE/full_vol/mri_mask.npy")
    print("Data Loaded.")
    # Setup
    print("X shape:", X.shape)
    print("X mask shape deeper:", X_mask.shape)
    pipeline = OAIPipeline(X, X_mask, y, y_mask, xray_paths, xray_mask, ids)
    pipeline.add_mri_sequence("mri_SAG_3D_DESS", dess_paths, dess_mask)
    #pipeline.add_mri_sequence("mri_COR_IW_TSE", cor_paths, cor_mask)
    input_years = 1
    target_years = 8
    num_classes = 4
    batch_size = 8
    modalities = ["tabular"]#,, "xray", "mri_SAG_3D_DESS" 
    stride = 4 # Process every xth slice to reduce memory load, can be tuned based on GPU capacity
    # Run
    pipeline.run(modalities=modalities, xray_base_dir=xray_base, batch_size=batch_size, num_workers=8, stride=stride)
    print("Modalities used:", modalities)

    print(f"Input Years: {input_years}, Target Years: {target_years}")
    wandb.init(
    project=os.path.basename(__file__),
    reinit=True,
    config={
        "modalities_used": modalities,
        "batch_size": batch_size,
        "model_architecture": "ResNet18+Transformer",
        "input_years": input_years,
        "target_years": target_years,
        "device": device,
        "num_classes": num_classes,
        "stride": stride
                }
    )
    loss, y_pred, y_probs, y_true = single_run_main(pipeline, device=device, input_years=input_years, target_years=target_years, batch_size=batch_size, modalities=modalities, num_classes=num_classes)

    #evaluate_classification(y_true, y_pred, y_probs, num_classes=num_classes)
    # wandb.log({
    #     "final/confusion_matrix": wandb.plot.confusion_matrix(
    #         probs=y_probs,
    #         y_true=y_true,
    #         preds=y_pred,
    #         class_names=["Non-Progressor", "Progressor"] 
    #     )
    # })
    # 1. Identify a "Successful Worsening" Patient from your test set
    # You need the actual weights from the model for a specific batch

    print("\n" + "="*80)
    print(f"{' FINAL RUN REPORT ':^80}")
    print("="*80)

    # 1. Print Configuration / Settings
    print("\n[ RUN CONFIGURATION ]")
    if wandb.run is not None:
        # wandb.config contains your epochs, patience, lr, etc.
        for k, v in wandb.run.config.items():
            print(f"  {k:<25}: {v}")

    # 2. Print Summary / Results
    print("\n[ RUN SUMMARY ]")
    if wandb.run is not None:
        # Using ._as_dict() prevents the KeyError: 'items'
        summary_dict = wandb.run.summary._as_dict()
        for k, v in summary_dict.items():
            if not k.startswith('_'):
                # Format floats for readability
                val = f"{v:.6f}" if isinstance(v, float) else v
                print(f"  {k:<25}: {val}")

    print("\n" + "="*80)
    wandb.finish()

if __name__ == "__main__":
    main()

