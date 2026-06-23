import sys
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score  
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torchvision
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import  WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
from datetime import datetime
import seaborn as sns
import matplotlib
import os
matplotlib.use('Agg')

class Trainer:
    def __init__(self, model, optimizer, main_loss_fn, aux_loss_fn=None, device=None, main_scheduler=None, class_weights=None, cont_idx=None, binary_idx=None, ordinal_idx=None, nominal_idx=None,
                 task="regression", classification_threshold=1.0,
                 input_years=3, target_years=1,
                 label_aggregation="mean", top_n_features=20, num_classes=4, modalities=["tabular"]):
        self.model = model.to(device)
        self.modalities= modalities
        self.optimizer = optimizer
        self.main_scheduler = main_scheduler
        self.main_loss_fn = main_loss_fn      # Your personal Focal Loss / CrossEntropy / MSE
        self.aux_loss_fn = aux_loss_fn        
        self.num_classes = num_classes
        self.device = device
        self.class_weights = class_weights
        self.cont_idx = cont_idx
        self.binary_idx = binary_idx
        self.ordinal_idx = ordinal_idx
        self.nominal_idx = nominal_idx
        self.feature_map = {
            0:"BMI", 1:"BMICAT", 2:"WEIGHT", 3:"AGE", 4:"PASE", 5:"PASE1HR", 6:"PASE2HR", 7:"PASE3HR", 
            8:"PASE4HR", 9:"PASE5HR", 10:"PASE6HR", 11:"CESD", 12:"HSMSS", 13:"SMOKER", 14:"SMKAGE", 
            15:"SMKAVE", 16:"SMKNOW", 17:"SMKAMT", 18:"SMKSTOP", 19:"DRNKAMT", 20:"COMORB", 21:"MARITST", 
            22:"CHNFQCV", 23:"GLCFQCV", 24:"KPMED", 25:"KPMEDCV", 26:"NSAIDS", 27:"NSAIDRX", 28:"COXIBS", 
            29:"NARCOT", 30:"TYLEN", 31:"CHON", 32:"GLUC", 33:"MSM", 34:"SAME", 35:"PNMEDT", 36:"DOXYCYC", 
            37:"KNINJ", 38:"HRTAT", 39:"BYPLEG", 40:"STROKE", 41:"ASTHMA", 42:"LUNG", 43:"ULCER", 44:"DIAB", 
            45:"KIDFXN", 46:"RA", 47:"POLYRH", 48:"LIVDAM", 49:"CANCER", 50:"RACE", 51:"SEX", 52:"COHORT", 
            53:"HEIGHT", 54:"EDCV", 55:"INCOME", 56:"KOOSKPL/R", 57:"KOOSYML/R", 58:"KOOSFSL/R", 59:"WOMKPL/R", 
            60:"WOMSTFL/R", 61:"WOMADLL/R", 62:"LKALNMT/R", 63:"INJL12/R12", 64:"ELKVSAF/R", 65:"ELKTLPR/R", 
            66:"LXRKL/R"
        }
        self.top_n_features = top_n_features
        self.active_indices = None # This will store the filtered indices

        self.task = task
        self.classification_threshold = classification_threshold
        self.input_years = input_years
        self.target_years = target_years
        self.label_aggregation = label_aggregation
        self.scaler = torch.amp.GradScaler()

        self.mean = None
        self.std = None

 
    def set_stats(self, X_trainds, y_raw_trainds, y_mask_trainds):
        """
        X_trainds: (N, 10, 67)
        """
        # 1. DERIVE LABELS FOR FEATURE SELECTION
        y_labels, _ = self.derive_labels(
            torch.tensor(y_raw_trainds), 
            torch.tensor(y_mask_trainds)
        )
        y_labels_np = y_labels.numpy()
        valid_y_mask = y_labels_np != -1

        # 2. FEATURE SELECTION
        if self.top_n_features is not None:
            print(f"🔍 Selecting top {self.top_n_features} features via Random Forest...")
            X_rf = X_trainds[valid_y_mask]
            y_rf = y_labels_np[valid_y_mask]
            
            # Flatten time for RF
            X_flat = np.nanmean(X_rf, axis=1)
            X_flat = np.nan_to_num(X_flat, nan=0.0)
            
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            rf.fit(X_flat, y_rf)
            
            importances = rf.feature_importances_
            
            # Get ALL indices sorted by importance
            all_sorted_indices = np.argsort(importances)[::-1]
            exclude_kl=False
            # --- OVERALL SWITCH OPTION ---
            if exclude_kl== True: # Default to True if not set
                # Keep all indices except 66
                filtered_indices = [i for i in all_sorted_indices if i != 66]
                print("🚫 Excluding 'LXRKL/R' (KL Grade) from feature selection.")
            else:
                filtered_indices = all_sorted_indices

            # Now take exactly top_n from the filtered list (fills the 30th slot automatically)
            top_idx = filtered_indices[:self.top_n_features]
            
            # for i in top_idx:
            #     print(f"Feature: {self.feature_map[i]} | Importance: {importances[i]:.2f}")
                
            self.active_indices = list(top_idx) 
            self.active_indices.sort() 
            
            print(f"✅ Active Features: {[self.feature_map[i] for i in self.active_indices]}")
        else:
            # Use all features (67 or 66 depending on switch)
            if getattr(self, 'exclude_kl', True):
                self.active_indices = [i for i in range(X_trainds.shape[-1]) if i != 66]
            else:
                self.active_indices = np.arange(X_trainds.shape[-1])

        # 3. CALCULATE STATS FOR ACTIVE FEATURES ONLY
        num_active = len(self.active_indices)
        self.mean = np.zeros(num_active)
        self.std = np.ones(num_active)
        
        to_scale_global = set(self.cont_idx + self.ordinal_idx)
        
        for local_idx, global_idx in enumerate(self.active_indices):
            if global_idx in to_scale_global:
                vals = X_trainds[:, :, global_idx].flatten()
                valid_vals = vals[~np.isnan(vals)]
                
                if len(valid_vals) > 1:
                    self.mean[local_idx] = np.mean(valid_vals)
                    self.std[local_idx] = np.std(valid_vals) + 1e-6

        print(f"✅ Stats Set for {num_active} features.")

    def _apply_normalization(self, x_tabular):
        """x_tabular shape: (B, T, len(active_indices))"""
        if self.mean is None: return x_tabular
            
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean, dtype=torch.float32, device=self.device)
            self.std = torch.tensor(self.std, dtype=torch.float32, device=self.device)
            
        return (x_tabular - self.mean) / self.std

    def derive_labels(self, y, y_mask=None):
        """
        Standardized label derivation for all tasks.
        Flexible to handle pre-clustered labels [B] or temporal scores [B, T].
        """
        device = y.device
        
        # --- FLEXIBILITY CHECK ---
        # If y is 1D [B] or 2D [B, 1], it's already a label tensor from clustering
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            labels = y.squeeze().long()
            # If pre-computed, we assume all samples are valid unless y is -1
            label_mask = (labels != -1)
            return labels, label_mask
        # -------------------------

        # Traditional temporal logic for [B, T]
        B, T = y.shape
        mask = y_mask.bool() if y_mask is not None else torch.ones_like(y, dtype=torch.bool, device=device)
        if self.label_aggregation == "mean":
            baseline_vals = y[:, :self.input_years]
            baseline_mask = y_mask[:, :self.input_years]
            
            future_vals = y[:, self.input_years:self.input_years+self.target_years]
            future_mask = y_mask[:, self.input_years:self.input_years+self.target_years]

        elif self.label_aggregation == "year_to_year":
            baseline_vals = y[:, self.input_years-1:self.input_years]
            baseline_mask = y_mask[:, self.input_years-1:self.input_years]

            future_vals = y[:, self.input_years:self.input_years+1]
            future_mask = y_mask[:, self.input_years:self.input_years+1]

        elif self.label_aggregation == "baseline_to_last":
            baseline_vals = y[:, :self.input_years]
            baseline_mask = y_mask[:, :self.input_years]

            future_vals = y[:, self.input_years+self.target_years-1:self.input_years+self.target_years]
            future_mask = y_mask[:, self.input_years+self.target_years-1:self.input_years+self.target_years]

        elif self.label_aggregation == "last_year_to_whole":
            baseline_vals = y[:, self.input_years-1:self.input_years]
            baseline_mask = y_mask[:, self.input_years-1:self.input_years]

            future_vals = y[:, self.input_years:self.input_years+self.target_years]
            future_mask = y_mask[:, self.input_years:self.input_years+self.target_years]
        elif self.label_aggregation == "womac_trajectory":
                        # 1. Calculate Baseline Level (Intercept) - Years 0, 1, 2
            baseline_mask = y_mask[:, :self.input_years]
            baseline_vals = y[:, :self.input_years]

            # 2. Calculate Future Trend (Slope/Change) 
            future_mask = y_mask[:, self.input_years:]
            future_vals = y[:, self.input_years:]
        elif self.label_aggregation == "KL_trajectory":
            # 1. Calculate Baseline Level (Intercept) - Years 0, 1, 2
            baseline_mask = y_mask[:, :self.input_years]
            baseline_vals = y[:, :self.input_years]

            # 2. Calculate Future Trend (Slope/Change) 
            future_mask = y_mask[:, self.input_years:]
            future_vals = y[:, self.input_years:]

        else:
            raise ValueError(f"Unknown aggregation: {self.label_aggregation}")
        baseline = (baseline_vals * baseline_mask).sum(dim=1) / baseline_mask.sum(dim=1).clamp(min=1)
        future = (future_vals * future_mask).sum(dim=1) / future_mask.sum(dim=1).clamp(min=1)

        label_mask = (baseline_mask.sum(dim=1) > 0) & (future_mask.sum(dim=1) > 0)
        delta = future - baseline
        # 2. Map to Classes
        if self.task == "classification" and self.label_aggregation == "womac_trajectory":
            # 3-Class logic (Persistent High, Progressor, Stable Low)
            high_baseline_threshold = 8.0 
            progression_threshold = 1.0 # Significant worsening
            improving_threshold = -5.0 # Significant improvement

            labels = torch.zeros(B, dtype=torch.long, device=device) #all low stable by default
            labels[(baseline < high_baseline_threshold) & (delta >= progression_threshold)] = 1 # Progressor
            labels[(baseline >= high_baseline_threshold)] =2 # & (delta >= improving_threshold)] = 2 # High-Persistent
            #labels[(baseline < high_baseline_threshold) & (delta < improving_threshold)] = 3 # Improving
        elif self.task == "classification" and self.label_aggregation == "womac_clsusters_3":


            labels = torch.zeros(B, dtype=torch.long, device=device) #all low stable by default
            labels[y==1] = 1 # Progressor
            labels[y==2] = 2 # High-Persistent
        elif self.task == "classification" and self.label_aggregation == "womac_clsusters_4":
            labels = torch.zeros(B, dtype=torch.long, device=device) #all low stable by default
            labels[y==1] = 1 # Improving
            labels[y==2] = 2 # worsening
            labels[y==3] = 3 # High-Persistent
        elif self.task == "classification" and self.label_aggregation == "KL_trajectory":
            labels = torch.zeros(B, dtype=torch.long, device=device)
            
            # # 1. Progression: Any increase of >= 1 full grade.
            # # We check if delta >= 1.0. To be robust against float averages, 
            # # some researchers use 0.7 or 0.8 as a threshold for "clinically significant" change.
            # is_progressing = (delta >= 1.0) | ((baseline < 2.0) & (future >= 2.0))
            
            # # 2. High-Persistent: Starts at 3 or 4 and stays high (delta is small/negative)
            # # We define "Starts High" as baseline >= 2.5 (mapping to KL 3 or 4)
            # is_high_stable = (baseline >= 2) & (delta < 1.0)
            
            # # 3. Low-Stable: Everything else. 
            # # Since labels is initialized to zeros (Class 0), we only need to assign 1 and 2.
            
            # labels[is_progressing] = 1
            # labels[is_high_stable] = 2

            #is_progressing = (delta >= 1.0) | ((baseline < 2.0) & (future >= 2.0))
            is_progressing = (delta >= 1) & (baseline>=1) | (delta > 1)
            labels[is_progressing] = 1

        else:
            # Standard Binary logic (using your threshold)
            labels = torch.zeros_like(delta, dtype=torch.long, device=device)
            labels[delta > self.classification_threshold] = 1

        labels[~label_mask] = -1
        return labels, label_mask
    
    def step(self, batch, train=True):
        model_inputs = {}
        # --- 1. Tabular Logic ---
        if "tabular" in batch:
            X = batch["tabular"].to(self.device)
            # SLICE TO TOP FEATURES
            if self.active_indices is not None:
                indices = list(self.active_indices) 
                X = X[:, :, indices]
            # NORMALIZE 
            X = self._apply_normalization(X)            
            
            model_inputs["tabular"] = X[:, :self.input_years, :]
            
            if "tabular_mask" in batch:
                m = batch["tabular_mask"].to(self.device)
                model_inputs["tabular_mask"] = m[:, :self.input_years, self.active_indices]

        # --- 2. X-ray Logic ---
        if "xray_images" in batch:
            model_inputs["xray"] = batch["xray_images"].to(self.device)[:, :self.input_years, ...]
            
            if "xray_mask" in batch:
                model_inputs["xray_mask"] = batch["xray_mask"].to(self.device)[:, :self.input_years]

        # --- 3. MRI Logic ---
        mri_keys = [k for k in batch.keys() if k.startswith("mri_") and not k.endswith("_mask")]
        for m_key in mri_keys:
            model_inputs[m_key] = batch[m_key].to(self.device)[:, :self.input_years, ...]
            mask_key = f"{m_key}_mask"
            if mask_key in batch:
                model_inputs[mask_key] = batch[mask_key].to(self.device)[:, :self.input_years]

        # --- 4. Forward Pass ---
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(**model_inputs)
            
            # --- FIX: UNPACK TUPLE FOR EVAL MODE ---
            # If output is (logits, attn_weights), we take the logits as y_hat
            if isinstance(output, tuple):
                y_hat, attn_weights = output
            else:
                y_hat = output
                attn_weights = None

        # 2. Extract Targets
        y = batch["y"].to(self.device)
        y_mask = batch.get("y_mask", None)
        if y_mask is not None: y_mask = y_mask.to(self.device)

        # 3. Task Dispatcher 
        # y_hat is now guaranteed to be the tensor (logits)
        if self.task == "regression":
            loss, preds, targets = self._compute_regression_loss(y_hat, y, y_mask)
        elif self.task == "classification":
            loss, preds, targets = self._compute_classification_loss(y_hat, y, y_mask)
        elif "aux" in self.task:
            loss, preds, targets = self._compute_hybrid_loss(y_hat, y, y_mask)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        # 4. Backprop
        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # You can optionally return attn_weights here if you want to log them to WandB
        return loss.item(), preds, targets, attn_weights
    
    def _compute_classification_loss(self, logits, y, y_mask):
        labels, valid_mask = self.derive_labels(y, y_mask)
        # Using the main_loss_fn passed during init
        loss = self.main_loss_fn(logits[valid_mask], labels[valid_mask])
        return loss, logits[valid_mask].detach(), labels[valid_mask].detach()

    def _compute_regression_loss(self, pred, y, y_mask):
        _, valid_mask = self.derive_labels(y, y_mask)
        # Target is the average WOMAC of the future window
        target = y[:, self.input_years : self.input_years + self.target_years].mean(dim=1)
        # Using main_loss_fn (e.g., MSE)
        loss = self.main_loss_fn(pred[valid_mask].squeeze(), target[valid_mask])
        return loss, pred[valid_mask].detach(), target[valid_mask].detach()

    def _compute_hybrid_loss(self, y_hat, y, y_mask):
        """
        Calculates loss for models returning a tuple (main_head, aux_head).
        """
        main_out, aux_out = y_hat
        
        # Compute Main Loss (Reuses standard logic)
        if self.task.startswith("reg"):
            main_loss, preds, targets = self._compute_regression_loss(main_out, y, y_mask)
        else:
            main_loss, preds, targets = self._compute_classification_loss(main_out, y, y_mask)

        # Compute Auxiliary Loss (Input state reconstruction)
        target_aux = y[:, :self.input_years]
        mask_aux = y_mask[:, :self.input_years].bool()
        
        if self.aux_loss_fn is not None:
            # Predicts raw values for each input year
            loss_aux = self.aux_loss_fn(aux_out[mask_aux], target_aux[mask_aux])
        else:
            # Fallback to functional MSE if no aux_loss_fn provided
            loss_aux = torch.nn.functional.mse_loss(aux_out[mask_aux], target_aux[mask_aux])

        total_loss = (main_loss + loss_aux)*0.5
        return total_loss, preds, targets
    
    def run_epoch(self, loader, train=True, num_classes=4):
        self.model.train() if train else self.model.eval()
        losses, all_probs, all_preds, all_targets, attn_weights = [], [], [], [], []

        with torch.set_grad_enabled(train):
            for batch_idx, batch in enumerate(loader):
                # 1. Run the actual training/val step
                # This is where derive_labels() happens internally
                loss_val, output, targets, attn_weights  = self.step(batch, train=train)


                if targets.numel() > 0:
                    losses.append(loss_val)
                    if self.task == "classification" or "aux" in self.task:
                        # 1. Get Soft Probs (for AUC)
                        probs = torch.softmax(output, dim=1)
                        # 2. Get Hard Preds (for Accuracy)
                        preds = torch.argmax(probs, dim=1)
                        
                        all_probs.append(probs.detach().cpu().numpy())
                        all_preds.append(preds.detach().cpu().numpy())
                    else:
                        all_preds.append(output.squeeze().detach().cpu().numpy())
                    
                    all_targets.append(targets.detach().cpu().numpy())
        avg_loss = np.mean(losses) if losses else 0
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        
        if self.task == "classification" or "aux" in self.task:
            y_prob = np.concatenate(all_probs)
            acc = accuracy_score(y_true, y_pred)
            bacc = balanced_accuracy_score(y_true, y_pred)
            # Per-class AUC storage
            per_class_auc = {}
            unique_present = np.unique(y_true)
            
            for i in range(self.num_classes): # For classes 0, 1, 2
                if i in unique_present and len(unique_present) > 1:
                    # Create binary targets for this specific class (One-vs-Rest)
                    y_true_binary = (y_true == i).astype(int)
                    # Use the i-th column of probabilities
                    per_class_auc[f"auc_class_{i}"] = roc_auc_score(y_true_binary, y_prob[:, i])
                else:
                    per_class_auc[f"auc_class_{i}"] = 0.5

            return avg_loss, acc, bacc, per_class_auc
        else:
            mse = np.mean((y_true - y_pred)**2)
            mae = np.mean(np.abs(y_true - y_pred))
            return avg_loss, mse, mae

    def fit(self, train_loader, val_loader=None, epochs=10, patience=10, min_delta=0.0, main_scheduler=None, monitor_metric="val/loss", interrupter=None):
        """
        Standard training loop with timestamps and elapsed time tracking.
        """
        # 1. Initialize monitoring and timing
        max_metrics = ["acc", "score", "f1", "auc", "precision", "recall"]
        should_maximize = any(m in monitor_metric.lower() for m in max_metrics)

        if should_maximize:
            is_better = lambda current, best: current > (best + min_delta)
            best_value = float("-inf")
        else:
            is_better = lambda current, best: current < (best - min_delta)
            best_value = float("inf")

        #self.interrupter = interrupter
        best_state = None
        epochs_no_improve = 0
        best_epoch = 0
        
        # Track the start of the whole process
        start_training_time = time.time()

        print(f"🚀 Starting training at {datetime.now().strftime('%H:%M:%S')}. Monitoring: {monitor_metric}")

        try:
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time() # Start time for this specific epoch
                
                # if hasattr(self, 'interrupter') and self.interrupter.stop_now:
                #     print("🛑 Stop signal received. Exiting training loop...")
                #     break

                # --- Training Phase ---
                t_loss, t_acc, t_bacc, t_aucs = self.run_epoch(train_loader, train=True)
                
                # Calculate elapsed time
                total_elapsed = time.time() - start_training_time
                m, s = divmod(int(total_elapsed), 60)
                h, m = divmod(m, 60)
                time_str = f"{h:02d}:{m:02d}:{s:02d}"
                curr_time = datetime.now().strftime('%H:%M:%S')

                log_dict = {
                    "epoch": epoch,
                    "train/loss": t_loss,
                    "train/acc": t_acc,
                    "train/balanced_acc": t_bacc,
                       
                    # "train/auc_low_stable": t_aucs["auc_class_0"],
                    # "train/auc_improving": t_aucs["auc_class_1"],
                    # "train/auc_worsening": t_aucs["auc_class_2"],
                    # "train/auc_high_persistent": t_aucs["auc_class_3"]
                }
                if self.num_classes == 2:
                    log_dict["train/auc_class_0"] = t_aucs["auc_class_0"]
                    log_dict["train/auc_class_1"] = t_aucs["auc_class_1"]
                elif self.num_classes == 3:
                    log_dict["train/auc_class_0"] = t_aucs["auc_class_0"]
                    log_dict["train/auc_class_1"] = t_aucs["auc_class_1"]
                    log_dict["train/auc_class_2"] = t_aucs["auc_class_2"]
                elif self.num_classes == 4:
                    log_dict["train/auc_class_0"] = t_aucs["auc_class_0"]
                    log_dict["train/auc_class_1"] = t_aucs["auc_class_1"]
                    log_dict["train/auc_class_2"] = t_aucs["auc_class_2"]
                    log_dict["train/auc_class_3"] = t_aucs["auc_class_3"]
                t_aucs_str = ", ".join([f"{k}: {v:.2f}" for k, v in t_aucs.items()])
                # Print with timestamp [HH:MM:SS] and elapsed time (00:00:00)
                print(f" Epoch {epoch}/{epochs} ({time_str}) | Train Loss: {t_loss:.2f} | Train Acc: {t_acc:.2f} | B-Acc: {t_bacc:.2f} | Train AUCs: {{{t_aucs_str}}} [{curr_time}]")

                # --- Validation ---
                if val_loader:
                    v_loss, v_acc, v_bacc, v_aucs = self.run_epoch(val_loader, train=False)
                    log_dict.update({
                        "val/loss": v_loss,
                        "val/balanced_acc": v_bacc,
                        # "val/auc_low_stable": v_aucs["auc_class_0"],
                        # "val/auc_improving": v_aucs["auc_class_1"],
                        # "val/auc_worsening": v_aucs["auc_class_2"],
                        # "val/auc_high_persistent": v_aucs["auc_class_3"],
                    })
                    
                    if self.num_classes == 2:
                        log_dict["val/auc_class_0"] = v_aucs["auc_class_0"]
                        log_dict["val/auc_class_1"] = v_aucs["auc_class_1"]
                    elif self.num_classes == 3:
                        log_dict["val/auc_class_0"] = v_aucs["auc_class_0"]
                        log_dict["val/auc_class_1"] = v_aucs["auc_class_1"]
                        log_dict["val/auc_class_2"] = v_aucs["auc_class_2"]
                    elif self.num_classes == 4:
                        log_dict["val/auc_class_0"] = v_aucs["auc_class_0"]
                        log_dict["val/auc_class_1"] = v_aucs["auc_class_1"]
                        log_dict["val/auc_class_2"] = v_aucs["auc_class_2"]
                        log_dict["val/auc_class_3"] = v_aucs["auc_class_3"]
                    v_aucs_str = ", ".join([f"{k}: {v:.2f}" for k, v in v_aucs.items()])

                    print(f"                | Val Loss: {v_loss:.2f} | Val Acc: {v_acc:.2f} | Val B_Acc: {v_bacc:.2f} | Val AUCs: {{{v_aucs_str}}}")
                    # --- Scheduler & Early Stopping Logic ---
                    current_val = log_dict[monitor_metric]

                    # --- Simplified Scheduler Logic ---
                    if main_scheduler is not None:
                        old_lrs = [group['lr'] for group in self.optimizer.param_groups]
                        
                        # Step based on validation performance
                        main_scheduler.step(current_val)
                        
                        new_lrs = [group['lr'] for group in self.optimizer.param_groups]
                        
                        # Log and Print changes
                        for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                            if new_lr < old_lr:
                                print(f"📉 [Main Scheduler] Group {i}: Reducing LR to {new_lr:.2e}")
                        
                        # Update log_dict with current learning rates
                        log_dict.update({f"lr/group_{i}": val for i, val in enumerate(new_lrs)})
                    if is_better(current_val, best_value):
                        best_value = current_val
                        best_epoch = epoch
                        best_state = {k: v.cpu().clone().detach() for k, v in self.model.state_dict().items()}
                        epochs_no_improve = 0
                        print(f"             ✨ New best {monitor_metric}!")
                    else:
                        epochs_no_improve += 1

                    # --- Early Stopping Trigger ---
                    if epochs_no_improve >= patience:
                        print(f"🛑 Early stopping. No improvement for {patience} epochs.")
                        break

                if wandb.run is not None:
                    wandb.log(log_dict)

        except KeyboardInterrupt:
            print("\n\n⚠️ Manual interrupt detected! Cleaning up...")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"✅ Best weights restored from epoch {best_epoch} (Best {monitor_metric}: {best_value:.2f}).")
    
    def _handle_interrupt(self, signum, frame):
        print("\n[!] Manual interrupt detected. Saving best state and exiting loop...")
        self.stop_training = True
    
    def evaluate(self, loader):
        self.model.eval()
        losses, all_preds, all_probs, all_targets = [], [], [], []
        
        # Storage for samples
        correct_samples_attn = {} 
        best_effort_attn = {} # Fallback for classes with 0 correct predictions
        highest_probs = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}
        
        target_class_ids = [0, 1, 2, 3]
        class_names = {0: "Low-Stable", 1: "Improving", 2: "Worsening", 3: "High-Persistent"}

        with torch.no_grad():
            for batch in loader:
                loss_val, output, targets, attn_weights = self.step(batch, train=False)
                
                if targets.numel() > 0:
                    losses.append(loss_val)
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                    for cls_id in target_class_ids:
                        # 1. Capture first CORRECT prediction
                        if cls_id not in correct_samples_attn:
                            correct_mask = (preds == targets) & (targets == cls_id)
                            indices = correct_mask.nonzero(as_tuple=True)[0]
                            if len(indices) > 0:
                                correct_samples_attn[cls_id] = attn_weights[indices[0]].cpu().detach()

                        # 2. Fallback: Capture sample with highest confidence for this class (even if wrong)
                        # This ensures you get an "Improving" plot even if recall is 0
                        cls_probs = probs[:, cls_id]
                        max_prob, max_idx = torch.max(cls_probs, dim=0)
                        if max_prob > highest_probs[cls_id]:
                            highest_probs[cls_id] = max_prob.item()
                            best_effort_attn[cls_id] = attn_weights[max_idx].cpu().detach()

        # --- Plotting Section with Subfolders ---
        import os
        from datetime import datetime
        
        # Create a unique subfolder for this specific run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_plot_dir = "/mnt/nfs/homedirs/anonymous/MDLPOPT/src/training/attention plots"
        run_subfolder = os.path.join(base_plot_dir, f"run_{timestamp}")
        os.makedirs(run_subfolder, exist_ok=True)

        keys_to_use = self.modalities if self.modalities is not None else ['tabular']

        for cls_id in target_class_ids:
            c_name = class_names[cls_id]
            
            # Determine which weights to use
            if cls_id in correct_samples_attn:
                weights = correct_samples_attn[cls_id]
                prefix = "CORRECT"
            elif cls_id in best_effort_attn:
                weights = best_effort_attn[cls_id]
                prefix = "BEST_EFFORT_FAIL" # Labeling it clearly as a misclassification
            else:
                continue

            file_name = f"{prefix}_{c_name}.png"
            full_path = os.path.join(run_subfolder, file_name)
            
            plot_patient_attention(weights.unsqueeze(0), keys_to_use, 
                                        save_path=full_path, input_years=self.input_years)

            if wandb.run is not None:
                wandb.log({f"attention/{c_name}": wandb.Image(full_path, caption=f"{prefix} {c_name}")})

        # Return standard metrics
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        evaluate_classification(y_true, y_pred, y_prob)
        
        return np.mean(losses), y_pred, y_prob, y_true

def debug_batch_logic(x_rays, mask, model_output):
    print("\n" + "="*40)
    print("LOGIC CHECK")
    print("="*40)
    
    # Check 1: Are the images and masks aligned?
    # x_rays shape: (B, T, C, H, W)
    # mask shape: (B, T)
    for t in range(x_rays.size(1)):
        img_std = torch.std(x_rays[0, t]).item()
        is_masked = mask[0, t].item() # True = Padding
        
        status = "❌ ERROR"
        if img_std == 0 and is_masked: status = "✅ OK (Padding)"
        if img_std > 0 and not is_masked: status = "✅ OK (Real Data)"
        
        print(f"Time {t}: Image Std={img_std:.2f} | Masked={is_masked} | Result: {status}")

    # Check 2: What is the model doing?
    # If the output is the same for every batch, the model isn't learning.
    print(f"\nModel Output (CLS Token):")
    print(f"  Shape: {model_output.shape}")
    print(f"  Values: {model_output[0, :5]}...") # Show first 5 values
    print("="*40 + "\n")

def print_confidence_stats(y_pred_probs, y_true, class_names):
    if hasattr(y_pred_probs, 'detach'): y_pred_probs = y_pred_probs.detach().cpu().numpy()
    if hasattr(y_true, 'detach'): y_true = y_true.detach().cpu().numpy()

    mask = (y_true >= 0)
    y_probs_clean = y_pred_probs[mask]

    # Calculate mean probability for each column output by the model
    avg_probs = y_probs_clean.mean(axis=0)
    
    print("\n🔮 AVERAGE PREDICTION CONFIDENCE:")
    for i, prob in enumerate(avg_probs):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"  {name}: {prob:.2f}")

def evaluate_classification(y_true, y_pred, y_probs, class_names=None, num_classes=4):
    # Filter out ignore_index (-100)
    mask = (y_true >= 0)
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_probs = y_probs[mask] # Ensure probabilities are masked too
    
    unique_classes = np.sort(np.unique(y_true))
    per_class_auc = {}
    
    # Calculate AUC using probabilities
    for i in range(len(unique_classes)):
        class_idx = int(unique_classes[i])
        y_true_binary = (y_true == class_idx).astype(int)
        try:
            # We use probabilities for the specific class column
            per_class_auc[f"auc_class_{class_idx}"] = roc_auc_score(y_true_binary, y_probs[:, class_idx])
        except:
            per_class_auc[f"auc_class_{class_idx}"] = 0.5

    if class_names is None:
        standard_names = {0: "Low-Stable", 1: "Improving", 2: "Worsening", 3: "High-Persistent"}
        class_names = [standard_names.get(int(c), f"Class {int(c)}") for c in unique_classes]

    print("\n" + "=" * 80 + "\n📊 CLASSIFICATION METRICS\n" + "=" * 80)
    auc_str = ", ".join([f"C{i}: {per_class_auc.get(f'auc_class_{i}', 0):.2f}" for i in range(len(unique_classes))])
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f} | Balanced Acc: {balanced_accuracy_score(y_true, y_pred):.2f} | AUC: [{auc_str}]")
    
    print("\n📉 Confusion Matrix (counts):")
    print("Columns = PREDICTED | Rows = GROUND TRUTH") # Added explanation
    print(f"{'':<17}", "".join([f"[{name[:10]:^10}]" for name in class_names]))
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:^12}" for val in row])
        print(f"{class_names[i]:<17} {row_str}")

    print("\n📋 Detailed Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=class_names, zero_division=0))
    print_confidence_stats(y_probs, y_true, class_names)
    
def create_weighted_sampler(labels, target_ratios=None):
    """
    labels: array or tensor of labels (0, 1, 2)
    target_ratios: dict mapping class_id to desired percentage. 
                   If None, defaults to equal distribution (balanced).
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
        
    valid_mask = labels != -1
    clean_labels = labels[valid_mask]
    
    unique_classes, class_sample_counts = np.unique(clean_labels, return_counts=True)
    
    # 1. Handle Default: Equal distribution (1/num_classes)
    if target_ratios is None:
        num_classes = len(unique_classes)
        target_ratios = {cls: 1.0 / num_classes for cls in unique_classes}
        print(f"⚖️ No target ratios provided. Defaulting to balanced: {target_ratios}")

    # 2. Calculate weights: Target_Percentage / Actual_Count
    # This ensures that Class 2 (rare) gets a huge weight per sample
    weight_dict = {}
    for cls, count in zip(unique_classes, class_sample_counts):
        target_prob = target_ratios.get(cls, 0.0)
        weight_dict[cls] = target_prob / count
    
    # 3. Map weights back to the full dataset size
    samples_weight = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if label != -1 and label in weight_dict:
            samples_weight[i] = weight_dict[label]
        else:
            samples_weight[i] = 0.0 
            
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(samples_weight).double(), 
        num_samples=len(samples_weight), # We keep the same epoch size
        replacement=True
    )
    return sampler

def rank_oai_features(X, y, feature_names, top_n=20):
    """
    X: shape (N, T, 67)
    y: shape (N,) derived labels
    """
    # 1. Flatten Time Dimension: Use the mean across the input years
    # This captures the 'baseline state' of the patient
    X_flat = np.nanmean(X, axis=1) 
    
    # Handle any remaining NaNs from the mean operation
    X_flat = np.nan_to_num(X_flat, nan=0.0)

    # 2. Train a Balanced Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        random_state=42,
        max_depth=10
    )
    rf.fit(X_flat, y)

    # 3. Map Importance to Names
    importances = rf.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names.values())
    sorted_feats = feat_importances.sort_values(ascending=False)

    # print("\n🏆 TOP PREDICTIVE FEATURES FOR PROGRESSION:")
    # print(sorted_feats.head(top_n))
    
    return sorted_feats.index[:top_n].tolist()

def visualize_attention(patient_id, tokens, attn_weights, feature_names):
    """
    attn_weights shape: (Heads, Seq_Len, Seq_Len)
    We care about the first row: What the CLS token (index 0) attended to.
    """
    # Average across heads or pick one head
    avg_attn = attn_weights.mean(dim=0) 
    cls_attn = avg_attn[0, 1:].cpu().numpy() # Skip CLS token itself

    plt.figure(figsize=(12, 4))
    sns.heatmap(cls_attn.reshape(1, -1), annot=True, xticklabels=feature_names, cmap='viridis')
    plt.title(f"Attention Map for Successful Worsening Prediction (PID: {patient_id})")
    plt.xlabel("Input Tokens (Years/Modalities)")
    plt.show()

def plot_patient_attention(model_output_weights, modality_keys, save_path, input_years):
    
    # 1. Directory setup
    folder_path = os.path.dirname(save_path)
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    # 2. Define labels
    labels = []
    for key in modality_keys:
        if input_years == 1:
            labels.append(f"{key}_Y0")
        else:
            for y in range(input_years):
                labels.append(f"{key}_Y{y}")
            labels.append(f"{key}_Delta")
    
    # 3. Extract weights
    if model_output_weights.ndim == 4:
        weights = model_output_weights[0].mean(dim=0)[0, 1:].cpu().detach().numpy()
    else:
        weights = model_output_weights[0, 0, 1:].cpu().detach().numpy()

    # 4. Debug: Print Heatmap Details to Log
    print(f"\n--- Attention Heatmap Weights ---")
    for lbl, w in zip(labels, weights):
        print(f"  {lbl:15}: {w:.4f}")
    print(f"----------------------------------\n")

    # 5. Safety Check
    if len(weights) != len(labels):
        print(f"⚠️ Warning: Weight size ({len(weights)}) != Label size ({len(labels)})")
        labels = labels[:len(weights)]

    # 6. Plotting
    plt.figure(figsize=(16, 4))
    sns.heatmap(weights.reshape(1, -1), xticklabels=labels, annot=True, 
                cmap="YlGnBu", cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f"Test Attention Map | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    plt.xlabel("Input Features")
    plt.xticks(rotation=45, ha='right')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Test attention plot saved: {save_path}")