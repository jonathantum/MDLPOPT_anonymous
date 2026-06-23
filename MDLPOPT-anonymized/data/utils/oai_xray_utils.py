import pandas as pd
import numpy as np
import torch
from pathlib import Path

class OAIImageTensorBuilder:
    def __init__(self, tabular_ids, metadata_path, output_dir):
        self.tabular_ids = list(tabular_ids)  # The N unique IDs from your tabular data
        self.df_meta = pd.read_csv(metadata_path)
        self.output_dir = Path(output_dir)
        
        # Mapping T (0-9) to OAI Visit Codes
        self.visit_map = {
            "V00": 0, "V01": 1, "V03": 2, "V05": 3, "V06": 4, 
            "V07": 5, "V08": 6, "V09": 7, "V10": 8, "V11": 9
        }
        
        # Mapping I (Image Types) to Indices
        self.type_map = {
            "Bilateral PA Fixed Flexion Knee": 0,
            "Full Limb": 1
            # You can add more types here later
        }

    def build_tensors(self):
        N = len(self.tabular_ids)
        T = 10  # 0 to 9
        I = len(self.type_map)
        
        # Initialize Tensors
        # Path tensor uses object dtype to store strings
        path_tensor = np.full((N, T, I), "", dtype=object)
        # Mask tensor: True if image exists, False otherwise
        mask_tensor = np.zeros((N, T, I), dtype=bool)
        
        # Create a fast lookup for tabular ID indices
        id_to_idx = {id_val: i for i, id_val in enumerate(self.tabular_ids)}
        
        # Filter metadata: only rows where the knee_id is in our tabular dataset
        # AND the visit/scan_type are in our maps
        valid_rows = self.df_meta[
            (self.df_meta['knee_id'].isin(id_to_idx)) & 
            (self.df_meta['visit'].isin(self.visit_map)) & 
            (self.df_meta['scan_type'].isin(self.type_map))
        ]
        
        print(f"Processing {len(valid_rows)} valid image entries for {N} patients...")

        for _, row in valid_rows.iterrows():
            n_idx = id_to_idx[row['knee_id']]
            t_idx = self.visit_map[row['visit']]
            i_idx = self.type_map[row['scan_type']]
            
            path_tensor[n_idx, t_idx, i_idx] = row['path']
            mask_tensor[n_idx, t_idx, i_idx] = True
            
        return path_tensor, mask_tensor

    def inspect_sample(self, path_tensor, mask_tensor, n_samples=5):
        """
        Guarantees a mix of samples with images and samples with placeholders.
        """
        print("\n" + "="*60)
        print(f"{'SANITY CHECK: MULTIMODAL TENSOR ALIGNMENT':^60}")
        print("="*60)
        
        # 1. Identify "Hits" (indices where at least one image exists)
        # .any(axis=(1, 2)) checks if True exists anywhere in the [T, I] grid for that patient
        hit_indices = np.where(mask_tensor.any(axis=(1, 2)))[0]
        
        # 2. Identify "Misses" (indices that are purely placeholders)
        miss_indices = np.where(~mask_tensor.any(axis=(1, 2)))[0]
        
        # 3. Select a balanced mix for the report
        # We'll try to show 3 hits and the rest misses (up to n_samples)
        to_check = []
        if len(hit_indices) > 0:
            num_hits = min(3, len(hit_indices))
            to_check.extend(np.random.choice(hit_indices, num_hits, replace=False))
            
        if len(miss_indices) > 0:
            num_misses = n_samples - len(to_check)
            to_check.extend(np.random.choice(miss_indices, num_misses, replace=False))

        # 4. Print the detailed report
        for idx in to_check:
            knee_id = self.tabular_ids[idx]
            has_data = mask_tensor[idx].any()
            
            status_msg = "DATA FOUND" if has_data else "PLACEHOLDER"
            print(f"\n[Index {idx:04d}] Knee ID: {knee_id:<10} | Status: {status_msg}")
            
            if not has_data:
                print(f"  -> No X-rays matched. Mask is correctly set to False for all T.")
            else:
                # Loop through T and I to show where the data lives
                for t in range(path_tensor.shape[1]):
                    for i_name, i_idx in self.type_map.items():
                        if mask_tensor[idx, t, i_idx]:
                            p = path_tensor[idx, t, i_idx]
                            print(f"  [T={t}] {i_name:<30} | Path: .../{Path(p).name}")
                            print(f"Path: {Path(p).name}")

        print("\n" + "="*60)