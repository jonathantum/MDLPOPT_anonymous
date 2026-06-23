#!/bin/bash
#SBATCH --job-name=MDLPOPT_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=48G              # Explicitly request more RAM for the CPU
#SBATCH --output="/mnt/nfs/homedirs/%u/slurm-output/slurm-%j.out"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

cd ${SLURM_SUBMIT_DIR}
# #/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  src/models/baseline/baseline_tabular_only_transformer_classification_2.py
# #/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  src/models/baseline/baseline_mri_transformer_classification.py
# #/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  src/models/multimodal_model/multimodal_model.py
# #/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  src/models/baseline/KL_baseline_xray_contrastive_L_classification.py
# #/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  src/models/baseline/KL_baseline_xray_resnet+transformer_classification.py
# #/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  src/models/baseline/baseline_xray_resnet+transformer_classification.py
# # /mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  src/models/baseline/baseline_xray_only_Resnet_classification.py
# # /mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python  data/clinical_raw/process_xray.py
# #/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python data/clinical_raw/new_process_tabular_data.py # s

# --- Define the script to run here ---
SCRIPT_TO_RUN="src/models/multimodal_model/multimodal_model.py"
PYTHON_BIN="/mnt/nfs/homedirs/$USER/miniconda3/envs/MDLPOPT/bin/python"

# 1. Debug: Check environment
echo "=========================================================="
echo "Job ID:        ${SLURM_JOBID}"
echo "Running on:    $(hostname)"
echo "Script Path:   $SCRIPT_TO_RUN"
echo "Python Bin:    $PYTHON_BIN"
echo "Start Time:    $(date)"
echo "=========================================================="

# 2. Check GPU status
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 3. Environment Setup
CONDA_ENV_PATH="/mnt/nfs/homedirs/anonymous/miniconda3/envs/MDLPOPT"

# --- FIX APPLIED HERE ---
# Clear the bad LD_PRELOAD that's causing the noise
unset LD_PRELOAD

# Redirect WandB away from the restricted /home path (Fixes the Permission Denied error)
export WANDB_DIR="/mnt/nfs/homedirs/anonymous/MDLPOPT/wandb"
export WANDB_CACHE_DIR="/mnt/nfs/homedirs/anonymous/MDLPOPT/.cache/wandb"
export WANDB_CONFIG_DIR="/mnt/nfs/homedirs/anonymous/MDLPOPT/.config/wandb"
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR

# Manually inject the environment into the PATH
export PATH="${CONDA_ENV_PATH}/bin:$PATH"

# Standard Python/Torch vars
export TORCH_HOME="/mnt/nfs/homedirs/anonymous/.cache/torch"
export MPLCONFIGDIR="/tmp/matplotlib_cache_anonymous"
mkdir -p $TORCH_HOME
mkdir -p $MPLCONFIGDIR

# 4. Execution
echo "Using Python from: $(which python)"
echo "Executing $SCRIPT_TO_RUN..."
echo "----------------------------------------------------------"

# Use the full path directly to be 100% sure
${CONDA_ENV_PATH}/bin/python -u "$SCRIPT_TO_RUN"

echo "----------------------------------------------------------"
echo "Job ${SLURM_JOBID} completed at $(date)"
