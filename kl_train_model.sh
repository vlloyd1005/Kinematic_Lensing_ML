#!/bin/bash
#SBATCH --job-name=kl_train
#SBATCH --output=/gpfs/projects/MirandaGroup/vic/Kinematic_Lensing_ML/kl_dataset/logs/train_%x_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu
# Notes on resource choices:
#   partition=a100-long : milan login nodes, 4x A100 80GB, max 48hr, shared node
#   gpu:a100:1          : request 1 of the 4 A100s (shared node — be a good neighbour)
#   cpus-per-task=8     : 8 workers for DataLoader; matches --num_workers below
#   mem=64G             : sufficient for 128px images + model; node has 256GB total

echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
module purge > /dev/null 2>&1
module load slurm
echo "Running on host: $(hostname)"
echo "Time is:         $(date)"
echo "Directory is:    $(pwd)"
echo "Job name:        $SLURM_JOB_NAME"
echo "Job ID:          $SLURM_JOBID"

source ~/.bashrc
source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
conda deactivate
conda activate vic_kl

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Redirect HuggingFace + torch hub caches to project space.
# The default ~/.cache/huggingface fills your home quota fast (timm weights,
# tokenizers, etc.).  Project space has much more headroom.
export HF_HOME=/gpfs/projects/MirandaGroup/vic/.cache/huggingface
export TORCH_HOME=/gpfs/projects/MirandaGroup/vic/.cache/torch
mkdir -p ${HF_HOME} ${TORCH_HOME}

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR=/gpfs/projects/MirandaGroup/vic/Kinematic_Lensing_ML/kl_dataset
IMAGES_ROOT=${SCRIPT_DIR}/images
CSV=${SCRIPT_DIR}/data/dataset_plan_with_ids.csv
OUT_DIR=${SCRIPT_DIR}/model_output

mkdir -p ${OUT_DIR}

# ── ensure numpy/sklearn binary compatibility ─────────────────────────────────
# sklearn compiled against a different numpy than what's active causes:
# "numpy.dtype size changed, may indicate binary incompatibility"
# Force a consistent reinstall to fix this before running.
pip install --quiet --upgrade --force-reinstall numpy scikit-learn

# ── install timm if needed ─────────────────────────────────────────────────────
pip install --quiet timm

# ── train ─────────────────────────────────────────────────────────────────────
srun python ${SCRIPT_DIR}/train_kl_model.py \
    --csv              dataset_plan_expanded_50draws.csv \
    --use_original_image \
    --images_root  ${IMAGES_ROOT} \
    --csv          ${CSV} \
    --outdir       ${OUT_DIR} \
    --npix         128 \
    --epochs       60 \
    --batch_size   32 \
    --lr           3e-4 \
    --weight_decay 1e-4 \
    --num_workers  8 \
    --patience     12 \
    --seed         42

echo ""
echo "=== Training complete ==="
echo "Time is: $(date)"
echo "Output files:"
ls -lh ${OUT_DIR}