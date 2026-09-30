#!/bin/bash
#SBATCH --job-name=kl_smooth
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/kl_sigma_%A_%a.out
#SBATCH --error=logs/kl_sigma_%A_%a.err

module load miniconda/3

# If your environment is called something specific:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vic_kl

cd "$SLURM_SUBMIT_DIR"

sigmas=(0 1 1.5 2 4)
sigma=${sigmas[$SLURM_ARRAY_TASK_ID]}

outdir="kl_dataset/model_output_full/sigma_${sigma}"

mkdir -p "$outdir"
mkdir -p logs

echo "======================================"
echo "Starting sigma = $sigma"
echo "Job ID        = $SLURM_JOB_ID"
echo "Array ID      = $SLURM_ARRAY_TASK_ID"
echo "Node          = $(hostname)"
echo "Start time    = $(date)"
echo "======================================"

export MPLBACKEND=Agg

python kl_dataset/train_kl_model.py \
    --use_original_image \
    --images_root kl_dataset/images \
    --csv kl_dataset/data/dataset_plan_to500_10draws.csv \
    --outdir "$outdir" \
    --npix 128 \
    --epochs 60 \
    --batch_size 64 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --num_workers 8 \
    --patience 12 \
    --seed 42 \
    --smooth_sigma "$sigma" \
    --min_vel_fill 0.1

status=$?

echo "======================================"
echo "sigma = $sigma finished"
echo "Return code = $status"
echo "End time = $(date)"
echo "======================================"

exit $status