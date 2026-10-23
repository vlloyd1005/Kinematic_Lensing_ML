#!/bin/bash
#SBATCH --job-name=kl_uq
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/kl_uq_%A.out
#SBATCH --error=logs/kl_uq_%A.err

module load miniconda/3
source ~/.bashrc
source /lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/miniforge/etc/profile.d/conda.sh
conda deactivate
conda activate vic_kl

cd "$SLURM_SUBMIT_DIR"

member=$SLURM_ARRAY_TASK_ID
outdir="kl_dataset/model_output_uq"
mkdir -p "$outdir"

echo "======================================"
echo "Ensemble member = $member"
echo "Job ID          = $SLURM_JOB_ID"
echo "Node            = $(hostname)"
echo "Start time      = $(date)"
echo "======================================"

export MPLBACKEND=Agg

python kl_dataset/train_kl_model_uq.py --use_original_image \
    --images_root kl_dataset/images \
    --csv kl_dataset/data/dataset_plan_to500_10draws.csv \
    --outdir "$outdir" \
    --npix 128 --epochs 60 --batch_size 64 --lr 3e-4 --weight_decay 1e-4 \
    --num_workers 8 --patience 12 --min_vel_fill 0.1 --smooth_sigma 0 \
    --n_members 5 --aggregate_only

status=$?
echo "======================================"
echo "Member $member finished, return code = $status, end time = $(date)"
echo "======================================"
exit $status