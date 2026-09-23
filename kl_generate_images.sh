#!/bin/bash
#SBATCH --job-name=kl_generate
#SBATCH --output=/gpfs/projects/MirandaGroup/vic/Kinematic_Lensing_ML/kl_dataset/logs/generate_%x_%a_%A.txt
#SBATCH --time=48:00:00
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-2          # one array task per snapshot (4 snaps total)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu

echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
module purge > /dev/null 2>&1
module load slurm
echo "Running on host: $(hostname)"
echo "Time is:         $(date)"
echo "Directory is:    $(pwd)"
echo "Job name:        $SLURM_JOB_NAME"
echo "Job ID:          $SLURM_JOBID"
echo "Array task ID:   $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
conda deactivate
conda activate vic

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR=/gpfs/projects/MirandaGroup/vic/Kinematic_Lensing_ML/kl_dataset
LOG_DIR=${SCRIPT_DIR}/logs
DATA_DIR=${SCRIPT_DIR}/data
OUT_DIR=${SCRIPT_DIR}/images

mkdir -p ${LOG_DIR} ${OUT_DIR}

TNG_API_KEY="16a29db7f934e4d33640dcd47e7f80be"   # ← paste your key
export TNG_API_KEY

# ── map array task ID → snapshot number ──────────────────────────────────────
# Each SLURM array task handles all galaxies for one snapshot, running
# sequentially through that snap's rows.  This respects the TNG API rate
# limit (~300 req/hr) while keeping wall-clock time manageable.
#
# FULL snapshots only — mini snaps lack GFM_StellarPhotometrics (HTTP 400).
# Array index → snapshot:
#   0 → snap 40  (z = 1.50)   upper H-alpha + [O III] overlap
#   1 → snap 50  (z = 1.00)   peak of KL dN/dz
#   2 → snap 59  (z = 0.70)   H-alpha mid-range
#   3 → snap 67  (z = 0.50)   low-z tail
SNAPS=(40 50 59 67)
SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "=== Array task ${SLURM_ARRAY_TASK_ID}: processing snapshot ${SNAP} ==="

# ── extract the rows for this snapshot into a per-task CSV ───────────────────
TASK_CSV=${DATA_DIR}/task_snap${SNAP}.csv
python - <<PYEOF
import pandas as pd
plan = pd.read_csv("${DATA_DIR}/dataset_plan_with_ids.csv")
subset = plan[plan["snap"] == ${SNAP}]
subset.to_csv("${TASK_CSV}", index=False)
print(f"Snap ${SNAP}: {len(subset)} rows written to ${TASK_CSV}")
PYEOF

if [ ! -f "${TASK_CSV}" ]; then
    echo "ERROR: task CSV not created for snap ${SNAP}. Exiting."
    exit 1
fi

echo "Rows to process:"
wc -l ${TASK_CSV}

# ── run image generation for this snapshot ────────────────────────────────────
# --skip_existing lets you safely requeue failed tasks without re-downloading
# galaxies that already completed.
srun python ${SCRIPT_DIR}/generate_kl_tng50.py \
    --csv            ${TASK_CSV} \
    --outdir         ${OUT_DIR}/snap${SNAP} \
    --apikey         ${TNG_API_KEY} \
    --sim            TNG50-1 \
    --snap           ${SNAP} \
    --npix           256 \
    --fov_kpc        30 \
    --skip_existing

EXIT_CODE=$?

echo ""
echo "=== Snapshot ${SNAP} finished with exit code ${EXIT_CODE} ==="
echo "Time is: $(date)"

# ── quick completion summary ─────────────────────────────────────────────────
python - <<PYEOF
import os
from pathlib import Path

out_dir = Path("${OUT_DIR}/snap${SNAP}")
if not out_dir.exists():
    print("Output directory not found.")
else:
    gal_dirs  = [d for d in out_dir.iterdir() if d.is_dir()]
    complete  = [d for d in gal_dirs
                 if (d / f"{d.name}_image_original.fits").exists()
                 and (d / f"{d.name}_velmap_original.fits").exists()]
    print(f"Snap ${SNAP}: {len(complete)}/{len(gal_dirs)} galaxies complete")
    sizes = [
        sum(f.stat().st_size for f in d.iterdir() if f.suffix == ".fits")
        for d in complete
    ]
    if sizes:
        total_mb = sum(sizes) / 1e6
        print(f"Total FITS data: {total_mb:.1f} MB  "
              f"({total_mb/len(complete):.2f} MB/galaxy)")
PYEOF

exit ${EXIT_CODE}