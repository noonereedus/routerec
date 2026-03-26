#!/bin/bash
#SBATCH --job-name=tryBlueBear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbdefault
#SBATCH --account=fengyv-kaiji
#SBATCH --time=03:00:00
#SBATCH --mem=64G 
#SBATCH --output=logs/final-evaluation_%j.out
#SBATCH --error=logs/final-evaluation_%j.err


set -e  # fail fast

export HF_HOME=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
export HF_DATASETS_CACHE=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
mkdir -p $HF_HOME
mkdir -p logs

SCRATCH_ROOT=/rds/projects/f/fengyv-kaiji/vxk276
CONTAINER_PATH=/rds/projects/f/fengyv-kaiji/vxk276/containers/my_container.sif
PROJECT_ROOT=/rds/projects/f/fengyv-kaiji/vxk276

apptainer exec --nv --bind ${SCRATCH_ROOT}:${SCRATCH_ROOT} ${CONTAINER_PATH} bash <<'INNEREOF'

# Allow user site-packages (h5py is installed there)
export HF_HOME=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
export HF_DATASETS_CACHE=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
cd /rds/projects/f/fengyv-kaiji/vxk276/python/SASRec.pytorch/python

echo ''
echo 'Job started at:' $(date)

echo '=== Python Environment ==='
command -v python
python --version
echo ''

python final-evaluation.py --data_dir evaluation_data

echo ''
echo 'Job finished at:' $(date)

INNEREOF
