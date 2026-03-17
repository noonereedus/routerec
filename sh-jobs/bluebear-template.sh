#!/bin/bash
#SBATCH --job-name=tryBlueBear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbdefault
#SBATCH --account=fengyv-kaiji
#SBATCH --time=01:00:00
#SBATCH --mem=64G 
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err


set -e  # fail fast

export HF_HOME=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
export HF_DATASETS_CACHE=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
mkdir -p $HF_HOME

SCRATCH_ROOT=/rds/projects/f/fengyv-kaiji/vxk276
CONTAINER_PATH=/rds/projects/f/fengyv-kaiji/vxk276/containers/my_container.sif
PROJECT_ROOT=/rds/projects/f/fengyv-kaiji/vxk276

apptainer exec --nv --bind ${SCRATCH_ROOT}:${SCRATCH_ROOT} ${CONTAINER_PATH} bash <<EOF

# Allow user site-packages (h5py is installed there)
export HF_HOME=${SCRATCH_ROOT}/hf_cache
export HF_DATASETS_CACHE=${SCRATCH_ROOT}/hf_cache
cd ${PROJECT_ROOT}

echo '=== Python Environment ==='
command -v python
python --version
echo ''

python hello.py

EOF

echo ''
echo 'Job finished at:' $(date)
