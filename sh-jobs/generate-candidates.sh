#!/bin/bash
#SBATCH --job-name=tryBlueBear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbdefault
#SBATCH --account=fengyv-kaiji
#SBATCH --time=03:00:00
#SBATCH --mem=64G 
#SBATCH --output=logs/generating_sasrec_%j.out
#SBATCH --error=logs/generating_sasrec_%j.err


set -e  # fail fast

export HF_HOME=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
export HF_DATASETS_CACHE=/rds/projects/f/fengyv-kaiji/vxk276/hf_cache
mkdir -p $HF_HOME
mkdir -p logs

SCRATCH_ROOT=/rds/projects/f/fengyv-kaiji/vxk276
CONTAINER_PATH=/rds/projects/f/fengyv-kaiji/vxk276/containers/my_container.sif
PROJECT_ROOT=/rds/projects/f/fengyv-kaiji/vxk276

apptainer exec --nv --bind ${SCRATCH_ROOT}:${SCRATCH_ROOT} ${CONTAINER_PATH} bash <<EOF

# Allow user site-packages (h5py is installed there)
export HF_HOME=${SCRATCH_ROOT}/hf_cache
export HF_DATASETS_CACHE=${SCRATCH_ROOT}/hf_cache
cd ${PROJECT_ROOT}/python/SASRec.pytorch/python

echo ''
echo 'Job started at:' $(date)

echo '=== Python Environment ==='
command -v python
python --version
echo ''

python generate-candidates.py --ml1m_path ../../../ml-1m

EOF

echo ''
echo 'Job finished at:' $(date)
