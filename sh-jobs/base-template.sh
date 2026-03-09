#!/bin/bash
#SBATCH --job-name=setupsqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err


set -e  # fail fast

SCRATCH_ROOT=/scratch/u5fi/byebyevlad.u5fi
CONTAINER_PATH=/scratch/u5fi/byebyevlad.u5fi/containers/my_container.sif
PROJECT_ROOT=/projects/u5fi/byebyevlad/

module load brics/apptainer-multi-node/0.3.2 || true


apptainer exec --nv --bind ${SCRATCH_ROOT}:${SCRATCH_ROOT} --bind ${PROJECT_ROOT}:${PROJECT_ROOT} --bind /home/u5fi/byebyevlad.u5fi:/home/u5fi/byebyevlad.u5fi ${CONTAINER_PATH} bash -c "
# Allow user site-packages (h5py is installed there)
export HF_HOME=${SCRATCH_ROOT}/hf_cache
cd ${PROJECT_ROOT}

echo '=== Python Environment ==='
which python
python --version
echo ''

python hello.py

"

echo ''
echo 'Job finished at:' $(date)
