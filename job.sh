#!/bin/bash
#SBATCH --job-name=tryBlueBear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbdefault
#SBATCH --account=fengyv-kaiji
#SBATCH --time=03:00:00
#SBATCH --mem=64G 
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err


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

python3 -c "
import json
with open('evaluation_data/sasrec_candidates.json') as f:
    data = json.load(f)
first_key = list(data.keys())[0]
print('First user key:', first_key)
print('Keys:', list(data[first_key].keys()))
print('Sample:', json.dumps(data[first_key], indent=2)[:500])
"

echo ''
echo 'Job finished at:' $(date)

INNEREOF
