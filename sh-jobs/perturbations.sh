#!/bin/bash
#SBATCH --job-name=tryBlueBear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbgpu
#SBATCH --gpus=1
#SBATCH --account=fengyv-kaiji
#SBATCH --time=03:00:00
#SBATCH --mem=64G 
#SBATCH --output=logs/perturbations_%j.out
#SBATCH --error=logs/perturbations_%j.err


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

echo "=== Reverse order ==="
python rerank-dep.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --output_name qwen \
    --run_index 0 \
    --prompt_version main \
    --candidate_order reverse \
    --temperature 0.1 \
    --device cuda \
    --max_users 500

echo "=== Random order ==="
python rerank-dep.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --output_name qwen \
    --run_index 0 \
    --prompt_version main \
    --candidate_order random \
    --temperature 0.1 \
    --device cuda \
    --max_users 500

echo "=== Soft diversity ==="
python rerank-dep.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --output_name qwen \
    --run_index 0 \
    --prompt_version diversity_v1 \
    --temperature 0.1 \
    --device cuda \
    --max_users 500

echo "=== Metric-specific diversity ==="
python rerank-dep.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --output_name qwen \
    --run_index 0 \
    --prompt_version diversity_v2 \
    --temperature 0.1 \
    --device cuda \
    --max_users 500

echo "=== Reverse order t=0.0 ==="
python rerank-dep.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --output_name qwen \
    --run_index 0 \
    --prompt_version main \
    --candidate_order reverse \
    --temperature 0.0 \
    --device cuda \
    --max_users 500

echo ''
echo 'Job finished at:' $(date)

INNEREOF
