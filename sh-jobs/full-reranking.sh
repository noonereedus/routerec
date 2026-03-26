#!/bin/bash
#SBATCH --job-name=tryBlueBear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbgpu
#SBATCH --gpus=1
#SBATCH --account=fengyv-kaiji
#SBATCH --time=03:00:00
#SBATCH --mem=64G 
#SBATCH --output=logs/dependability-study_%j.out
#SBATCH --error=logs/dependability-study_%j.err


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

for RUN in 0 1 2; do
    echo "===== Qwen run $RUN ====="
    python rerank-dep.py \
        --model_name "Qwen/Qwen2-7B-Instruct" \
        --output_name qwen \
        --run_index $RUN \
        --prompt_version main \
        --temperature 0.1 \
        --device cuda \
        --max_users 500
done

for RUN in 0 1 2; do
    echo "===== Nemotron run $RUN ====="
    python rerank-dep.py \
        --model_name "nvidia/Nemotron-Mini-4B-Instruct" \
        --output_name nemotron \
        --run_index $RUN \
        --prompt_version main \
        --temperature 0.1 \
        --device cuda \
        --max_users 500
done

for RUN in 0 1 2; do
    python rerank-dep.py \
        --model_name "Qwen/Qwen2-7B-Instruct" \
        --output_name qwen \
        --run_index $RUN \
        --prompt_version main \
        --temperature 0.0 \
        --device cuda \
        --max_users 500
done

python rerank-dep.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --output_name qwen \
    --run_index 0 \
    --prompt_version rephrase_v1 \
    --temperature 0.1 \
    --device cuda \
    --max_users 500

echo ''
echo 'Job finished at:' $(date)

INNEREOF
