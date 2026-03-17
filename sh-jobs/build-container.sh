#!/bin/bash
#SBATCH --job-name=build_container
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbdefault
#SBATCH --time=01:00:00
#SBATCH --account=fengyv-kaiji
#SBATCH --mem=64G
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

set -e

cd /rds/projects/f/fengyv-kaiji/vxk276
mkdir -p ./apptainer_cache
export APPTAINER_CACHEDIR=$(pwd)/apptainer_cache
export APPTAINER_TMPDIR=$(pwd)/apptainer_cache
apptainer build containers/my_container.sif baseline_apptainer.def
rm -rf ./apptainer_cache