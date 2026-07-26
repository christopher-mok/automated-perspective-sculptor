#!/bin/bash
#SBATCH --job-name=render_imports
# gpu-debug rather than the 3090-gcondo condo: this is a two-minute render, and
# the condo's group GPU allocation is routinely saturated (jobs sit on
# QOSGrpGRES). gpu-debug allows an hour and 4 GPUs per user.
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

source /oscar/home/cjmok/.bashrc
conda activate myenv

python render_import.py imports/horse1.json imports/sun1.json
