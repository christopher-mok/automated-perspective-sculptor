#!/bin/bash
#SBATCH --job-name=water_fire_full
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=125G
#SBATCH --cpus-per-task=6
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/srd_ablation/logs/%x_%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/srd_ablation/logs/%x_%j.err

set -euo pipefail
cd /oscar/home/cjmok/automated-perspective-sculptor
# Relax errexit/nounset while sourcing shell-init and conda: Oscar's
# /etc/bashrc references unbound vars (e.g. BASHRCSOURCED) and conda
# activate can return non-zero, either of which aborts the job otherwise.
set +eu
source /oscar/home/cjmok/.bashrc && conda activate myenv
set -eu
python run_ablation.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/water.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/fire.png --mode full --steps 500 --n-patches 20 --lr 0.0035 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/srd_ablation/water_fire_full --trials 3
