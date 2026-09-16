#!/bin/bash
#SBATCH --job-name=neg_teapot_droplets_srd_restart_softiou
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/negspace/neg_teapot_softiou_restart_c64_np30_20260916/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/negspace/neg_teapot_softiou_restart_c64_np30_20260916/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/teapot.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/droplets.png --arm srd --overlap-mode planar --seed 0 --steps 4000 --early-stop --min-steps 1500 --patience-steps 300 --eval-interval 25 --render-every 25 --render-every-scale 1 --n-patches 30 --swept-resolution 256 --swept-spawn-fraction 1.0 --srd-candidates 64 --srd-min-patch-area 0.01 --lambda-count 0.05 --lr 0.0035 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/negspace/neg_teapot_softiou_restart_c64_np30_20260916/neg_teapot_droplets_srd_restart_softiou --view-loss softiou --conflict-restart
