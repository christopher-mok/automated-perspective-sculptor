#!/bin/bash
#SBATCH --job-name=neg_dance_argument_importnet_nonorm4
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/negspace/neg_importnet_20260724/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/negspace/neg_importnet_20260724/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-hinge2/images/dance.png --target2 /oscar/home/cjmok/aps-hinge2/images/argument.png --arm srd --overlap-mode planar --seed 0 --steps 20000 --early-stop --min-steps 1500 --patience-steps 300 --eval-interval 25 --render-every 25 --render-every-scale 1 --n-patches 20 --swept-resolution 256 --swept-spawn-fraction 1.0 --silhouette-weight 0.9 --negative-space-weight 3.6 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-hinge2/results/negspace/neg_importnet_20260724/neg_dance_argument_importnet_nonorm4 --deletion-importance --deletion-temperature 1 --deletion-proxy net
