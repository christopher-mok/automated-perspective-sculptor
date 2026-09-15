#!/bin/bash
#SBATCH --job-name=neg_sun_moon_hinge2_restart_importnet_unconstrained_nonorm1p25
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/negspace/neg_h2ri_adddel_n5_unconstrained_20260915/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/negspace/neg_h2ri_adddel_n5_unconstrained_20260915/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-hinge2/images/sun.png --target2 /oscar/home/cjmok/aps-hinge2/images/moon.png --arm srd --overlap-mode planar --seed 0 --steps 4000 --early-stop --min-steps 1500 --patience-steps 300 --eval-interval 25 --render-every 25 --render-every-scale 1 --n-patches 5 --swept-resolution 256 --swept-spawn-fraction 1.0 --srd-candidates 32 --srd-min-patch-area 0.01 --lambda-count 0.05 --silhouette-weight 2 --negative-space-weight 2.5 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-hinge2/results/negspace/neg_h2ri_adddel_n5_unconstrained_20260915/neg_sun_moon_hinge2_restart_importnet_unconstrained_nonorm1p25 --count-objective --count-target-iou 0.91 --count-power 1 --count-margin-reward 5 --count-hard-floor --count-start-mode on_target --count-start-margin 0.02 --count-start-patience 2 --delete-eval-steps 15 --shed-patience-evals 8 --count-lambda 500 --deletion-importance --deletion-temperature 1 --deletion-proxy net --conflict-restart --unconstrained-theta --split-weight 0
