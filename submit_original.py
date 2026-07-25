"""Submit the original-weights batch from the `original` branch to SLURM.

This branch is 8e0580f4 -- the last commit before the seven features that
landed as uncommitted work between 2026-07-22 and 2026-07-24 -- plus three of
them ported back deliberately:

    render snapshots        --render-every / --render-every-scale
    conflict-gated restart  --conflict-restart / --conflict-eps
    deletion lookahead      --delete-eval-steps

Everything else from that period is absent: no area-normalized view loss, no
fit-then-shed count scheduling, no hard floor or margin reward, no deletion
importance sampling. Per-panel (planar) overlap with the hard repair pass is
already in the baseline and is the default; it is passed explicitly here so the
sbatch record shows it.

Weights are the SceneOptimizer class defaults -- silhouette 2.0, negative space
2.5, non-normalized -- which is what every run before the negative-space batches
used implicitly, and what "original weights" refers to.

Examples
--------
    python submit_original.py --dry-run
    python submit_original.py
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from submit_ablations import IMAGE_PAIRS

_PROJECT_ROOT = Path(__file__).resolve().parent
ORIGINAL_ROOT = _PROJECT_ROOT / "results" / "original"

# The seven standard pairs, in the order the negspace batches use them.
PAIRS = (
    "sun_moon", "water_fire", "cat_face_bass", "dance_argument",
    "horse_circle", "acm_scf", "teapot_droplets",
)

# The SceneOptimizer class defaults.
SILHOUETTE_WEIGHT = 2.0
NEGATIVE_SPACE_WEIGHT = 2.5

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --gres={gres}
#SBATCH --chdir={project_root}
#SBATCH --output={log_dir}/%x.%j.out
#SBATCH --error={log_dir}/%x.%j.err

set +eu
{env_setup}
set -eu

ulimit -n 50000

{command}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the original-weights batch to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-name", default=None,
                        help="Batch dir under results/original/ (default: orig_<timestamp>).")
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--steps", type=int, default=20000, help="Hard step ceiling.")
    parser.add_argument("--min-steps", type=int, default=1500)
    parser.add_argument("--patience-steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--render-every-scale", type=int, default=2)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--swept-resolution", type=int, default=300)
    parser.add_argument("--swept-spawn-fraction", type=float, default=1.0)
    parser.add_argument("--srd-candidates", type=int, default=64)
    parser.add_argument("--lambda-count", type=float, default=0.0)
    parser.add_argument("--delete-eval-steps", type=int, default=1,
                        help="Left at the historical 1 so the deletion "
                             "lookahead does not outrun the 4-step add "
                             "lookahead and bias this batch toward deletes.")
    parser.add_argument("--conflict-eps", type=float, default=1e-3)
    parser.add_argument("--no-conflict-restart", action="store_true")
    parser.add_argument("--render-scale", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=23.0)
    parser.add_argument("--device", default="cuda")

    cluster = parser.add_argument_group("cluster resources")
    cluster.add_argument("--partition", default="3090-gcondo")
    cluster.add_argument("--gres", default="gpu:1")
    cluster.add_argument("--time", default="24:00:00")
    cluster.add_argument("--mem", default="125G")
    cluster.add_argument("--cpus", type=int, default=6)
    cluster.add_argument("--python", default="python")
    cluster.add_argument("--env-setup",
                         default="source /oscar/home/cjmok/.bashrc\nconda activate myenv")

    mode = parser.add_argument_group("actions")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pairs is None:
        args.pairs = list(PAIRS)
    if args.sweep_name is None:
        args.sweep_name = f"orig_{datetime.now():%Y%m%d_%H%M}"
    return args


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        job_name = f"orig_{pair}"
        output_dir = sweep_dir / job_name
        command = [
            args.python, "run_final.py",
            "--target1", str(image_dir / target1),
            "--target2", str(image_dir / target2),
            "--arm", "srd",
            # Per-panel overlap plus the hard repair pass. Both are already the
            # defaults on this branch; passed explicitly so the record is
            # unambiguous rather than inherited.
            "--overlap-mode", "planar",
            "--overlap-repair",
            "--seed", str(args.seed),
            "--steps", str(args.steps),
            "--early-stop",
            "--min-steps", str(args.min_steps),
            "--patience-steps", str(args.patience_steps),
            "--eval-interval", str(args.eval_interval),
            "--render-every", str(args.render_every),
            "--render-every-scale", str(args.render_every_scale),
            "--n-patches", str(args.n_patches),
            "--swept-resolution", str(args.swept_resolution),
            "--swept-spawn-fraction", str(args.swept_spawn_fraction),
            "--srd-candidates", str(args.srd_candidates),
            "--delete-eval-steps", str(args.delete_eval_steps),
            "--lambda-count", f"{args.lambda_count:g}",
            "--silhouette-weight", f"{SILHOUETTE_WEIGHT:g}",
            "--negative-space-weight", f"{NEGATIVE_SPACE_WEIGHT:g}",
            "--render-scale", str(args.render_scale),
            "--max-hours", str(args.max_hours),
            "--device", args.device,
            "--output-dir", str(output_dir),
        ]
        if not args.no_conflict_restart:
            command += ["--conflict-restart", "--conflict-eps", f"{args.conflict_eps:g}"]
        jobs.append({
            "name": job_name, "pair": pair,
            "output_dir": output_dir, "command": command,
        })
    return jobs


def _write_sbatch_script(job: dict, args: argparse.Namespace, sweep_dir: Path) -> Path:
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = SBATCH_TEMPLATE.format(
        job_name=job["name"], partition=args.partition, gres=args.gres,
        time=args.time, mem=args.mem, cpus=args.cpus, log_dir=log_dir,
        project_root=_PROJECT_ROOT, env_setup=args.env_setup,
        command=shlex.join(job["command"]),
    )
    script_path = sweep_dir / "scripts" / f"{job['name']}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _submit(script_path: Path) -> str:
    result = subprocess.run(["sbatch", str(script_path)],
                            capture_output=True, text=True, check=True)
    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _write_manifest(sweep_dir: Path, rows: list[dict]) -> Path:
    manifest_path = sweep_dir / "manifest.tsv"
    lines = ["job_id\tjob_name\tpair\tsilhouette_weight\tnegative_space_weight\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t"
            f"{SILHOUETTE_WEIGHT:.6g}\t{NEGATIVE_SPACE_WEIGHT:.6g}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = _parse_args()
    sweep_dir = ORIGINAL_ROOT / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args, sweep_dir)
    print(f"[Sweep] original/{args.sweep_name}: {len(jobs)} job(s)")
    print(f"[Sweep]   weights silhouette={SILHOUETTE_WEIGHT:g} "
          f"negative_space={NEGATIVE_SPACE_WEIGHT:g} (non-normalized, class defaults)")
    print(f"[Sweep]   planar overlap + repair, fan fill, swept volume "
          f"{args.swept_resolution} at {args.swept_spawn_fraction:g} spawn, "
          f"{args.srd_candidates} SRD candidates")
    print(f"[Sweep]   conflict_restart={not args.no_conflict_restart}, "
          f"delete_eval_steps={args.delete_eval_steps}, "
          f"image sequence every {args.render_every} steps at scale "
          f"{args.render_every_scale}")

    for job in jobs:
        script_path = _write_sbatch_script(job, args, sweep_dir)
        if args.dry_run:
            print(f"[Dry run] wrote {script_path}")
            continue
        job["job_id"] = _submit(script_path)
        print(f"[Submitted] {job['job_id']}: {job['name']}")

    manifest_path = _write_manifest(sweep_dir, jobs)
    print(f"[Sweep] manifest written to {manifest_path}")
    if not args.dry_run:
        print("Monitor with: squeue --me | grep orig_")


if __name__ == "__main__":
    main()
