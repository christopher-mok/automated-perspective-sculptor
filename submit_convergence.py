"""Submit time-to-convergence jobs (SRD vs. base) for every image pair.

One SLURM job per (pair, method, seed) so that each job stays inside the
partition walltime: with a 10h/trial cap, packing 3 trials into one job would
need 30h.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_convergence.py --dry-run

    # Submit all 5 pairs x 2 methods x 3 seeds = 30 jobs
    python submit_convergence.py

    # After jobs finish: aggregate every convergence.txt into one table
    python submit_convergence.py --collect
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

METHODS = ("srd", "base")

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
        description="Submit SRD-vs-base convergence jobs to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory name (default: convergence_<timestamp>).",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        choices=sorted(IMAGE_PAIRS),
        default=sorted(IMAGE_PAIRS),
    )
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    parser.add_argument("--trials", type=int, default=3, help="Seeds per (pair, method).")
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--swept-resolution", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--plateau-delta", type=float, default=0.002,
                        help="Relative loss-improvement floor over the window.")
    parser.add_argument("--plateau-window", type=int, default=250)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--target-iou", type=float, default=0.85,
                        help="Reported-only mean_iou milestone (never stops the run). "
                             "Calibrated to silhouette IoU, which flattens near 0.80 "
                             "on horse_circle; iou_history.csv allows any other "
                             "threshold to be recomputed after the fact.")
    parser.add_argument("--device", default="cuda")

    cluster = parser.add_argument_group("cluster resources")
    cluster.add_argument("--partition", default="3090-gcondo")
    cluster.add_argument("--gres", default="gpu:1")
    cluster.add_argument("--time", default="11:00:00", help="SLURM walltime (> --max-hours).")
    cluster.add_argument("--mem", default="125G")
    cluster.add_argument("--cpus", type=int, default=6)
    cluster.add_argument("--python", default="python")
    cluster.add_argument(
        "--env-setup",
        default="source /oscar/home/cjmok/.bashrc\nconda activate myenv",
    )

    mode = parser.add_argument_group("actions")
    mode.add_argument("--dry-run", action="store_true", help="Write scripts, submit nothing.")
    mode.add_argument("--collect", action="store_true", help="Aggregate results and exit.")
    return parser.parse_args()


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        for method in args.methods:
            for seed in range(args.trials):
                job_name = f"conv_{pair}_{method}_seed{seed}"
                output_dir = sweep_dir / job_name
                command = [
                    args.python, "compare_convergence.py",
                    "--target1", str(image_dir / target1),
                    "--target2", str(image_dir / target2),
                    "--method", method,
                    "--seed", str(seed),
                    "--n-patches", str(args.n_patches),
                    "--swept-resolution", str(args.swept_resolution),
                    "--max-steps", str(args.max_steps),
                    "--max-hours", str(args.max_hours),
                    "--eval-interval", str(args.eval_interval),
                    "--plateau-delta", str(args.plateau_delta),
                    "--plateau-window", str(args.plateau_window),
                    "--plateau-patience", str(args.plateau_patience),
                    "--target-iou", str(args.target_iou),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ]
                jobs.append({
                    "name": job_name,
                    "pair": pair,
                    "method": method,
                    "seed": seed,
                    "output_dir": output_dir,
                    "command": command,
                })
    return jobs


def _write_sbatch_script(job: dict, args: argparse.Namespace, sweep_dir: Path) -> Path:
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = SBATCH_TEMPLATE.format(
        job_name=job["name"],
        partition=args.partition,
        gres=args.gres,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
        log_dir=log_dir,
        project_root=_PROJECT_ROOT,
        env_setup=args.env_setup,
        command=shlex.join(job["command"]),
    )
    script_path = sweep_dir / "scripts" / f"{job['name']}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _submit(script_path: Path) -> str:
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _write_manifest(sweep_dir: Path, rows: list[dict]) -> Path:
    manifest_path = sweep_dir / "manifest.tsv"
    lines = ["job_id\tjob_name\tpair\tmethod\tseed\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t"
            f"{row['method']}\t{row['seed']}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def _collect(sweep_dir: Path) -> None:
    """Aggregate every convergence.txt into a per-trial table plus per-arm means."""
    reports = sorted(sweep_dir.glob("*/convergence.txt"))
    if not reports:
        print(f"No convergence.txt files found under {sweep_dir}")
        return

    def _grab(text: str, key: str) -> str:
        match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else "-"

    rows = []
    for report in reports:
        text = report.read_text(encoding="utf-8")
        rows.append({
            "job": report.parent.name,
            "method": _grab(text, "method"),
            "pair": report.parent.name.replace("conv_", "").rsplit("_", 2)[0],
            "stop_reason": _grab(text, "stop_reason"),
            "plateau_steps": _grab(text, "plateau_steps"),
            "plateau_seconds": _grab(text, "plateau_seconds"),
            "plateau_mean_iou": _grab(text, "plateau_mean_iou"),
            "target_reached": _grab(text, "target_iou_reached"),
            "target_steps": _grab(text, "target_iou_steps"),
            "target_seconds": _grab(text, "target_iou_seconds"),
            "best_iou": _grab(text, "best_mean_iou"),
        })

    header = (
        f"{'job':<34} {'stop':<10} {'plat_steps':>10} {'plat_s':>10} "
        f"{'plat_iou':>9} {'hit_tgt':>7} {'t_steps':>8} {'t_secs':>10} {'best_iou':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['job']:<34} {row['stop_reason']:<10} {row['plateau_steps']:>10} "
            f"{row['plateau_seconds']:>10} {row['plateau_mean_iou']:>9} "
            f"{row['target_reached'][:5]:>7} {row['target_steps']:>8} "
            f"{row['target_seconds']:>10} {row['best_iou']:>9}"
        )

    print("\nPer-pair means (plateau, over trials that plateaued):")
    print(f"{'pair':<18} {'method':<6} {'n':>3} {'steps':>10} {'seconds':>11} {'mean_iou':>9}")
    for pair in sorted({row["pair"] for row in rows}):
        for method in METHODS:
            group = [
                row for row in rows
                if row["pair"] == pair and row["method"] == method
                and row["plateau_steps"] not in ("-", "-1")
            ]
            if not group:
                continue
            steps = sum(float(r["plateau_steps"]) for r in group) / len(group)
            secs = sum(float(r["plateau_seconds"]) for r in group) / len(group)
            iou = sum(float(r["plateau_mean_iou"]) for r in group) / len(group)
            print(f"{pair:<18} {method:<6} {len(group):>3} {steps:>10.1f} {secs:>11.1f} {iou:>9.4f}")

    print("\nPer-pair means (steps/seconds to the mean_iou target, reached trials only):")
    print(f"{'pair':<18} {'method':<6} {'n':>3} {'steps':>10} {'seconds':>11}")
    for pair in sorted({row["pair"] for row in rows}):
        for method in METHODS:
            group = [
                row for row in rows
                if row["pair"] == pair and row["method"] == method
                and row["target_reached"] == "True"
            ]
            if not group:
                print(f"{pair:<18} {method:<6} {0:>3} {'never':>10} {'never':>11}")
                continue
            steps = sum(float(r["target_steps"]) for r in group) / len(group)
            secs = sum(float(r["target_seconds"]) for r in group) / len(group)
            print(f"{pair:<18} {method:<6} {len(group):>3} {steps:>10.1f} {secs:>11.1f}")


def main() -> None:
    args = _parse_args()
    sweep_name = args.sweep_name or f"convergence_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = _PROJECT_ROOT / "results" / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    print(
        f"[Sweep] {sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.methods)} method(s) x {args.trials} seed(s))"
    )

    submitted: list[dict] = []
    for job in jobs:
        script_path = _write_sbatch_script(job, args, sweep_dir)
        if args.dry_run:
            print(f"[Dry run] wrote {script_path}")
            submitted.append(job)
            continue
        job["job_id"] = _submit(script_path)
        print(f"[Submitted] {job['job_id']}: {job['name']}")
        submitted.append(job)

    manifest_path = _write_manifest(sweep_dir, submitted)
    print(f"[Sweep] manifest written to {manifest_path}")
    if not args.dry_run:
        print("Monitor with: squeue --me | grep conv_")
        print(f"Aggregate when done: python submit_convergence.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
