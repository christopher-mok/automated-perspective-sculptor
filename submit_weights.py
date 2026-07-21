"""Sweep the view-loss weight ratio on the base (no-SRD) approach.

The per-view loss is

    loss_view = rgb + silhouette_weight * silhouette
                    + negative_space_weight * negative_space

with the same two weights applied to view 1 and view 2. This sweep varies only
the *ratio* silhouette : negative_space, holding their sum at --weight-sum so
the view loss keeps a constant magnitude relative to the untouched rgb and
overlap terms — otherwise a ratio change would be confounded with an overall
view-loss scale change. overlap_weight is left at its default throughout.

Runs use the base approach (SRD disabled), which is faster, and stop at
plateau (compare_convergence.py's loss-plateau criterion) rather than a fixed
step budget. Each job writes iou_history.csv, so loss and IoU over steps are
recorded for every trial.

Sweeps live under results/weights/<sweep-name>/ and --collect writes its
aggregates into that same folder.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_weights.py --dry-run

    # Submit 3 pairs x 6 ratios x 3 seeds = 54 jobs
    python submit_weights.py

    # After jobs finish: aggregate into results/weights/<name>/
    python submit_weights.py --sweep-name <name> --collect
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

# All weight sweeps land here, separate from the convergence/fanfill sweeps.
_WEIGHTS_ROOT = _PROJECT_ROOT / "results" / "weights"

# The pairs this experiment covers.
PAIRS = ("sun_moon", "horse_circle", "cat_face_bass")

# silhouette_weight / negative_space_weight. The default pair (2.0, 2.5) is a
# ratio of 0.8, so the grid brackets it by ~half an order of magnitude either
# way: negative-space dominant through silhouette dominant.
RATIOS = (0.25, 0.5, 0.8, 1.0, 2.0, 4.0)

# Sum held fixed across the grid; 2.0 + 2.5 = the current defaults.
DEFAULT_WEIGHT_SUM = 4.5

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


def _weights_for_ratio(ratio: float, weight_sum: float) -> tuple[float, float]:
    """Split weight_sum into (silhouette, negative_space) at the given ratio."""
    negative_space = weight_sum / (1.0 + ratio)
    silhouette = weight_sum - negative_space
    return silhouette, negative_space


def _ratio_tag(ratio: float) -> str:
    """Filesystem-safe job-name fragment for a ratio (0.25 -> r0p25)."""
    return "r" + f"{ratio:g}".replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit view-loss weight-ratio sweep jobs (base approach) to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory name under results/weights/ (default: sweep_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", choices=sorted(PAIRS), default=sorted(PAIRS))
    parser.add_argument(
        "--ratios",
        nargs="*",
        type=float,
        default=list(RATIOS),
        help="silhouette_weight / negative_space_weight values to test.",
    )
    parser.add_argument(
        "--weight-sum",
        type=float,
        default=DEFAULT_WEIGHT_SUM,
        help="silhouette + negative_space, held constant across ratios.",
    )
    parser.add_argument("--trials", type=int, default=3, help="Seeds per (pair, ratio).")
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--swept-resolution", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--plateau-delta", type=float, default=0.002)
    parser.add_argument("--plateau-window", type=int, default=250)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--target-iou", type=float, default=0.85)
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
        for ratio in args.ratios:
            silhouette, negative_space = _weights_for_ratio(ratio, args.weight_sum)
            for seed in range(args.trials):
                job_name = f"w_{pair}_{_ratio_tag(ratio)}_seed{seed}"
                output_dir = sweep_dir / job_name
                command = [
                    args.python, "compare_convergence.py",
                    "--target1", str(image_dir / target1),
                    "--target2", str(image_dir / target2),
                    "--method", "base",
                    "--seed", str(seed),
                    "--silhouette-weight", f"{silhouette:.6g}",
                    "--negative-space-weight", f"{negative_space:.6g}",
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
                    "ratio": ratio,
                    "silhouette": silhouette,
                    "negative_space": negative_space,
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
    lines = ["job_id\tjob_name\tpair\tratio\tsilhouette\tnegative_space\tseed\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['ratio']:g}\t"
            f"{row['silhouette']:.6g}\t{row['negative_space']:.6g}\t{row['seed']}\t"
            f"{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def _collect(sweep_dir: Path) -> None:
    """Aggregate the sweep into results/weights/<sweep>/{trials,ratios,curves}."""
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
        job = report.parent.name
        # w_<pair>_r<ratio>_seed<n>
        match = re.match(r"^w_(?P<pair>.+)_r(?P<ratio>[0-9p]+)_seed(?P<seed>\d+)$", job)
        rows.append({
            "job": job,
            "pair": match.group("pair") if match else "-",
            "seed": match.group("seed") if match else "-",
            "ratio": _grab(text, "weight_ratio"),
            "silhouette": _grab(text, "silhouette_weight"),
            "negative_space": _grab(text, "negative_space_weight"),
            "stop_reason": _grab(text, "stop_reason"),
            "plateau_steps": _grab(text, "plateau_steps"),
            "plateau_seconds": _grab(text, "plateau_seconds"),
            "plateau_mean_iou": _grab(text, "plateau_mean_iou"),
            "final_mean_iou": _grab(text, "final_mean_iou"),
            "final_loss": _grab(text, "final_loss"),
            "best_iou": _grab(text, "best_mean_iou"),
        })

    columns = [
        "job", "pair", "ratio", "silhouette", "negative_space", "seed",
        "stop_reason", "plateau_steps", "plateau_seconds", "plateau_mean_iou",
        "final_mean_iou", "final_loss", "best_iou",
    ]
    trials_path = sweep_dir / "trials.tsv"
    trials_path.write_text(
        "\n".join(
            ["\t".join(columns)]
            + ["\t".join(str(row[column]) for column in columns) for row in rows]
        ) + "\n",
        encoding="utf-8",
    )

    def _mean(group: list[dict], key: str) -> float | None:
        values = [float(r[key]) for r in group if r[key] not in ("-", "-1")]
        return sum(values) / len(values) if values else None

    def _fmt(value: float | None, spec: str = "10.4f") -> str:
        return format(value, spec) if value is not None else f"{'-':>{spec.split('.')[0]}}"

    summary_lines = ["pair\tratio\tsilhouette\tnegative_space\tn\tbest_iou\tfinal_iou\tplateau_steps\tplateau_seconds"]
    print(f"{'pair':<16} {'ratio':>7} {'sil':>6} {'neg':>6} {'n':>3} "
          f"{'best_iou':>10} {'final_iou':>10} {'plat_steps':>11} {'plat_secs':>10}")
    for pair in sorted({row["pair"] for row in rows}):
        pair_rows = [row for row in rows if row["pair"] == pair]
        for ratio in sorted({r["ratio"] for r in pair_rows}, key=lambda value: float(value)):
            group = [row for row in pair_rows if row["ratio"] == ratio]
            best = _mean(group, "best_iou")
            final = _mean(group, "final_mean_iou")
            steps = _mean(group, "plateau_steps")
            secs = _mean(group, "plateau_seconds")
            print(
                f"{pair:<16} {ratio:>7} {group[0]['silhouette']:>6} "
                f"{group[0]['negative_space']:>6} {len(group):>3} "
                f"{_fmt(best)} {_fmt(final)} {_fmt(steps, '11.1f')} {_fmt(secs, '10.1f')}"
            )
            summary_lines.append(
                f"{pair}\t{ratio}\t{group[0]['silhouette']}\t{group[0]['negative_space']}\t"
                f"{len(group)}\t{best}\t{final}\t{steps}\t{secs}"
            )
    summary_path = sweep_dir / "ratio_summary.tsv"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # One long-format table of every trial's loss/IoU trajectory, so the
    # over-steps curves can be plotted without walking the job folders.
    curve_lines = ["pair\tratio\tseed\tstep\tseconds\tmean_iou\tview1_iou\tview2_iou\tloss\tpatches"]
    for row in rows:
        history = sweep_dir / row["job"] / "iou_history.csv"
        if not history.exists():
            continue
        for line in history.read_text(encoding="utf-8").splitlines()[1:]:
            if line.strip():
                fields = line.split(",")
                curve_lines.append(
                    f"{row['pair']}\t{row['ratio']}\t{row['seed']}\t" + "\t".join(fields)
                )
    curves_path = sweep_dir / "curves.tsv"
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    print(f"\nWrote:\n  {trials_path}\n  {summary_path}\n  {curves_path}")
    print(f"  ({len(curve_lines) - 1} curve samples: loss and IoU over steps)")


def main() -> None:
    args = _parse_args()
    sweep_name = args.sweep_name or f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = _WEIGHTS_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    print(
        f"[Sweep] weights/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.ratios)} ratio(s) x {args.trials} seed(s))"
    )
    print(f"{'ratio':>7} {'silhouette':>11} {'negative_space':>15}")
    for ratio in args.ratios:
        silhouette, negative_space = _weights_for_ratio(ratio, args.weight_sum)
        print(f"{ratio:>7g} {silhouette:>11.4g} {negative_space:>15.4g}")

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
        print("Monitor with: squeue --me | grep ' w_'")
        print(f"Aggregate when done: python submit_weights.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()


