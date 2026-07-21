"""Submit the hard-pair comparison/ablation batch to SLURM.

Everything lives under ``results/hard_experiments/``. One sweep, three arms:

    basic          the base approach, SRD disabled entirely
    srd            the full method
    srd_no_swept   SRD with swept-volume-guided additions disabled

The comparison reads basic vs srd; the ablation reads srd vs srd_no_swept.
They share the srd arm, so both questions come out of a single sweep.

The pairs are harder than the earlier batches': both views are detailed
silhouettes rather than one shape against a primitive, so the two views
genuinely compete for the same geometry.

Every job runs with fan-fill triangulation, swept-volume resolution 256, 100%
of SRD additions drawn from the swept volume, the planar overlap test with its
hard repair pass, and a silhouette:negative-space weight ratio of 2 at a fixed
sum of 4.5 (-> 3.0 / 1.5).

Runs stop when both mean IoU and loss have plateaued, but never before
--min-steps (1000), and never past --steps (4000) or --max-hours. Walltime is
24h with the internal cap an hour below it, so a run that is still improving
at step 4000 is limited by the step ceiling rather than by the clock.

Metrics tracked at every evaluation, in ``history.csv``:

    loss, mean/per-view IoU
    mean/per-view spill, precision, coverage  (the negative-space breakdown)
    patches                                    (piece count over time)
    overlap, overlap_repaired_pairs

IoU is a true IoU, so it *already* charges for covering area that should be
empty -- that area lands in the union. ``spill`` (= |render & ~target| /
|target|) is logged next to it not as a competing score but to say which side
of the IoU a run is losing: unfilled target, or paint outside the silhouette.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_hard.py --dry-run

    # Submit 5 pairs x 3 arms x 3 seeds = 45 jobs
    python submit_hard.py

    # After jobs finish: aggregate into hard_experiments/<name>/collected/
    python submit_hard.py --sweep-name <name> --collect
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

HARD_ROOT = _PROJECT_ROOT / "results" / "hard_experiments"

HARD_PAIRS = (
    "siggraph_sigchi",
    "crane_crab",
    "teapot_droplets",
    "dancer_guitar",
    "dance_argument",
)

ARMS = ("basic", "srd", "srd_no_swept")

# silhouette : negative_space = 2, holding their sum at 4.5 so the view loss
# keeps the same magnitude relative to the untouched rgb and overlap terms.
WEIGHT_RATIO = 2.0
WEIGHT_SUM = 4.5

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


def _weights(ratio: float, weight_sum: float) -> tuple[float, float]:
    """Split weight_sum into (silhouette, negative_space) at the given ratio."""
    negative_space = weight_sum / (1.0 + ratio)
    return weight_sum - negative_space, negative_space


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the hard-pair comparison/ablation batch to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory under results/hard_experiments/ (default: hard_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument("--arms", nargs="*", default=list(ARMS), help="Override the arm list.")
    parser.add_argument("--trials", type=int, default=3, help="Seeds per configuration.")
    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Hard ceiling; runs normally stop earlier on convergence.",
    )
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--patience-steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--swept-resolution", type=int, default=256)
    parser.add_argument("--swept-spawn-fraction", type=float, default=1.0)
    parser.add_argument("--weight-ratio", type=float, default=WEIGHT_RATIO)
    parser.add_argument("--weight-sum", type=float, default=WEIGHT_SUM)
    parser.add_argument("--max-hours", type=float, default=23.0)
    parser.add_argument("--device", default="cuda")

    cluster = parser.add_argument_group("cluster resources")
    cluster.add_argument("--partition", default="3090-gcondo")
    cluster.add_argument("--gres", default="gpu:1")
    cluster.add_argument("--time", default="24:00:00", help="SLURM walltime (> --max-hours).")
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
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)

    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        for arm in args.arms:
            for seed in range(args.trials):
                job_name = f"hard_{pair}_{arm}_seed{seed}"
                output_dir = sweep_dir / job_name
                command = [
                    args.python, "run_final.py",
                    "--target1", str(image_dir / target1),
                    "--target2", str(image_dir / target2),
                    "--arm", arm,
                    "--overlap-mode", "planar",
                    "--seed", str(seed),
                    "--steps", str(args.steps),
                    "--early-stop",
                    "--min-steps", str(args.min_steps),
                    "--patience-steps", str(args.patience_steps),
                    "--eval-interval", str(args.eval_interval),
                    "--n-patches", str(args.n_patches),
                    "--swept-resolution", str(args.swept_resolution),
                    "--swept-spawn-fraction", str(args.swept_spawn_fraction),
                    "--silhouette-weight", f"{silhouette:g}",
                    "--negative-space-weight", f"{negative_space:g}",
                    "--max-hours", str(args.max_hours),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ]
                jobs.append({
                    "name": job_name,
                    "pair": pair,
                    "arm": arm,
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
    lines = ["job_id\tjob_name\tpair\tarm\tseed\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t"
            f"{row['arm']}\t{row['seed']}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "arm", "seed", "overlap_mode", "overlap_repair", "stop_reason",
    "final_step", "final_loss",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "final_mean_spill", "final_view1_spill", "final_view2_spill",
    "final_mean_precision", "final_mean_coverage",
    "best_mean_spill",
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_overlap",
    "best_mean_iou", "best_mean_iou_step", "best_loss", "best_loss_step",
    "srd_total_adds", "srd_total_deletes", "total_seconds",
)

_MILESTONE_KEYS = tuple(
    f"iou_{t}_{unit}"
    for t in ("0p50", "0p60", "0p70", "0p75", "0p80", "0p85", "0p90")
    for unit in ("steps", "seconds")
) + tuple(
    f"rel_{p}pct_{unit}"
    for p in ("90", "95", "99")
    for unit in ("steps", "seconds")
)

# Job names are "hard_<pair>_<arm>_seed<n>", and both pair names and arms
# contain underscores, so split on the known arm set rather than on position.
_JOB_ARMS = tuple(sorted(ARMS, key=len, reverse=True))


def _split_job_name(job_name: str) -> tuple[str, str]:
    """Recover (pair, arm) from a job directory name."""
    stripped = re.sub(r"_seed\d+$", "", re.sub(r"^hard_", "", job_name))
    for arm in _JOB_ARMS:
        if stripped.endswith(f"_{arm}"):
            return stripped[: -(len(arm) + 1)], arm
    return stripped, "-"


def _grab(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "-"


def _mean(values: list[str]) -> str:
    """Mean of the parseable, non-sentinel values; '-' when there are none."""
    numbers = []
    for value in values:
        try:
            number = float(value)
        except ValueError:
            continue
        if number < 0:  # -1 marks a milestone the run never reached
            continue
        numbers.append(number)
    if not numbers:
        return "-"
    return f"{sum(numbers) / len(numbers):.6g}"


def _reached(values: list[str]) -> str:
    """How many of the runs in a group reached the milestone at all."""
    hit = 0
    for value in values:
        try:
            if float(value) >= 0:
                hit += 1
        except ValueError:
            pass
    return f"{hit}/{len(values)}"


def _collect(sweep_dir: Path) -> None:
    reports = sorted(sweep_dir.glob("*/report.txt"))
    if not reports:
        print(f"No report.txt files found under {sweep_dir}")
        return

    collected = sweep_dir / "collected"
    collected.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for report in reports:
        text = report.read_text(encoding="utf-8")
        row = {"job": report.parent.name}
        for key in _SUMMARY_KEYS + _MILESTONE_KEYS:
            row[key] = _grab(text, key)
        row["pair"], row["arm_tag"] = _split_job_name(report.parent.name)
        rows.append(row)

    # --- per-trial summary
    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "arm_tag"] + list(_SUMMARY_KEYS) + list(_MILESTONE_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- every metric over steps, every trial, one long-format table
    curves_path = collected / "curves.csv"
    curve_lines: list[str] = []
    missing_curves = 0
    for row in rows:
        history = sweep_dir / row["job"] / "history.csv"
        if not history.exists():
            missing_curves += 1
            continue
        body = history.read_text(encoding="utf-8").splitlines()
        if not body:
            missing_curves += 1
            continue
        if not curve_lines:  # header, taken from the runner so it stays in sync
            curve_lines.append(f"pair,arm,seed,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(
                    f"{row['pair']},{row['arm_tag']},{row['seed']},{line}"
                )
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # --- per (pair, arm) means
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["pair"], row["arm_tag"]), []).append(row)

    means_path = collected / "means_by_group.tsv"
    mean_columns = [
        "final_loss", "final_mean_iou", "best_mean_iou",
        "final_mean_spill", "final_mean_precision", "final_mean_coverage",
        "best_loss", "final_patches", "max_patches", "mean_patches",
        "final_overlap", "srd_total_adds", "srd_total_deletes",
        "final_step", "total_seconds",
    ]
    lines = ["pair\tarm\tn\t" + "\t".join(mean_columns)]
    for (pair, arm), members in sorted(groups.items()):
        values = [_mean([m[column] for m in members]) for column in mean_columns]
        lines.append(f"{pair}\t{arm}\t{len(members)}\t" + "\t".join(values))
    means_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- convergence: mean steps/seconds to each milestone, plus how many runs
    #     in the group ever got there. This is the ablation's headline table.
    convergence_path = collected / "convergence.tsv"
    lines = ["pair\tarm\tn\tmilestone\tmean_steps\tmean_seconds\treached"]
    milestones = [key[:-6] for key in _MILESTONE_KEYS if key.endswith("_steps")]
    for (pair, arm), members in sorted(groups.items()):
        for milestone in milestones:
            steps = [m[f"{milestone}_steps"] for m in members]
            seconds = [m[f"{milestone}_seconds"] for m in members]
            lines.append(
                f"{pair}\t{arm}\t{len(members)}\t{milestone}\t"
                f"{_mean(steps)}\t{_mean(seconds)}\t{_reached(steps)}"
            )
    convergence_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- where each run stopped and how many pieces it landed with
    pieces_path = collected / "pieces.tsv"
    lines = ["pair\tarm\tn\tfinal_patches\tmax_patches\tmean_patches\t"
             "srd_total_adds\tsrd_total_deletes\tstop_reasons"]
    for (pair, arm), members in sorted(groups.items()):
        reasons: dict[str, int] = {}
        for member in members:
            reasons[member["stop_reason"]] = reasons.get(member["stop_reason"], 0) + 1
        reason_text = ",".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        lines.append(
            f"{pair}\t{arm}\t{len(members)}\t"
            f"{_mean([m['final_patches'] for m in members])}\t"
            f"{_mean([m['max_patches'] for m in members])}\t"
            f"{_mean([m['mean_patches'] for m in members])}\t"
            f"{_mean([m['srd_total_adds'] for m in members])}\t"
            f"{_mean([m['srd_total_deletes'] for m in members])}\t{reason_text}"
        )
    pieces_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- console view
    print(f"{'job':<46} {'arm':<14} {'stop':<10} {'f_loss':>10} "
          f"{'f_iou':>8} {'spill':>8} {'pieces':>7} {'step':>6} {'secs':>9}")
    print("-" * 126)
    for row in rows:
        print(
            f"{row['job']:<46} {row['arm_tag']:<14} {row['stop_reason']:<10} "
            f"{row['final_loss']:>10} {row['final_mean_iou']:>8} "
            f"{row['final_mean_spill']:>8} {row['final_patches']:>7} "
            f"{row['final_step']:>6} {row['total_seconds']:>9}"
        )

    print(f"\nPer-group means ({len(groups)} groups):")
    print(f"{'pair':<20} {'arm':<14} {'n':>3} {'final_loss':>11} "
          f"{'final_iou':>10} {'best_iou':>9} {'spill':>8} {'pieces':>8} {'steps':>8}")
    for (pair, arm), members in sorted(groups.items()):
        print(
            f"{pair:<20} {arm:<14} {len(members):>3} "
            f"{_mean([m['final_loss'] for m in members]):>11} "
            f"{_mean([m['final_mean_iou'] for m in members]):>10} "
            f"{_mean([m['best_mean_iou'] for m in members]):>9} "
            f"{_mean([m['final_mean_spill'] for m in members]):>8} "
            f"{_mean([m['final_patches'] for m in members]):>8} "
            f"{_mean([m['final_step'] for m in members]):>8}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are "
              f"absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, means_path, convergence_path, pieces_path):
        print(f"  {path.name}")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(HARD_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")
    unknown_arms = [arm for arm in args.arms if arm not in ARMS]
    if unknown_arms:
        raise SystemExit(f"Unknown arm(s): {', '.join(unknown_arms)}")

    sweep_name = args.sweep_name or f"hard_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = HARD_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)
    print(
        f"[Sweep] hard_experiments/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.arms)} arm(s) x {args.trials} seed(s))"
    )
    print(
        f"[Sweep] weights: silhouette={silhouette:g}, "
        f"negative_space={negative_space:g} (ratio {args.weight_ratio:g}), "
        f"steps {args.min_steps}..{args.steps} with early stop"
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
        print("Monitor with: squeue --me | grep hard_")
        print(f"Aggregate when done: python submit_hard.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
