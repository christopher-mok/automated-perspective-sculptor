"""Submit the piece-count-weight sweep to SLURM.

One question: as the piece-count penalty ``lambda_count`` grows, how do final
IoU and final piece count trade off? SRD now optimizes
``loss + lambda_count * n_patches`` (see ``optimizer/srd.py``), so a larger
lambda makes every SRD addition pay for itself in loss before it is accepted.
We want the lambda that keeps IoU high while holding the piece count down.

Everything lives under ``results/piece_count/``. The sweep is the five basic
pairs x five orders of magnitude of lambda x one seed = 25 jobs, all on the
full ``srd`` arm (the only arm whose piece count moves). Lambda spans
1e-4 .. 1e0: at the low end the penalty is below a typical accepted SRD
improvement (~1e-4..3e-2) and barely bites; at the high end it suppresses
essentially every addition.

Every job carries the same fixed configuration as the hard batch: fan-fill
triangulation, swept-volume resolution 256, 100% of SRD additions drawn from
the swept volume, the planar overlap test with its hard repair pass, and a
silhouette:negative-space weight ratio of 2 at a fixed sum of 4.5 (-> 3.0/1.5).
Runs stop when both mean IoU and loss plateau, never before --min-steps (1000)
and never past --steps (4000) or --max-hours.

``run_final.py`` tracks IoU and piece count at every evaluation in
``history.csv`` and exports the two final rendered views per job, named
``<t1>-<t2>_srd_lam<value>_seed<n>_view{1,2}.png`` so they organize cleanly.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_piececount.py --dry-run

    # Submit 5 pairs x 5 lambdas x 1 seed = 25 jobs
    python submit_piececount.py

    # After jobs finish: aggregate into piece_count/<name>/collected/
    python submit_piececount.py --sweep-name <name> --collect
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

PIECE_COUNT_ROOT = _PROJECT_ROOT / "results" / "piece_count"

# The five basic pairs: one detailed shape against a simple primitive.
BASIC_PAIRS = (
    "sun_moon",
    "horse_circle",
    "cat_face_bass",
    "water_fire",
    "acm_scf",
)

# Five orders of magnitude, centred on the old SRD default (0.05). Accepted SRD
# improvements cluster in ~1e-4..3e-2, so this spans "no effect" to "suppresses
# every addition".
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1e0)

ARM = "srd"

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


def _lambda_tag(value: float) -> str:
    """Filesystem- and sort-friendly tag for a lambda value (0.01 -> lam0p01)."""
    return f"lam{value:g}".replace("-", "m").replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the piece-count-weight sweep to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory under results/piece_count/ (default: pc_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument(
        "--lambdas",
        nargs="*",
        type=float,
        default=list(LAMBDAS),
        help="Piece-count weights to sweep.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single trial seed per config.")
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
    parser.add_argument("--render-scale", type=int, default=4)
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
        for lam in args.lambdas:
            job_name = f"pc_{pair}_{_lambda_tag(lam)}"
            output_dir = sweep_dir / job_name
            command = [
                args.python, "run_final.py",
                "--target1", str(image_dir / target1),
                "--target2", str(image_dir / target2),
                "--arm", ARM,
                "--lambda-count", f"{lam:g}",
                "--overlap-mode", "planar",
                "--seed", str(args.seed),
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
                "--render-scale", str(args.render_scale),
                "--max-hours", str(args.max_hours),
                "--device", args.device,
                "--output-dir", str(output_dir),
            ]
            jobs.append({
                "name": job_name,
                "pair": pair,
                "lambda": lam,
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
    lines = ["job_id\tjob_name\tpair\tlambda\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t"
            f"{row['lambda']:g}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "lambda_count", "seed", "overlap_mode", "stop_reason",
    "final_step", "final_loss",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "best_mean_iou", "best_mean_iou_step",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_overlap", "srd_total_adds", "srd_total_deletes", "total_seconds",
)


def _grab(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "-"


def _mean(values: list[str]) -> str:
    numbers = []
    for value in values:
        try:
            number = float(value)
        except ValueError:
            continue
        if number < 0:
            continue
        numbers.append(number)
    if not numbers:
        return "-"
    return f"{sum(numbers) / len(numbers):.6g}"


def _lambda_key(text: str) -> float:
    """Numeric lambda for sorting; +inf when unparseable so it sinks to the end."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("inf")


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
        row = {"job": report.parent.name, "pair": _grab(text, "target1")}
        # Recover the pair name from the job dir (pc_<pair>_lam<tag>).
        stripped = re.sub(r"^pc_", "", re.sub(r"_lam[^_]+$", "", report.parent.name))
        row["pair"] = stripped
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        rows.append(row)

    rows.sort(key=lambda r: (r["pair"], _lambda_key(r["lambda_count"])))

    # --- per-job summary
    summary_path = collected / "summary.tsv"
    columns = ["job", "pair"] + list(_SUMMARY_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- every metric over steps, every job, one long-format table
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
        if not curve_lines:
            curve_lines.append(f"pair,lambda_count,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{row['pair']},{row['lambda_count']},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # --- headline: IoU vs piece count per lambda, averaged over pairs
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["lambda_count"], []).append(row)

    by_lambda_path = collected / "by_lambda.tsv"
    mean_columns = [
        "final_mean_iou", "best_mean_iou", "final_patches", "max_patches",
        "mean_patches", "final_loss", "final_mean_spill", "srd_total_adds",
        "srd_total_deletes", "final_step", "total_seconds",
    ]
    lines = ["lambda_count\tn\t" + "\t".join(mean_columns)]
    for lam in sorted(groups, key=_lambda_key):
        members = groups[lam]
        values = [_mean([m[column] for m in members]) for column in mean_columns]
        lines.append(f"{lam}\t{len(members)}\t" + "\t".join(values))
    by_lambda_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- console view
    print(f"{'job':<34} {'pair':<16} {'lambda':>8} {'f_iou':>7} {'best_iou':>8} "
          f"{'pieces':>7} {'max_pc':>7} {'f_loss':>9} {'step':>6} {'stop':<10}")
    print("-" * 122)
    for row in rows:
        print(
            f"{row['job']:<34} {row['pair']:<16} {row['lambda_count']:>8} "
            f"{row['final_mean_iou']:>7} {row['best_mean_iou']:>8} "
            f"{row['final_patches']:>7} {row['max_patches']:>7} "
            f"{row['final_loss']:>9} {row['final_step']:>6} {row['stop_reason']:<10}"
        )

    print(f"\nBy lambda (mean over {len(BASIC_PAIRS)} pairs) -- the IoU/piece tradeoff:")
    print(f"{'lambda':>10} {'n':>3} {'final_iou':>10} {'best_iou':>9} "
          f"{'pieces':>8} {'max_pc':>8} {'loss':>9} {'adds':>7} {'dels':>7}")
    for lam in sorted(groups, key=_lambda_key):
        members = groups[lam]
        print(
            f"{lam:>10} {len(members):>3} "
            f"{_mean([m['final_mean_iou'] for m in members]):>10} "
            f"{_mean([m['best_mean_iou'] for m in members]):>9} "
            f"{_mean([m['final_patches'] for m in members]):>8} "
            f"{_mean([m['max_patches'] for m in members]):>8} "
            f"{_mean([m['final_loss'] for m in members]):>9} "
            f"{_mean([m['srd_total_adds'] for m in members]):>7} "
            f"{_mean([m['srd_total_deletes'] for m in members]):>7}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are "
              f"absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_lambda_path):
        print(f"  {path.name}")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(BASIC_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"pc_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = PIECE_COUNT_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)
    print(
        f"[Sweep] piece_count/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.lambdas)} lambda(s) x 1 seed)"
    )
    print(
        f"[Sweep] arm={ARM}, lambdas={', '.join(f'{lam:g}' for lam in args.lambdas)}, "
        f"weights: silhouette={silhouette:g}, negative_space={negative_space:g} "
        f"(ratio {args.weight_ratio:g}), steps {args.min_steps}..{args.steps} with early stop"
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
        print("Monitor with: squeue --me | grep pc_")
        print(f"Aggregate when done: python submit_piececount.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
