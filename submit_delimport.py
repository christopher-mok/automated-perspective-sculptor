"""Submit the damage-based deletion-importance A/B to SLURM.

SRD normally offers deletion candidates *uniformly*: every eligible piece is
equally likely to be put up for removal, and the compatibility + lookahead pass
then decides which offers survive. This batch tests biasing that *offer* step by
a cheap per-piece damage proxy -- ``area * spill_fraction``, the piece's own
silhouette rendered in isolation, weighted by how much of it lands in negative
space -- softmax-sampled at a temperature (``run_final.py
--deletion-importance --deletion-temperature``). The weighting is soft, never an
argmax: argmax collapses into greedy pruning and kills the stochastic diversity
that lets SRD reconsider structure, so a temperatured softmax keeps the
"mostly-good, occasionally-exploratory" character. The proxy is per-piece and
ignores that damage is coupled across overlapping pieces -- the same blind spot
the greedy compatibility + lookahead already has, so it is no worse.

Four arms, one seed each, on the seven standard pairs:

    srd_uniform   Baseline. Full SRD, no count constraint, uniform deletion.
                  Unconstrained theta. This is the thing importance sampling is
                  measured against.

    srd_importance  Full SRD, no count constraint, deletion importance on.
                    Unconstrained theta. The A/B partner of srd_uniform: only
                    the deletion offer distribution differs.

    hinge2_importance  Hinge2 (the count objective at its best setting -- the
                       fit-then-shed FULL_STACK schedule at count_lambda 500)
                       with deletion importance on. Unconstrained theta. Tests
                       whether the proxy helps the arm that actively sheds.

    srd_importance_theta  Secondary baseline. Full SRD, deletion importance on,
                          but *constrained* theta (the default 15-degree camera
                          margin) rather than free rotation. Isolates how much
                          of importance sampling's effect survives when the
                          orientation is not free to relieve spill by rotating.

Shared configuration, held across all arms so the deletion rule and theta
constraint are the only moving parts:

  * area weights at ratio 0.03125 (silhouette : negative_space = 0.1364 : 4.364,
    sum 4.5) with --area-normalized-view-loss. This is the strongly
    spill-averse end the aw_low sweep pointed at, so deletion importance is
    tested where spill matters most.
  * --min-steps 1500 with --early-stop, so no arm can stop before the fit has
    had room to settle; hard ceiling --steps 4000.
  * --render-every so every run leaves its geometry over time under renders/.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_delimport.py --dry-run

    # 4 arms x 7 pairs x 1 seed = 28 jobs
    python submit_delimport.py

    # Just the core A/B
    python submit_delimport.py --arms srd_uniform srd_importance

    # After jobs finish
    python submit_delimport.py --sweep-name <name> --collect
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from submit_ablations import IMAGE_PAIRS
from submit_hinge2 import ALL_PAIRS, FULL_STACK, WEIGHT_SUM

_PROJECT_ROOT = Path(__file__).resolve().parent
DELIMPORT_ROOT = _PROJECT_ROOT / "results" / "delimport"

# Strongly spill-averse area weights (silhouette : negative_space), the low end
# the aw_low sweep pointed at. Held fixed across arms.
RATIO = 0.03125

# Hinge2's count objective runs at this lambda -- the arm that produced the real
# sheds in the h2 batch. 1/500 = 0.002 IoU is what a piece may cost.
HINGE2_LAMBDA = "500"
HINGE2_COUNT_TARGET_IOU = 0.91

# arm -> the switches that distinguish it. base_arm is run_final's --arm;
# count_objective adds the fit-then-shed stack; deletion_importance turns on the
# proxy sampler; unconstrained_theta frees the orientation (default is the
# 15-degree camera-margin constraint).
ARM_SETTINGS: dict[str, dict] = {
    "srd_uniform": {
        "base_arm": "srd", "count_objective": False,
        "deletion_importance": False, "unconstrained_theta": True,
    },
    "srd_importance": {
        "base_arm": "srd", "count_objective": False,
        "deletion_importance": True, "unconstrained_theta": True,
    },
    "hinge2_importance": {
        "base_arm": "srd", "count_objective": True,
        "deletion_importance": True, "unconstrained_theta": True,
    },
    "srd_importance_theta": {
        "base_arm": "srd", "count_objective": False,
        "deletion_importance": True, "unconstrained_theta": False,
    },
}
ARMS = tuple(ARM_SETTINGS)

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


def _hinge2_flags(count_lambda: str) -> list[str]:
    """FULL_STACK from submit_hinge2, at this batch's lambda, as CLI flags."""
    settings = {**FULL_STACK, "count_lambda": count_lambda}
    flags: list[str] = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        else:
            flags.extend([flag, str(value)])
    return flags


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the deletion-importance A/B to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name", default=None,
        help="Batch directory under results/delimport/ (default: di_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument(
        "--arms", nargs="*", default=None, choices=sorted(ARMS),
        help="Override the arm list (default: all four).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single trial seed per config.")
    parser.add_argument(
        "--deletion-temperature", type=float, default=1.0,
        help="Softmax temperature for the deletion proxy (std units).",
    )
    parser.add_argument(
        "--deletion-proxy", choices=("spill", "net"), default="spill",
        help="Damage proxy for the importance arms: 'spill' = area x "
             "spill_fraction; 'net' = area x (spill_fraction - coverage_fraction).",
    )
    parser.add_argument("--weight-sum", type=float, default=WEIGHT_SUM)
    parser.add_argument("--count-lambda", default=HINGE2_LAMBDA)
    parser.add_argument("--count-target-iou", type=float, default=HINGE2_COUNT_TARGET_IOU)

    parser.add_argument("--steps", type=int, default=4000, help="Hard step ceiling.")
    parser.add_argument("--min-steps", type=int, default=1500)
    parser.add_argument("--patience-steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--render-every", type=int, default=25)
    parser.add_argument("--render-every-scale", type=int, default=1)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--swept-resolution", type=int, default=256)
    parser.add_argument("--swept-spawn-fraction", type=float, default=1.0)
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
        help="Shell run before the command, inside the sbatch script.",
    )

    mode = parser.add_argument_group("actions")
    mode.add_argument("--dry-run", action="store_true", help="Write scripts, submit nothing.")
    mode.add_argument("--collect", action="store_true", help="Aggregate results and exit.")
    return parser.parse_args()


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    silhouette, negative_space = _weights(RATIO, args.weight_sum)

    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        for arm in args.arms:
            settings = ARM_SETTINGS[arm]
            job_name = f"di_{pair}_{arm}"
            output_dir = sweep_dir / job_name
            command = [
                args.python, "run_final.py",
                "--target1", str(image_dir / target1),
                "--target2", str(image_dir / target2),
                "--arm", settings["base_arm"],
                "--overlap-mode", "planar",
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
                "--silhouette-weight", f"{silhouette:g}",
                "--negative-space-weight", f"{negative_space:g}",
                "--area-normalized-view-loss",
                "--render-scale", str(args.render_scale),
                "--max-hours", str(args.max_hours),
                "--device", args.device,
                "--output-dir", str(output_dir),
            ]
            if settings["deletion_importance"]:
                command += [
                    "--deletion-importance",
                    "--deletion-temperature", f"{args.deletion_temperature:g}",
                    "--deletion-proxy", args.deletion_proxy,
                ]
            if settings["unconstrained_theta"]:
                command += ["--unconstrained-theta"]
            if settings["count_objective"]:
                command += [
                    "--count-objective",
                    "--count-target-iou", f"{args.count_target_iou:g}",
                    *_hinge2_flags(args.count_lambda),
                ]
            jobs.append({
                "name": job_name, "pair": pair, "arm": arm,
                "silhouette": silhouette, "negative_space": negative_space,
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
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=True,
    )
    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _write_manifest(sweep_dir: Path, rows: list[dict]) -> Path:
    manifest_path = sweep_dir / "manifest.tsv"
    lines = ["job_id\tjob_name\tpair\tarm\tsilhouette_weight\tnegative_space_weight\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['arm']}\t"
            f"{row['silhouette']:.6g}\t{row['negative_space']:.6g}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "stop_reason", "final_step", "final_loss",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "best_mean_iou", "best_mean_iou_step",
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_overlap", "srd_total_adds", "srd_total_splits", "srd_total_deletes",
    "total_seconds",
)

# spill:miss weighting for the Tversky lens, matching submit_areaweights' default.
TVERSKY_ALPHA = 0.9
TVERSKY_BETA = 0.1


def _grab(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "-"


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[str]) -> str:
    numbers = [n for n in (_safe_float(v) for v in values) if n is not None and n >= 0]
    return f"{sum(numbers) / len(numbers):.6g}" if numbers else "-"


def _tversky(coverage: float, spill: float, alpha: float, beta: float) -> float:
    """Asymmetric IoU from area-normalized coverage and spill; alpha=beta=1 is IoU."""
    denominator = coverage + alpha * spill + beta * (1.0 - coverage)
    return coverage / denominator if denominator > 0 else 0.0


def _best_tversky_from_history(history: Path, alpha: float, beta: float) -> float | None:
    if not history.exists():
        return None
    best: float | None = None
    with history.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            coverage = _safe_float(record.get("mean_coverage", ""))
            spill = _safe_float(record.get("mean_spill", ""))
            if coverage is None or spill is None:
                continue
            value = _tversky(coverage, spill, alpha, beta)
            if best is None or value > best:
                best = value
    return best


def _pair_and_arm(job_name: str) -> tuple[str, str]:
    stripped = re.sub(r"^di_", "", job_name)
    for arm in sorted(ARMS, key=len, reverse=True):
        if stripped.endswith(f"_{arm}"):
            return stripped[: -len(arm) - 1], arm
    return stripped, "-"


def _collect(sweep_dir: Path, alpha: float, beta: float) -> None:
    reports = sorted(sweep_dir.glob("*/report.txt"))
    if not reports:
        print(f"No report.txt files found under {sweep_dir}")
        return

    collected = sweep_dir / "collected"
    collected.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for report in reports:
        text = report.read_text(encoding="utf-8")
        pair, arm = _pair_and_arm(report.parent.name)
        row = {"job": report.parent.name, "pair": pair, "arm": arm}
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        coverage = _safe_float(row.get("final_mean_coverage", ""))
        spill = _safe_float(row.get("final_mean_spill", ""))
        row["final_mean_tversky"] = (
            f"{_tversky(coverage, spill, alpha, beta):.6g}"
            if coverage is not None and spill is not None else "-"
        )
        best = _best_tversky_from_history(sweep_dir / row["job"] / "history.csv", alpha, beta)
        row["best_mean_tversky"] = f"{best:.6g}" if best is not None else "-"
        rows.append(row)

    pair_order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    rows.sort(key=lambda r: (pair_order.get(r["pair"], 99), arm_order.get(r["arm"], 99)))

    # --- per-job summary (Tversky beside the IoU block)
    summary_keys = list(_SUMMARY_KEYS)
    insert_at = summary_keys.index("best_mean_iou_step") + 1
    summary_keys[insert_at:insert_at] = ["final_mean_tversky", "best_mean_tversky"]
    columns = ["job", "pair", "arm"] + summary_keys
    summary_path = collected / "summary.tsv"
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- IoU / spill / piece curves over steps, tagged by arm
    curves_path = collected / "curves.csv"
    curve_lines: list[str] = []
    missing_curves = 0
    for row in rows:
        history = sweep_dir / row["job"] / "history.csv"
        body = history.read_text(encoding="utf-8").splitlines() if history.exists() else []
        if not body:
            missing_curves += 1
            continue
        prefix = f"{row['pair']},{row['arm']}"
        if not curve_lines:
            curve_lines.append(f"pair,arm,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{prefix},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # --- final views, renamed by pair and arm
    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_count = 0
    for row in rows:
        for image in sorted((sweep_dir / row["job"]).glob("*.png")):
            suffix = image.stem.split("_")[-1]
            shutil.copy2(image, views_dir / f"{row['pair']}_{row['arm']}_{suffix}.png")
            view_count += 1

    # --- headline: per-arm means, the A/B read
    mean_columns = [
        "final_mean_iou", "best_mean_iou", "final_mean_tversky", "best_mean_tversky",
        "final_patches", "max_patches", "final_mean_spill", "final_mean_coverage",
        "srd_total_deletes", "final_step", "total_seconds",
    ]
    by_arm_path = collected / "by_arm.tsv"
    by_arm_lines = ["arm\tn\t" + "\t".join(mean_columns)]
    for arm in ARMS:
        members = [r for r in rows if r["arm"] == arm]
        if not members:
            continue
        values = [_mean([m[column] for m in members]) for column in mean_columns]
        by_arm_lines.append(f"{arm}\t{len(members)}\t" + "\t".join(values))
    by_arm_path.write_text("\n".join(by_arm_lines) + "\n", encoding="utf-8")

    # --- console
    print(f"Tversky = TP / (TP + {alpha:g}*spill + {beta:g}*miss); "
          f"IoU is the same with both weights 1.")
    print(f"{'job':<34} {'pair':<16} {'arm':<20} {'f_iou':>8} {'f_tvsky':>8} "
          f"{'f_pc':>5} {'dels':>5} {'spill':>8} {'step':>6} {'stop':<10}")
    print("-" * 132)
    for row in rows:
        print(
            f"{row['job']:<34} {row['pair']:<16} {row['arm']:<20} "
            f"{row['final_mean_iou']:>8} {row['final_mean_tversky']:>8} "
            f"{row['final_patches']:>5} {row['srd_total_deletes']:>5} "
            f"{row['final_mean_spill']:>8} {row['final_step']:>6} {row['stop_reason']:<10}"
        )

    print("\nBy arm -- the A/B (means over pairs):")
    print(f"{'arm':<22} {'n':>3} {'f_iou':>8} {'best_iou':>9} {'f_tvsky':>8} "
          f"{'best_tv':>8} {'f_pc':>6} {'dels':>6} {'spill':>8}")
    for line in by_arm_lines[1:]:
        fields = line.split("\t")
        arm, n = fields[0], fields[1]
        values = dict(zip(mean_columns, fields[2:]))
        print(
            f"{arm:<22} {n:>3} {values['final_mean_iou']:>8} {values['best_mean_iou']:>9} "
            f"{values['final_mean_tversky']:>8} {values['best_mean_tversky']:>8} "
            f"{values['final_patches']:>6} {values['srd_total_deletes']:>6} "
            f"{values['final_mean_spill']:>8}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_arm_path):
        print(f"  {path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    if args.arms is None:
        args.arms = list(ARMS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"di_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = DELIMPORT_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir, TVERSKY_ALPHA, TVERSKY_BETA)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    silhouette, negative_space = _weights(RATIO, args.weight_sum)
    print(
        f"[Sweep] delimport/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.arms)} arm(s) x 1 seed)"
    )
    print(f"[Sweep] area ratio {RATIO:g} -> silhouette={silhouette:.4g} "
          f"negative_space={negative_space:.4g}, area-normalized; "
          f"min-steps {args.min_steps}, deletion proxy {args.deletion_proxy}, "
          f"temperature {args.deletion_temperature:g}")
    for arm in args.arms:
        s = ARM_SETTINGS[arm]
        print(f"[Sweep]   {arm:<22} count={'hinge2' if s['count_objective'] else 'none':<6} "
              f"deletion={'importance' if s['deletion_importance'] else 'uniform':<10} "
              f"theta={'free' if s['unconstrained_theta'] else 'constrained'}")

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
        print("Monitor with: squeue --me | grep di_")
        print(f"Aggregate when done: python submit_delimport.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
