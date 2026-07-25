"""Submit the final comparison, ablation and overlap-test batches to SLURM.

Everything lives under ``results/final/``. Three experiments share one runner
(``run_final.py``) and one collector:

    main      The comparison and the ablation in a single sweep, because they
              share their srd arm. Arms: basic (no SRD), srd (full method) and
              srd_no_swept (SRD without swept-volume-guided additions). The
              comparison reads basic vs srd; the ablation reads srd vs
              srd_no_swept, which is the only component the final batch
              ablates.

    overlap   Does the new overlap test actually help? Arm is held at srd and
              only --overlap-mode varies, planar (new: clip both outlines
              against the plane intersection line) vs sphere (old: bounding
              spheres that ignore theta). Three pairs x three seeds.

Every job uses fan-fill triangulation, swept-volume resolution 256, all SRD
additions drawn from the swept volume, and a silhouette:negative-space weight
ratio of 4 at a fixed sum of 4.5. The main and ablation batches also use the
new planar overlap test; only the overlap experiment varies it.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_final.py --experiment main --dry-run

    # Submit 5 pairs x 3 arms x 3 seeds = 45 jobs
    python submit_final.py --experiment main

    # Submit 3 pairs x 2 overlap modes x 3 seeds = 18 jobs
    python submit_final.py --experiment overlap

    # After jobs finish: aggregate into results/final/<name>/collected/
    python submit_final.py --experiment main --sweep-name <name> --collect
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

# Everything this module writes lives here.
FINAL_ROOT = _PROJECT_ROOT / "results" / "final"

MAIN_PAIRS = ("sun_moon", "horse_circle", "cat_face_bass", "water_fire", "acm_scf")
# The overlap A/B only needs to show a consistent direction, so it reuses the
# three pairs the earlier fan-fill and weight sweeps ran on.
OVERLAP_PAIRS = ("sun_moon", "horse_circle", "cat_face_bass")

MAIN_ARMS = ("basic", "srd", "srd_no_swept")
OVERLAP_MODES = ("planar", "sphere")

# silhouette : negative_space = 4, holding their sum at 4.5 so the view loss
# keeps the same magnitude relative to the untouched rgb and overlap terms.
WEIGHT_RATIO = 4.0
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


def _parse_n_patches_map(spec: str | None) -> dict[str, int]:
    """Parse 'pair=N,pair=N' into {pair: N}, for per-pair starting counts.

    Used when the piece count is not a property of the sweep but of the pair --
    e.g. seeding a no-SRD baseline at the count an SRD run converged to, which
    differs per pair.
    """
    if not spec:
        return {}
    mapping: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--n-patches-map entry {item!r} is not pair=N")
        pair, _, count = item.partition("=")
        pair, count = pair.strip(), count.strip()
        if pair not in IMAGE_PAIRS:
            raise SystemExit(f"--n-patches-map: unknown pair {pair!r}")
        if not count.isdigit() or int(count) < 1:
            raise SystemExit(f"--n-patches-map: {pair} count {count!r} must be a positive integer")
        mapping[pair] = int(count)
    return mapping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the final comparison/ablation/overlap batches to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=("main", "overlap"),
        default="main",
        help="'main' = comparison + swept-volume ablation; "
             "'overlap' = new vs old overlap test.",
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory under results/final/ (default: <experiment>_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument("--arms", nargs="*", default=None, choices=sorted(MAIN_ARMS),
                        help="Override the arm list for the main experiment "
                             "(ignored by --experiment overlap, which varies "
                             "overlap mode at a fixed srd arm).")
    parser.add_argument("--trials", type=int, default=3, help="Seeds per configuration.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--n-patches-map", default=None,
                        help="Per-pair starting piece counts as 'pair=N,pair=N', "
                             "overriding --n-patches for those pairs only. For "
                             "seeding a baseline at counts that differ per pair.")
    parser.add_argument("--swept-resolution", type=int, default=256)
    parser.add_argument("--swept-spawn-fraction", type=float, default=1.0)
    parser.add_argument("--weight-ratio", type=float, default=WEIGHT_RATIO)
    parser.add_argument("--weight-sum", type=float, default=WEIGHT_SUM)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")

    frames = parser.add_argument_group("frame export")
    frames.add_argument("--render-every", type=int, default=0,
                        help="Export both camera views every N steps into "
                             "renders/, giving a frame series for a video. "
                             "0 (default) keeps the historical behaviour of "
                             "exporting only the final views.")
    frames.add_argument("--render-every-scale", type=int, default=1,
                        help="Resolution multiplier for the --render-every "
                             "snapshots, separate from --render-scale because "
                             "these are written hundreds of times per run.")

    stop = parser.add_argument_group("early stopping")
    stop.add_argument("--early-stop", action="store_true",
                      help="Stop once mean IoU and loss have both plateaued, "
                           "never before --min-steps. Off by default so "
                           "existing batches are unchanged.")
    stop.add_argument("--min-steps", type=int, default=1000,
                      help="Never early-stop before this step.")
    stop.add_argument("--patience-steps", type=int, default=300,
                      help="Steps without gain before the plateau stop fires.")

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


def _configurations(args: argparse.Namespace) -> list[dict]:
    """(arm, overlap_mode, tag) triples this experiment varies over."""
    if args.experiment == "overlap":
        return [
            {"arm": "srd", "overlap_mode": mode, "tag": mode}
            for mode in OVERLAP_MODES
        ]
    return [
        {"arm": arm, "overlap_mode": "planar", "tag": arm}
        for arm in (args.arms or MAIN_ARMS)
    ]


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)
    prefix = "final" if args.experiment == "main" else "ovl"

    patch_map = _parse_n_patches_map(args.n_patches_map)

    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        n_patches = patch_map.get(pair, args.n_patches)
        for config in _configurations(args):
            for seed in range(args.trials):
                job_name = f"{prefix}_{pair}_{config['tag']}_seed{seed}"
                output_dir = sweep_dir / job_name
                command = [
                    args.python, "run_final.py",
                    "--target1", str(image_dir / target1),
                    "--target2", str(image_dir / target2),
                    "--arm", config["arm"],
                    "--overlap-mode", config["overlap_mode"],
                    "--seed", str(seed),
                    "--steps", str(args.steps),
                    "--eval-interval", str(args.eval_interval),
                    "--n-patches", str(n_patches),
                    "--swept-resolution", str(args.swept_resolution),
                    "--swept-spawn-fraction", str(args.swept_spawn_fraction),
                    "--silhouette-weight", f"{silhouette:g}",
                    "--negative-space-weight", f"{negative_space:g}",
                    "--max-hours", str(args.max_hours),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ]
                if args.render_every:
                    command.extend([
                        "--render-every", str(args.render_every),
                        "--render-every-scale", str(args.render_every_scale),
                    ])
                if args.early_stop:
                    command.extend([
                        "--early-stop",
                        "--min-steps", str(args.min_steps),
                        "--patience-steps", str(args.patience_steps),
                    ])
                # Repair is the planar test's hard constraint; the sphere arm
                # of the overlap A/B has no exact test to repair against, so it
                # runs soft-penalty-only exactly as the old method did.
                if config["overlap_mode"] != "planar":
                    command.append("--no-overlap-repair")
                jobs.append({
                    "name": job_name,
                    "pair": pair,
                    "arm": config["arm"],
                    "overlap_mode": config["overlap_mode"],
                    "tag": config["tag"],
                    "seed": seed,
                    "n_patches": n_patches,
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
    lines = ["job_id\tjob_name\tpair\tarm\toverlap_mode\tseed\tn_patches\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['arm']}\t"
            f"{row['overlap_mode']}\t{row['seed']}\t{row.get('n_patches', '-')}\t"
            f"{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "arm", "seed", "overlap_mode", "overlap_repair", "stop_reason",
    "final_step", "final_loss", "final_mean_iou", "final_view1_iou",
    "final_view2_iou", "final_patches", "final_overlap",
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


# Job names are "<prefix>_<pair>_<tag>_seed<n>", and both pair names and tags
# contain underscores, so split on the known tag set rather than on position.
_JOB_TAGS = tuple(sorted(set(MAIN_ARMS) | set(OVERLAP_MODES), key=len, reverse=True))


def _split_job_name(job_name: str) -> tuple[str, str]:
    """Recover (pair, group) from a job directory name."""
    stripped = re.sub(r"_seed\d+$", "", re.sub(r"^(final|ovl)_", "", job_name))
    for tag in _JOB_TAGS:
        if stripped.endswith(f"_{tag}"):
            return stripped[: -(len(tag) + 1)], tag
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
        row["pair"], row["group"] = _split_job_name(report.parent.name)
        rows.append(row)

    # --- per-trial summary
    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "group"] + list(_SUMMARY_KEYS) + list(_MILESTONE_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- loss/IoU over time, every trial in one long-format table
    curves_path = collected / "curves.csv"
    curve_lines = [
        "pair,group,seed,step,seconds,loss,mean_iou,view1_iou,view2_iou,"
        "patches,overlap,overlap_repaired_pairs"
    ]
    missing_curves = 0
    for row in rows:
        history = sweep_dir / row["job"] / "history.csv"
        if not history.exists():
            missing_curves += 1
            continue
        body = history.read_text(encoding="utf-8").splitlines()[1:]
        for line in body:
            if line.strip():
                curve_lines.append(
                    f"{row['pair']},{row['group']},{row['seed']},{line}"
                )
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # --- per (pair, group) means
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["pair"], row["group"]), []).append(row)

    means_path = collected / "means_by_group.tsv"
    mean_columns = [
        "final_loss", "final_mean_iou", "best_mean_iou", "best_loss",
        "final_patches", "final_overlap", "srd_total_adds", "total_seconds",
    ]
    lines = ["pair\tgroup\tn\t" + "\t".join(mean_columns)]
    for (pair, group), members in sorted(groups.items()):
        values = [_mean([m[column] for m in members]) for column in mean_columns]
        lines.append(f"{pair}\t{group}\t{len(members)}\t" + "\t".join(values))
    means_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- convergence: mean steps/seconds to each milestone, plus how many runs
    #     in the group ever got there. This is the ablation's headline table.
    convergence_path = collected / "convergence.tsv"
    lines = ["pair\tgroup\tn\tmilestone\tmean_steps\tmean_seconds\treached"]
    milestones = [key[:-6] for key in _MILESTONE_KEYS if key.endswith("_steps")]
    for (pair, group), members in sorted(groups.items()):
        for milestone in milestones:
            steps = [m[f"{milestone}_steps"] for m in members]
            seconds = [m[f"{milestone}_seconds"] for m in members]
            lines.append(
                f"{pair}\t{group}\t{len(members)}\t{milestone}\t"
                f"{_mean(steps)}\t{_mean(seconds)}\t{_reached(steps)}"
            )
    convergence_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- console view
    print(f"{'job':<44} {'group':<14} {'stop':<10} {'f_loss':>10} "
          f"{'f_iou':>8} {'b_iou':>8} {'ovl':>10} {'secs':>9}")
    print("-" * 118)
    for row in rows:
        print(
            f"{row['job']:<44} {row['group']:<14} {row['stop_reason']:<10} "
            f"{row['final_loss']:>10} {row['final_mean_iou']:>8} "
            f"{row['best_mean_iou']:>8} {row['final_overlap']:>10} "
            f"{row['total_seconds']:>9}"
        )

    print(f"\nPer-group means ({len(groups)} groups):")
    print(f"{'pair':<18} {'group':<14} {'n':>3} {'final_loss':>11} "
          f"{'final_iou':>10} {'best_iou':>9} {'seconds':>10}")
    for (pair, group), members in sorted(groups.items()):
        print(
            f"{pair:<18} {group:<14} {len(members):>3} "
            f"{_mean([m['final_loss'] for m in members]):>11} "
            f"{_mean([m['final_mean_iou'] for m in members]):>10} "
            f"{_mean([m['best_mean_iou'] for m in members]):>9} "
            f"{_mean([m['total_seconds'] for m in members]):>10}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are "
              f"absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, means_path, convergence_path):
        print(f"  {path.name}")


def main() -> None:
    args = _parse_args()

    default_pairs = MAIN_PAIRS if args.experiment == "main" else OVERLAP_PAIRS
    if args.pairs is None:
        args.pairs = list(default_pairs)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = (
        args.sweep_name
        or f"{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    sweep_dir = FINAL_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    configs = _configurations(args)
    print(
        f"[Sweep] final/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(configs)} arm(s) x {args.trials} seed(s))"
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
        prefix = "final_" if args.experiment == "main" else "ovl_"
        print(f"Monitor with: squeue --me | grep {prefix}")
        print(f"Aggregate when done: python submit_final.py "
              f"--experiment {args.experiment} --sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
