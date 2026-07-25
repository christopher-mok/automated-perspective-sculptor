"""Submit the area-normalized silhouette : negative-space weight sweep to SLURM.

The per-view loss is

    loss_view = rgb + silhouette_weight * silhouette
                    + negative_space_weight * negative_space

and until now the two terms were not on the same scale. ``negative_space`` is
a mean over the target *background*: total squared alpha where nothing should
be, divided by background area. ``silhouette`` was a mean over the *whole
image*, so a target filling a tenth of the frame produced a tenth of the loss a
frame-filling target would for the same relative miss. The effective ratio
between the two therefore drifted with the shape and size of each target, and a
single weight setting did not mean the same thing for sun/moon as for
teapot/droplets.

This batch runs with ``--area-normalized-view-loss``, which restricts the
silhouette term to the foreground and divides by foreground area, mirroring
what negative space already does with the background. Both terms then read as
mean squared alpha error per unit of their own region, in [0, 1], and the ratio
between them is a property of the experiment rather than of the images. That
also changes what the loss *numbers* mean, so this sweep's values are not
comparable to the pre-normalization batches -- only within itself.

The sweep varies only the ratio silhouette : negative_space, holding their sum
at --weight-sum (4.5) so the view loss keeps a constant magnitude relative to
the untouched rgb and overlap terms; otherwise a ratio change would be
confounded with an overall view-loss scale change. The grid is log-spaced and
symmetric about 1.0, which under normalization is the point where a unit of
missed silhouette and a unit of spilled negative space cost the same:

    0.25  0.5  1.0  2.0  4.0        (-> silhouette 0.9 .. 3.6)

2.0 is the ratio the opweights / countobj / pcompare batches ran at, so it is
the point of contact with them.

Six pairs x five arms x one seed = 30 jobs, all under ``results/areaweights/``.
Three pairs are one detailed shape against a simple primitive (sun/moon,
cat-face/bass, acm/scf) and three put two detailed silhouettes in competition
(teapot/droplets, water/fire, dance/argument), so the sweep can show whether
the best ratio depends on that.

Everything else is the fixed configuration shared with those batches: the full
SRD arm, fan-fill triangulation, swept-volume resolution 256, 100% of SRD
additions drawn from the swept volume, the planar overlap test with its hard
repair pass, the default 0.35/0.15/0.50 operation mix, and lambda_count = 0 so
nothing rations piece count -- it is one of the two tracked outcomes.

``run_final.py`` logs step, mean IoU and piece count at every evaluation into
``history.csv`` and exports both final rendered camera views per job; --collect
gathers every one of those views into ``collected/views/``. Runs stop when both
mean IoU and loss plateau, never before --min-steps (1000) and never past
--steps (4000) or --max-hours.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_areaweights.py --dry-run

    # Submit 6 pairs x 5 ratios x 1 seed = 30 jobs
    python submit_areaweights.py

    # After jobs finish: aggregate into areaweights/<name>/collected/
    python submit_areaweights.py --sweep-name <name> --collect
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from submit_ablations import IMAGE_PAIRS

_PROJECT_ROOT = Path(__file__).resolve().parent

AREAWEIGHTS_ROOT = _PROJECT_ROOT / "results" / "areaweights"

# One detailed shape against a simple primitive.
BASIC_PAIRS = (
    "sun_moon",
    "cat_face_bass",
    "acm_scf",
)

# Two detailed silhouettes, so both views compete for the same geometry.
HARD_PAIRS = (
    "teapot_droplets",
    "water_fire",
    "dance_argument",
)

ALL_PAIRS = BASIC_PAIRS + HARD_PAIRS

# silhouette_weight / negative_space_weight, log-spaced about 1.0. With both
# terms area-normalized, 1.0 is the point where missing a unit of silhouette
# and spilling a unit into negative space cost the same. 2.0 is what the
# opweights / countobj / pcompare batches ran at.
RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)

# Held fixed across the grid, as in every other batch.
DEFAULT_WEIGHT_SUM = 4.5

# The ratio shared with the other running batches, marked in the printout.
REFERENCE_RATIO = 2.0

BASE_ARM = "srd"

# Piece count is a measured outcome here, so nothing rations it.
LAMBDA_COUNT = 0.0

# Tversky = TP / (TP + alpha*FP + beta*FN) is the asymmetric generalization of
# IoU (which is alpha=beta=1): alpha prices spill -- render painted into
# negative space -- and beta prices miss -- target left unfilled. This sweep
# exists to buy low spill with silhouette weight, and plain IoU charges a
# saved-spill pixel and a lost-coverage pixel the same, so it cannot see that
# trade. The default weights spill 9:1 over miss, the point at which the
# sub-0.25 ratios overtake 0.25 on the live runs. IoU is reported unchanged
# alongside it; this is an added lens, not a replacement.
TVERSKY_ALPHA = 0.9
TVERSKY_BETA = 0.1

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


def _tag(value: float) -> str:
    """0.25 -> 'r0p25', for job and arm names."""
    return "r" + f"{value:g}".replace(".", "p")


ARM_RATIO = {_tag(ratio): ratio for ratio in RATIOS}
ARMS = tuple(ARM_RATIO)


def _weights(ratio: float, weight_sum: float) -> tuple[float, float]:
    """Split weight_sum into (silhouette, negative_space) at the given ratio."""
    negative_space = weight_sum / (1.0 + ratio)
    return weight_sum - negative_space, negative_space


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the area-normalized view-weight sweep to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory under results/areaweights/ (default: aw_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument(
        "--ratios",
        nargs="*",
        type=float,
        default=list(RATIOS),
        help="silhouette:negative-space ratios to sweep.",
    )
    parser.add_argument("--weight-sum", type=float, default=DEFAULT_WEIGHT_SUM)
    parser.add_argument(
        "--tversky-alpha", type=float, default=TVERSKY_ALPHA,
        help="Spill (false-positive) weight in the Tversky score (collect only).",
    )
    parser.add_argument(
        "--tversky-beta", type=float, default=TVERSKY_BETA,
        help="Miss (false-negative) weight in the Tversky score (collect only).",
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
    parser.add_argument("--srd-candidates", type=int, default=32)
    parser.add_argument("--lambda-count", type=float, default=LAMBDA_COUNT)
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
            arm = _tag(ratio)
            silhouette, negative_space = _weights(ratio, args.weight_sum)
            job_name = f"aw_{pair}_{arm}"
            output_dir = sweep_dir / job_name
            command = [
                args.python, "run_final.py",
                "--target1", str(image_dir / target1),
                "--target2", str(image_dir / target2),
                "--arm", BASE_ARM,
                "--srd-candidates", str(args.srd_candidates),
                "--lambda-count", f"{args.lambda_count:g}",
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
                "--area-normalized-view-loss",
                "--render-scale", str(args.render_scale),
                "--max-hours", str(args.max_hours),
                "--device", args.device,
                "--output-dir", str(output_dir),
            ]
            jobs.append({
                "name": job_name,
                "pair": pair,
                "arm": arm,
                "ratio": ratio,
                "silhouette": silhouette,
                "negative_space": negative_space,
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
    lines = [
        "job_id\tjob_name\tpair\tarm\tratio\tsilhouette_weight\t"
        "negative_space_weight\toutput_dir"
    ]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['arm']}\t"
            f"{row['ratio']:.6g}\t{row['silhouette']:.6g}\t"
            f"{row['negative_space']:.6g}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "silhouette_weight", "negative_space_weight", "weight_ratio",
    "area_normalized_view_loss", "lambda_count", "seed", "overlap_mode",
    "stop_reason", "final_step", "final_loss",
    # The two tracked outcomes: IoU ...
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "best_mean_iou", "best_mean_iou_step",
    # ... and piece count.
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_overlap",
    "srd_total_adds", "srd_total_splits", "srd_total_deletes",
    "total_seconds",
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


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tversky(coverage: float, spill: float, alpha: float, beta: float) -> float:
    """Asymmetric IoU from area-normalized coverage and spill.

    Both are normalized by target area, so TP/|T| = coverage, FP/|T| = spill and
    FN/|T| = 1 - coverage; dividing Tversky's numerator and denominator by |T|
    gives this closed form. alpha=beta=1 reproduces IoU exactly.
    """
    miss = 1.0 - coverage
    denominator = coverage + alpha * spill + beta * miss
    return coverage / denominator if denominator > 0 else 0.0


def _best_tversky_from_history(
    history: Path, alpha: float, beta: float
) -> float | None:
    """Peak per-step Tversky over a run, the analogue of best_mean_iou.

    Computed from the mean_coverage and mean_spill columns so it tracks the same
    per-view mean the report's final_mean_* values do. Returns None if the file
    is missing or carries neither column yet.
    """
    if not history.exists():
        return None
    import csv

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
    """Split 'aw_<pair>_<arm>' back into its parts.

    The arm is the trailing ``r<ratio>`` tag (``_tag`` output), matched by shape
    rather than against the module-default ``ARMS`` -- a sweep run with a
    ``--ratios`` override carries tags this file's default grid never lists, and
    keying off the constant would silently drop those rows.
    """
    match = re.match(r"^aw_(?P<pair>.+)_(?P<arm>r[0-9p]+)$", job_name)
    if match:
        return match.group("pair"), match.group("arm")
    return re.sub(r"^aw_", "", job_name), "-"


def _ratio_from_tag(tag: str) -> float:
    """Invert ``_tag``: 'r0p03125' -> 0.03125. NaN for an unparseable tag."""
    try:
        return float(tag[1:].replace("p", "."))
    except (TypeError, ValueError):
        return float("nan")


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
        # Tversky sits beside IoU: the final one from the report's mean
        # coverage/spill, the best one scanned over the run like best_mean_iou.
        coverage = _safe_float(row.get("final_mean_coverage", ""))
        spill = _safe_float(row.get("final_mean_spill", ""))
        row["final_mean_tversky"] = (
            f"{_tversky(coverage, spill, alpha, beta):.6g}"
            if coverage is not None and spill is not None else "-"
        )
        best_tversky = _best_tversky_from_history(
            sweep_dir / row["job"] / "history.csv", alpha, beta
        )
        row["best_mean_tversky"] = (
            f"{best_tversky:.6g}" if best_tversky is not None else "-"
        )
        rows.append(row)

    def _ratio_key(arm: str) -> float:
        ratio = _ratio_from_tag(arm)
        return 1e9 if ratio != ratio else ratio  # nan -> sort last

    group_order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    rows.sort(key=lambda r: (group_order.get(r["pair"], 99), _ratio_key(r["arm"])))

    # --- per-job summary (Tversky columns inserted next to the IoU block)
    summary_keys = list(_SUMMARY_KEYS)
    insert_at = summary_keys.index("best_mean_iou_step") + 1
    summary_keys[insert_at:insert_at] = ["final_mean_tversky", "best_mean_tversky"]
    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "arm"] + summary_keys
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- IoU and piece count over steps for every job, tagged by ratio
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
        ratio = _ratio_from_tag(row["arm"])
        prefix = f"{row['pair']},{row['arm']},{ratio:.6g}"
        if not curve_lines:
            curve_lines.append(f"pair,arm,weight_ratio,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{prefix},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # --- every final image, gathered and renamed by pair and ratio
    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_count = 0
    for row in rows:
        for image in sorted((sweep_dir / row["job"]).glob("*.png")):
            suffix = image.stem.split("_")[-1]
            shutil.copy2(image, views_dir / f"{row['pair']}_{row['arm']}_{suffix}.png")
            view_count += 1

    # --- headline: outcome against weight ratio, basic vs hard vs all
    mean_columns = [
        "final_mean_iou", "best_mean_iou",
        "final_mean_tversky", "best_mean_tversky",
        "final_view1_iou", "final_view2_iou",
        "final_patches", "max_patches",
        "final_mean_spill", "final_mean_coverage", "final_mean_precision",
        "final_loss", "final_step", "total_seconds",
    ]
    by_ratio_path = collected / "by_ratio.tsv"
    lines = [
        "group\tarm\tratio\tsilhouette_weight\tnegative_space_weight\tn\t"
        + "\t".join(mean_columns)
    ]
    arms_present = sorted({r["arm"] for r in rows}, key=_ratio_key)
    for label, members_of in (
        ("basic", set(BASIC_PAIRS)),
        ("hard", set(HARD_PAIRS)),
        ("all", set(ALL_PAIRS)),
    ):
        for arm in arms_present:
            members = [r for r in rows if r["arm"] == arm and r["pair"] in members_of]
            if not members:
                continue
            silhouette = _mean([m["silhouette_weight"] for m in members])
            negative_space = _mean([m["negative_space_weight"] for m in members])
            values = [_mean([m[column] for m in members]) for column in mean_columns]
            lines.append(
                f"{label}\t{arm}\t{_ratio_from_tag(arm):.6g}\t{silhouette}\t"
                f"{negative_space}\t{len(members)}\t" + "\t".join(values)
            )
    by_ratio_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- console view
    print(f"Tversky = TP / (TP + {alpha:g}*spill + {beta:g}*miss); "
          f"IoU is the same with both weights 1.")
    print(f"{'job':<40} {'pair':<16} {'arm':<7} {'ratio':>6} {'f_iou':>8} "
          f"{'f_tvsky':>8} {'f_pc':>5} {'max_pc':>7} {'spill':>8} {'cover':>8} "
          f"{'step':>6} {'stop':<10}")
    print("-" * 141)
    for row in rows:
        print(
            f"{row['job']:<40} {row['pair']:<16} {row['arm']:<7} "
            f"{_ratio_from_tag(row['arm']):>6.3g} "
            f"{row['final_mean_iou']:>8} {row['final_mean_tversky']:>8} "
            f"{row['final_patches']:>5} "
            f"{row['max_patches']:>7} {row['final_mean_spill']:>8} "
            f"{row['final_mean_coverage']:>8} {row['final_step']:>6} "
            f"{row['stop_reason']:<10}"
        )

    print("\nBy ratio -- IoU, Tversky and piece count against "
          "silhouette:negative-space:")
    print(f"{'group':<7} {'arm':<7} {'ratio':>6} {'sil':>6} {'neg':>6} {'n':>3} "
          f"{'f_iou':>8} {'f_tvsky':>8} {'f_pc':>6} {'max_pc':>7} "
          f"{'spill':>8} {'cover':>8}")
    for line in lines[1:]:
        fields = line.split("\t")
        group, arm, ratio, silhouette, negative_space, n = fields[:6]
        values = dict(zip(mean_columns, fields[6:]))
        print(
            f"{group:<7} {arm:<7} {ratio:>6} {silhouette:>6} {negative_space:>6} "
            f"{n:>3} {values['final_mean_iou']:>8} {values['final_mean_tversky']:>8} "
            f"{values['final_patches']:>6} "
            f"{values['max_patches']:>7} {values['final_mean_spill']:>8} "
            f"{values['final_mean_coverage']:>8}"
        )

    # --- where the optimum lands under each lens: best ratio per pair
    print("\nBest ratio per pair -- by IoU vs by Tversky "
          f"(spill:miss = {alpha:g}:{beta:g}):")
    print(f"{'pair':<16} {'best_iou_ratio':>15} {'best_tversky_ratio':>19}")
    seen_pairs = [p for p in ALL_PAIRS if any(r["pair"] == p for r in rows)]
    seen_pairs += [p for p in dict.fromkeys(r["pair"] for r in rows)
                   if p not in seen_pairs]
    for pair in seen_pairs:
        members = [r for r in rows if r["pair"] == pair]

        def _best(members: list[dict], key: str) -> str:
            scored = [(m, _safe_float(m.get(key, ""))) for m in members]
            scored = [(m, v) for m, v in scored if v is not None]
            if not scored:
                return "-"
            winner = max(scored, key=lambda mv: mv[1])[0]
            return f"{_ratio_from_tag(winner['arm']):g}"

        print(f"{pair:<16} {_best(members, 'best_mean_iou'):>15} "
              f"{_best(members, 'best_mean_tversky'):>19}")

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are "
              f"absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_ratio_path):
        print(f"  {path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"aw_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = AREAWEIGHTS_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir, args.tversky_alpha, args.tversky_beta)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    print(
        f"[Sweep] areaweights/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.ratios)} ratio(s) x 1 seed)"
    )
    print("[Sweep] area-normalized view loss: silhouette / foreground area, "
          "negative space / background area")
    for ratio in args.ratios:
        silhouette, negative_space = _weights(ratio, args.weight_sum)
        marker = "  <- shared with opweights/countobj/pcompare" if (
            abs(ratio - REFERENCE_RATIO) < 1e-9
        ) else ""
        print(
            f"[Sweep]   {_tag(ratio):<7} ratio={ratio:<6g} "
            f"silhouette={silhouette:.4g} negative_space={negative_space:.4g}"
            f"{marker}"
        )
    print(
        f"[Sweep] base arm={BASE_ARM}, lambda_count={args.lambda_count:g}, "
        f"weight sum held at {args.weight_sum:g}, "
        f"steps {args.min_steps}..{args.steps} with early stop"
    )
    print(f"[Sweep] pairs: basic={', '.join(p for p in args.pairs if p in BASIC_PAIRS)} | "
          f"hard={', '.join(p for p in args.pairs if p in HARD_PAIRS)}")

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
        print("Monitor with: squeue --me | grep aw_")
        print(f"Aggregate when done: python submit_areaweights.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
