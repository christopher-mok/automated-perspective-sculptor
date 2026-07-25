"""Submit the negative-space weighting batch to SLURM.

Two mechanisms -- full SRD (uniform deletion, no count constraint) and hinge2
(the fit-then-shed count objective) -- crossed with three ways of pricing
negative space, on the seven standard pairs. No importance deletion anywhere;
theta stays constrained (the default 15-degree camera margin) for every job.
The runs render frames over time and are allowed to run until the curve flattens
rather than being cut short.

Three loss configurations (the "thirds"):

    aw0p125    Area-normalized view loss at silhouette : negative_space = 0.125,
               i.e. silhouette 0.5, negative_space 4.0 (sum 4.5). Strongly
               negative-space weighted, in the normalized regime the aw sweeps
               used.

    nonorm4    The original *non*-area-normalized view loss, negative space
               weighted 4x silhouette: negative_space 3.6, silhouette 0.9. These
               are the same absolute weights as area-normalized ratio 0.25, but
               without normalization, so the pair isolates what normalization
               itself does. Silhouette here is averaged over the whole image
               rather than the foreground, which further tilts the effective
               loss toward negative space.

    aw0p1875   Area-normalized view loss at ratio 0.1875, i.e. silhouette
               0.711, negative_space 3.789. Sits between aw0p125 and the 0.25
               the aw sweep selected.

Two arms per configuration:

    srd        Full SRD, --arm srd, no count objective, uniform deletion.
    hinge2     The count objective at its best setting (FULL_STACK fit-then-shed
               at count_lambda 500, count_target_iou 0.91), uniform deletion.

7 pairs x 2 arms x 3 configs = 42 jobs, all under results/negspace/.

Examples
--------
    python submit_negspace.py --dry-run
    python submit_negspace.py
    python submit_negspace.py --sweep-name <name> --collect
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
NEGSPACE_ROOT = _PROJECT_ROOT / "results" / "negspace"

HINGE2_LAMBDA = "500"
HINGE2_COUNT_TARGET_IOU = 0.91


def _weights(ratio: float, weight_sum: float) -> tuple[float, float]:
    """Split weight_sum into (silhouette, negative_space) at silhouette:neg = ratio."""
    negative_space = weight_sum / (1.0 + ratio)
    return weight_sum - negative_space, negative_space


# The three loss configurations. silhouette / negative_space are absolute view
# weights; area_normalized toggles --area-normalized-view-loss. label is for the
# printout only.
def _aw(ratio: float) -> tuple[float, float]:
    return _weights(ratio, WEIGHT_SUM)


CONFIGS: dict[str, dict] = {
    "aw0p125": {
        "silhouette": _aw(0.125)[0], "negative_space": _aw(0.125)[1],
        "area_normalized": True, "label": "area-norm sil:neg 0.125",
    },
    "nonorm4": {
        # negative space weighted 4x silhouette, no normalization.
        "silhouette": 0.9, "negative_space": 3.6,
        "area_normalized": False, "label": "non-normalized neg:sil 4:1",
    },
    "aw0p1875": {
        "silhouette": _aw(0.1875)[0], "negative_space": _aw(0.1875)[1],
        "area_normalized": True, "label": "area-norm sil:neg 0.1875",
    },
    "aw0p25": {
        # The ratio the area-weight sweep selected. Same absolute weights as
        # nonorm4 (0.9 / 3.6), but normalized, so the pair isolates
        # normalization itself.
        "silhouette": _aw(0.25)[0], "negative_space": _aw(0.25)[1],
        "area_normalized": True, "label": "area-norm sil:neg 0.25",
    },
    "aw0p073": {
        # The area-normalized weights that reproduce what the working
        # non-normalized 2.0 / 2.5 actually does, derived rather than guessed.
        #
        # Non-normalized silhouette_loss averages (alpha - mask)^2 over the
        # whole frame, so in the background it is alpha^2 -- the term already
        # carries a spill penalty. Splitting it by foreground fraction f:
        #
        #     L_sil_nonnorm = f * L_sil_norm + (1 - f) * L_negspace
        #
        # so 2.0*L_sil_nn + 2.5*L_ns is identically
        #     (2.0*f) * L_sil_norm + (2.0*(1-f) + 2.5) * L_ns,
        # i.e. normalized weights (2f, 4.5 - 2f), which sum to 4.5 for free.
        #
        # f measured through the real loss path (10% transparent border,
        # fit_image_to_resolution into the 192x256 canvas, alpha as mask) is
        # 0.1263 for water_fire and 0.1797 for sun_moon -- the two pairs that
        # look best on the non-normalized weights. Their combined f = 0.1530
        # gives silhouette 0.306, negative_space 4.194, ratio 0.0730.
        #
        # Every earlier aw config (0.125, 0.1875, 0.25) sits well above that,
        # so this is the first normalized run on the negative-space side of the
        # ratio the working weights imply.
        "silhouette": 0.306, "negative_space": 4.194,
        "area_normalized": True, "label": "area-norm sil:neg 0.073 (matches nonorm 2.0/2.5)",
    },
    "nonorm1p25": {
        # The SceneOptimizer class defaults, which every run before the
        # negative-space batches used implicitly: negative space only 1.25x
        # silhouette, no normalization. Here as the pre-negspace baseline.
        "silhouette": 2.0, "negative_space": 2.5,
        "area_normalized": False, "label": "non-normalized neg:sil 1.25:1",
    },
}
CONFIG_TAGS = tuple(CONFIGS)

# arm -> the switch that distinguishes it. srd/hinge2 use uniform deletion (no
# importance); importnet biases the deletion *offer* by the net damage proxy.
# Theta stays constrained (no --unconstrained-theta) for every arm here.
ARM_SETTINGS: dict[str, dict] = {
    "srd": {"base_arm": "srd", "count_objective": False},
    "hinge2": {"base_arm": "srd", "count_objective": True},
    "importnet": {
        "base_arm": "srd", "count_objective": False,
        # Full SRD with the net deletion-importance proxy -- area x
        # (spill_fraction - coverage_fraction), softmax-sampled at temperature 1.
        # The A/B partner of the "srd" arm: only the deletion offer distribution
        # differs (net importance vs uniform), theta constrained for both.
        "deletion_importance": True, "deletion_proxy": "net",
        "deletion_temperature": 1.0,
    },
    "srd_restart": {
        "base_arm": "srd", "count_objective": False,
        # Full SRD with conflict-gated restart: a deletion candidate that helps
        # one view but hurts the other is deleted and respawned into the residual
        # it vacated (swept-volume seeded); pieces bad in both views stay plain
        # deletes. A/B partner of the "srd" (uniform delete) arm.
        "conflict_restart": True,
    },
    "hinge2_restart": {
        "base_arm": "srd", "count_objective": True,
        # The fit-then-shed count objective with conflict-gated restart layered
        # on: shedding still removes spare pieces, but conflicting pieces are
        # rebuilt in place rather than dropped. A/B partner of the "hinge2" arm.
        "conflict_restart": True,
    },
    "noswept": {
        # Swept-volume ablation: --arm srd_no_swept sets
        # disable_swept_volume_adds, so SRD additions are placed without the
        # swept-volume guidance rather than sampled from it. The swept volume is
        # still built and still seeds initialization -- run_final passes it to
        # initialize_patches unconditionally -- so this ablates the *additions*
        # only. A/B partner of the "srd" arm.
        "base_arm": "srd_no_swept", "count_objective": False,
    },
    "noswept_restart": {
        # The same swept-volume-additions ablation with conflict-gated restart
        # allowed, so it pairs against "srd_restart" the way "noswept" pairs
        # against "srd".
        "base_arm": "srd_no_swept", "count_objective": False,
        "conflict_restart": True,
    },
    "hinge2_restart_importnet": {
        "base_arm": "srd", "count_objective": True,
        # Everything at once: fit-then-shed counting, conflict-gated restart,
        # and the net deletion-importance proxy biasing which piece gets offered
        # for deletion. The restart and importance arms were only ever measured
        # separately on top of plain SRD; this is the first run of both together
        # under the count objective.
        "conflict_restart": True,
        "deletion_importance": True, "deletion_proxy": "net",
        "deletion_temperature": 1.0,
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
        description="Submit the negative-space weighting batch to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-name", default=None,
                        help="Batch dir under results/negspace/ (default: neg_<timestamp>).")
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument("--arms", nargs="*", default=None, choices=sorted(ARMS),
                        help="Override the arm list.")
    parser.add_argument("--configs", nargs="*", default=None, choices=sorted(CONFIG_TAGS),
                        help="Override the loss-config list.")
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--srd-candidates", type=int, default=32,
                        help="Per-pass SRD candidate budget, split by the "
                             "0.35/0.15/0.50 add/delete/split weights.")
    parser.add_argument("--srd-min-patch-area", type=float, default=0.01,
                        help="SRD's tiny-area deletion threshold. New pieces "
                             "spawn at area 0.009291, so the historical 0.01 "
                             "culls every add at the next rewrite step; drop "
                             "this below the spawn area to let adds survive.")
    parser.add_argument("--lambda-count", type=float, default=0.05,
                        help="Per-piece penalty in SRD's acceptance test: an add "
                             "must beat the loss by more than this to be accepted, "
                             "and a delete is rebated it. 0 scores rewrites on the "
                             "raw loss change, which is what lets adds land.")
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
    mode.add_argument("--collect", action="store_true")
    return parser.parse_args()


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        for config_tag in args.configs:
            cfg = CONFIGS[config_tag]
            for arm in args.arms:
                settings = ARM_SETTINGS[arm]
                job_name = f"neg_{pair}_{arm}_{config_tag}"
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
                    "--srd-candidates", str(args.srd_candidates),
                    "--srd-min-patch-area", f"{args.srd_min_patch_area:g}",
                    "--lambda-count", f"{args.lambda_count:g}",
                    "--silhouette-weight", f"{cfg['silhouette']:g}",
                    "--negative-space-weight", f"{cfg['negative_space']:g}",
                    "--render-scale", str(args.render_scale),
                    "--max-hours", str(args.max_hours),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ]
                if cfg["area_normalized"]:
                    command += ["--area-normalized-view-loss"]
                if settings["count_objective"]:
                    command += [
                        "--count-objective",
                        "--count-target-iou", f"{args.count_target_iou:g}",
                        *_hinge2_flags(args.count_lambda),
                    ]
                if settings.get("deletion_importance"):
                    command += [
                        "--deletion-importance",
                        "--deletion-temperature", f"{settings['deletion_temperature']:g}",
                        "--deletion-proxy", settings["deletion_proxy"],
                    ]
                if settings.get("conflict_restart"):
                    command += ["--conflict-restart"]
                if settings.get("unconstrained_theta"):
                    command += ["--unconstrained-theta"]
                jobs.append({
                    "name": job_name, "pair": pair, "arm": arm, "config": config_tag,
                    "silhouette": cfg["silhouette"], "negative_space": cfg["negative_space"],
                    "area_normalized": cfg["area_normalized"],
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
    lines = ["job_id\tjob_name\tpair\tarm\tconfig\tsilhouette_weight\t"
             "negative_space_weight\tarea_normalized\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['arm']}\t"
            f"{row['config']}\t{row['silhouette']:.6g}\t{row['negative_space']:.6g}\t"
            f"{row['area_normalized']}\t{row['output_dir']}"
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


def _pair_arm_config(job_name: str) -> tuple[str, str, str]:
    """Split 'neg_<pair>_<arm>_<config>' back into its parts."""
    stripped = re.sub(r"^neg_", "", job_name)
    for config in sorted(CONFIG_TAGS, key=len, reverse=True):
        if stripped.endswith(f"_{config}"):
            rest = stripped[: -len(config) - 1]
            for arm in sorted(ARMS, key=len, reverse=True):
                if rest.endswith(f"_{arm}"):
                    return rest[: -len(arm) - 1], arm, config
            return rest, "-", config
    return stripped, "-", "-"


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
        pair, arm, config = _pair_arm_config(report.parent.name)
        row = {"job": report.parent.name, "pair": pair, "arm": arm, "config": config}
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
    config_order = {c: i for i, c in enumerate(CONFIG_TAGS)}
    arm_order = {a: i for i, a in enumerate(ARMS)}
    rows.sort(key=lambda r: (config_order.get(r["config"], 99),
                             arm_order.get(r["arm"], 99),
                             pair_order.get(r["pair"], 99)))

    # --- per-job summary (Tversky beside the IoU block)
    summary_keys = list(_SUMMARY_KEYS)
    insert_at = summary_keys.index("best_mean_iou_step") + 1
    summary_keys[insert_at:insert_at] = ["final_mean_tversky", "best_mean_tversky"]
    columns = ["job", "pair", "arm", "config"] + summary_keys
    summary_path = collected / "summary.tsv"
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- curves over steps, tagged by config + arm
    curves_path = collected / "curves.csv"
    curve_lines: list[str] = []
    missing_curves = 0
    for row in rows:
        history = sweep_dir / row["job"] / "history.csv"
        body = history.read_text(encoding="utf-8").splitlines() if history.exists() else []
        if not body:
            missing_curves += 1
            continue
        prefix = f"{row['pair']},{row['arm']},{row['config']}"
        if not curve_lines:
            curve_lines.append(f"pair,arm,config,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{prefix},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # --- final views, renamed by pair/arm/config
    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_count = 0
    for row in rows:
        for image in sorted((sweep_dir / row["job"]).glob("*.png")):
            suffix = image.stem.split("_")[-1]
            shutil.copy2(image, views_dir / f"{row['pair']}_{row['arm']}_{row['config']}_{suffix}.png")
            view_count += 1

    # --- headline: means per (config, arm)
    mean_columns = [
        "final_mean_iou", "best_mean_iou", "final_mean_tversky", "best_mean_tversky",
        "final_patches", "max_patches", "final_mean_spill", "final_mean_coverage",
        "srd_total_deletes", "final_step", "total_seconds",
    ]
    by_group_path = collected / "by_config_arm.tsv"
    by_group_lines = ["config\tarm\tn\t" + "\t".join(mean_columns)]
    for config in CONFIG_TAGS:
        for arm in ARMS:
            members = [r for r in rows if r["config"] == config and r["arm"] == arm]
            if not members:
                continue
            values = [_mean([m[column] for m in members]) for column in mean_columns]
            by_group_lines.append(f"{config}\t{arm}\t{len(members)}\t" + "\t".join(values))
    by_group_path.write_text("\n".join(by_group_lines) + "\n", encoding="utf-8")

    # --- console
    print(f"Tversky = TP / (TP + {alpha:g}*spill + {beta:g}*miss); "
          f"IoU is the same with both weights 1.")
    print(f"{'job':<40} {'config':<9} {'arm':<7} {'f_iou':>8} {'f_tvsky':>8} "
          f"{'f_pc':>5} {'dels':>5} {'spill':>8} {'step':>6} {'stop':<10}")
    print("-" * 120)
    for row in rows:
        print(
            f"{row['job']:<40} {row['config']:<9} {row['arm']:<7} "
            f"{row['final_mean_iou']:>8} {row['final_mean_tversky']:>8} "
            f"{row['final_patches']:>5} {row['srd_total_deletes']:>5} "
            f"{row['final_mean_spill']:>8} {row['final_step']:>6} {row['stop_reason']:<10}"
        )

    print("\nBy config x arm (means over pairs):")
    print(f"{'config':<10} {'arm':<7} {'n':>3} {'f_iou':>8} {'best_iou':>9} "
          f"{'f_tvsky':>8} {'best_tv':>8} {'f_pc':>6} {'spill':>8}")
    for line in by_group_lines[1:]:
        fields = line.split("\t")
        config, arm, n = fields[0], fields[1], fields[2]
        values = dict(zip(mean_columns, fields[3:]))
        print(
            f"{config:<10} {arm:<7} {n:>3} {values['final_mean_iou']:>8} "
            f"{values['best_mean_iou']:>9} {values['final_mean_tversky']:>8} "
            f"{values['best_mean_tversky']:>8} {values['final_patches']:>6} "
            f"{values['final_mean_spill']:>8}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_group_path):
        print(f"  {path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    if args.arms is None:
        args.arms = list(ARMS)
    if args.configs is None:
        args.configs = list(CONFIG_TAGS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"neg_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = NEGSPACE_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir, TVERSKY_ALPHA, TVERSKY_BETA)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    print(f"[Sweep] negspace/{sweep_name}: {len(jobs)} job(s) "
          f"({len(args.pairs)} pair(s) x {len(args.arms)} arm(s) x {len(args.configs)} config(s))")
    print(f"[Sweep] uniform deletion, constrained theta, render-every {args.render_every}, "
          f"min-steps {args.min_steps}, early stop on plateau, ceiling {args.steps}")
    for config_tag in args.configs:
        cfg = CONFIGS[config_tag]
        norm = "area-normalized" if cfg["area_normalized"] else "NON-normalized"
        print(f"[Sweep]   {config_tag:<9} silhouette={cfg['silhouette']:.4g} "
              f"negative_space={cfg['negative_space']:.4g}  ({norm}; {cfg['label']})")
    print(f"[Sweep] arms: {', '.join(args.arms)}")

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
        print("Monitor with: squeue --me | grep neg_")
        print(f"Aggregate when done: python submit_negspace.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
