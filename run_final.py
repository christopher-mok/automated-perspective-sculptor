"""Single-trial runner for the final comparison, ablation and overlap batches.

One invocation is one trial (one arm, one seed), so trials spread across SLURM
jobs. Every run records loss and IoU on a fixed step interval, not just at the
end, because the questions these batches answer are as much about *when* a run
gets somewhere as about where it ends up -- the swept-volume ablation in
particular is expected to reach a similar final loss more slowly.

Arms
----
    basic          SRD disabled entirely (the baseline method).
    srd            The full method: SRD with swept-volume-guided additions.
    srd_no_swept   SRD with swept-volume-guided additions disabled; every
                   other SRD component (rule-based deletion, splitting) stays
                   on. This is the only ablation the final batch runs.

Fan-fill triangulation is on by default so geometry is a property of the run
rather than of whichever triangulation library the environment happens to
have. Swept-volume resolution defaults to 256 and every SRD addition is drawn
from the swept volume.

Outputs into ``--output-dir``:
    report.txt    human-readable report, parsed by submit_final.py
    history.csv   step,seconds,loss,mean_iou,view1_iou,view2_iou,patches,
                  overlap,overlap_repaired_pairs

Example:
    python run_final.py --target1 images/horse.png --target2 images/circle.png \
        --arm srd --seed 0 --steps 2000 --output-dir results/final/run
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from run_ablation import (
    _HANGING_PLANE_Y,
    _load_target_image_with_border,
    _make_scene_cameras,
)

_PROJECT_ROOT = Path(__file__).resolve().parent

ARMS = ("basic", "srd", "srd_no_swept")

# Absolute mean-IoU milestones whose first crossing is reported, so a run that
# converges faster shows up as smaller steps/seconds at the same milestone
# rather than only as a different endpoint.
IOU_MILESTONES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)

# Fractions of a run's *own* best IoU, which stay meaningful when a pair never
# reaches the absolute milestones at all.
RELATIVE_MILESTONES = (0.90, 0.95, 0.99)


def _save_render(render, path: Path) -> None:
    """Write an RGBA render tensor/array to a PNG."""
    from PIL import Image

    array = np.asarray(render)
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGBA").save(path)


def _render_stem(args: argparse.Namespace) -> str:
    """Self-describing basename for exported views, e.g. horse-circle_srd_lam0p01_seed0.

    Encodes both targets, the arm and the piece-count weight so a view stays
    organizable after being pulled out of its job directory. The decimal point
    in lambda becomes 'p' (and a leading minus 'm') so the name is filesystem-
    and sort-friendly.
    """
    stem1 = Path(args.target1).stem
    stem2 = Path(args.target2).stem
    lam_tag = f"lam{args.lambda_count:g}".replace("-", "m").replace(".", "p")
    return f"{stem1}-{stem2}_{args.arm}_{lam_tag}_seed{args.seed}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one final-batch trial, tracking loss and IoU over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", required=True, help="View 2 target image path.")
    parser.add_argument(
        "--arm",
        choices=ARMS,
        required=True,
        help="'basic' = no SRD, 'srd' = full method, "
             "'srd_no_swept' = SRD without swept-volume-guided additions.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Trial seed.")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Sample loss and IoU every N steps (one extra no-grad render).",
    )
    parser.add_argument(
        "--no-fan-fill",
        action="store_true",
        help="Control arm: keep the default earcut triangulation.",
    )

    run = parser.add_argument_group("run settings")
    run.add_argument("--n-patches", type=int, default=20)
    run.add_argument("--swept-resolution", type=int, default=256)
    run.add_argument(
        "--swept-spawn-fraction",
        type=float,
        default=1.0,
        help="Fraction of SRD additions drawn from the swept volume.",
    )
    run.add_argument("--lr", type=float, default=3.5e-3)
    run.add_argument("--device", default="cuda")
    run.add_argument("--palette", default="")
    run.add_argument("--hanging-plane-size", type=float, default=5.0)
    run.add_argument("--srd-interval", type=int, default=50)
    run.add_argument("--srd-candidates", type=int, default=32)

    ops = parser.add_argument_group(
        "SRD operation mix",
        "How the per-step candidate budget (--srd-candidates) is divided "
        "between the three rewrite kinds. These are proportions of the "
        "candidates SRD gets to *look at*, not acceptance probabilities: "
        "every candidate still has to beat the loss to be applied. The "
        "weights are normalized, so only their ratios matter. The defaults "
        "0.35/0.15/0.50 are the historical mix.",
    )
    ops.add_argument("--add-weight", type=float, default=0.35)
    ops.add_argument("--delete-weight", type=float, default=0.15)
    ops.add_argument("--split-weight", type=float, default=0.50)
    run.add_argument(
        "--lambda-count",
        type=float,
        default=0.05,
        help="Piece-count penalty: SRD optimizes loss + lambda_count * n_patches, "
             "so a rewrite must beat the loss by more than this per piece it adds. "
             "Higher values trade IoU for fewer pieces. No effect on the 'basic' arm. "
             "Under --lambda-mode dual this is only the starting value.",
    )

    dual = parser.add_argument_group(
        "piece-count penalty schedule",
        "A fixed --lambda-count is an absolute threshold in loss units, so it "
        "rations pieces hardest on the shapes whose pieces are each worth "
        "least -- i.e. the complicated ones that need the most. Dual mode "
        "instead treats lambda_count as a multiplier on the constraint "
        "mean_iou >= --target-iou and adapts it during the run, letting each "
        "shape find its own piece count.",
    )
    dual.add_argument(
        "--lambda-mode",
        choices=("fixed", "dual"),
        default="fixed",
        help="'fixed' holds --lambda-count constant; 'dual' adapts it by dual "
             "ascent toward --target-iou.",
    )
    dual.add_argument(
        "--target-iou",
        type=float,
        default=0.90,
        help="Mean-IoU constraint for --lambda-mode dual.",
    )
    dual.add_argument(
        "--lambda-eta",
        type=float,
        default=0.5,
        help="Dual-ascent step size on log lambda_count.",
    )
    dual.add_argument("--lambda-min", type=float, default=1e-5)
    dual.add_argument(
        "--lambda-max",
        type=float,
        default=1.0,
        help="Bounds on the adapted lambda_count. The floor must stay above "
             "zero: the update is multiplicative, so a lambda that reaches 0 "
             "can never grow back.",
    )

    count = parser.add_argument_group(
        "piece-count objective",
        "Minimise the number of pieces subject to an IoU floor, by scoring "
        "SRD rewrites on J = n_patches + lambda * hinge(target - mean_iou)^p "
        "instead of on the image loss. The hinge switches the IoU term off "
        "once the run is above --count-target-iou, so from there on rewrites "
        "are judged purely on pieces and SRD spends the rest of the run "
        "shedding them; below the threshold it pays for coverage again. The "
        "gradient steps between rewrites still descend the ordinary image "
        "loss -- this changes which discrete rewrites are accepted, which is "
        "where the piece count is decided.",
    )
    count.add_argument(
        "--count-objective",
        action="store_true",
        help="Score rewrites by J instead of by loss. --lambda-count is then unused.",
    )
    count.add_argument("--count-target-iou", type=float, default=0.90)
    count.add_argument(
        "--count-lambda",
        type=float,
        default=2000.0,
        help="Exchange rate: how many pieces one unit of squared IoU deficit "
             "is worth. At 2000, being 0.02 below target costs 0.8 pieces and "
             "0.10 below costs 20.",
    )
    count.add_argument(
        "--count-power",
        type=float,
        default=2.0,
        help="Hinge exponent. 2 is the quadratic in the brief; its slope "
             "vanishes at the threshold, so a run settles a little under the "
             "target. 1 keeps a constant slope up to the kink.",
    )
    count.add_argument(
        "--count-softplus-beta",
        type=float,
        default=0.0,
        help="0 uses the hard max. Positive values replace the kink with a "
             "softplus of that sharpness -- smooth, but it leaks a little "
             "penalty above the threshold too.",
    )

    prune = parser.add_argument_group(
        "pruning path",
        "After the run, delete the least-damaging piece repeatedly and record "
        "mean IoU at every piece count, giving the whole IoU-vs-pieces curve "
        "for this shape from one run. The model is left at the knee, not at "
        "one piece.",
    )
    prune.add_argument(
        "--prune-path",
        action="store_true",
        help="Walk the pruning path after the run and write pruning_path.csv.",
    )
    prune.add_argument(
        "--prune-min-pieces",
        type=int,
        default=1,
        help="Stop the path at this many pieces.",
    )
    prune.add_argument(
        "--prune-refit-steps",
        type=int,
        default=4,
        help="Gradient steps to re-fit the survivors after each deletion. "
             "Scoring a deletion on frozen geometry overstates its cost, "
             "because the neighbours would have grown to cover the hole.",
    )
    prune.add_argument(
        "--prune-tolerance",
        type=float,
        default=0.01,
        help="Keep the fewest pieces within this much mean IoU of the best "
             "point on the path. Ignored when --prune-target-iou is set.",
    )
    prune.add_argument(
        "--prune-target-iou",
        type=float,
        default=None,
        help="Instead of a tolerance, keep the fewest pieces still reaching "
             "this mean IoU. Falls back to the best point if unreachable.",
    )
    run.add_argument("--max-hours", type=float, default=10.0, help="Hard wall-clock cap.")
    run.add_argument(
        "--no-renders",
        action="store_true",
        help="Skip exporting the two final rendered camera views.",
    )
    run.add_argument(
        "--render-scale",
        type=int,
        default=4,
        help="Resolution multiplier for the exported final views "
             "(relative to the 192x256 optimization resolution).",
    )
    run.add_argument("--output-dir", required=True, help="Report/CSV directory.")

    stop = parser.add_argument_group("early stopping")
    stop.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop once both mean IoU and loss have plateaued (never before "
             "--min-steps). Off by default so existing batches are unchanged.",
    )
    stop.add_argument(
        "--min-steps",
        type=int,
        default=1000,
        help="Never early-stop before this step, however flat the curve looks.",
    )
    stop.add_argument(
        "--patience-steps",
        type=int,
        default=300,
        help="Steps without improvement in *either* metric before stopping.",
    )
    stop.add_argument(
        "--iou-delta",
        type=float,
        default=0.002,
        help="Absolute mean-IoU gain over the running best that counts as progress.",
    )
    stop.add_argument(
        "--loss-delta",
        type=float,
        default=0.001,
        help="Relative loss drop below the running best that counts as progress.",
    )

    weights = parser.add_argument_group("view-loss weights")
    weights.add_argument(
        "--silhouette-weight",
        type=float,
        default=3.6,
        help="Silhouette term weight, applied to both views.",
    )
    weights.add_argument(
        "--negative-space-weight",
        type=float,
        default=0.9,
        help="Negative-space term weight, applied to both views. The default "
             "pair is a silhouette:negative-space ratio of 4 at a fixed sum "
             "of 4.5.",
    )

    overlap = parser.add_argument_group("overlap test")
    overlap.add_argument(
        "--overlap-mode",
        choices=("sphere", "planar"),
        default="planar",
        help="'sphere' is the legacy orientation-blind bounding-sphere test; "
             "'planar' clips both outlines against the plane intersection line.",
    )
    overlap.add_argument(
        "--overlap-repair",
        dest="overlap_repair",
        action="store_true",
        default=True,
        help="Hard-separate pairs the exact planar test finds interpenetrating.",
    )
    overlap.add_argument(
        "--no-overlap-repair",
        dest="overlap_repair",
        action="store_false",
        help="Soft penalty only; no hard repair pass.",
    )
    overlap.add_argument("--overlap-repair-interval", type=int, default=5)
    return parser.parse_args()


def _srd_config(args: argparse.Namespace) -> dict[str, object] | None:
    if args.arm == "basic":
        return None
    return {
        "enabled": True,
        "interval": args.srd_interval,
        "candidate_count": args.srd_candidates,
        "add_weight": args.add_weight,
        "delete_weight": args.delete_weight,
        "split_weight": args.split_weight,
        "count_objective": args.count_objective,
        "count_target_iou": args.count_target_iou,
        "count_lambda": args.count_lambda,
        "count_power": args.count_power,
        "count_softplus_beta": args.count_softplus_beta,
        "lambda_count": args.lambda_count,
        "lambda_mode": args.lambda_mode,
        "lambda_target_iou": args.target_iou,
        "lambda_eta": args.lambda_eta,
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "disable_swept_volume_adds": args.arm == "srd_no_swept",
        "swept_volume_spawn_fraction": args.swept_spawn_fraction,
        "loss_only_deletion": False,
        "disable_splitting": False,
    }


HISTORY_COLUMNS = (
    "step", "seconds", "loss", "mean_iou", "view1_iou", "view2_iou",
    "patches", "overlap", "overlap_repaired_pairs",
    # Negative-space decomposition of the IoU above. IoU already charges for
    # covering background (that area lands in the union); these say whether a
    # given IoU lost its area to unfilled target or to spill outside it.
    "mean_spill", "view1_spill", "view2_spill",
    "mean_precision", "view1_precision", "view2_precision",
    "mean_coverage", "view1_coverage", "view2_coverage",
    # The weighted loss terms, so the negative-space pressure is legible too.
    "view1_negative_space", "view2_negative_space",
    "view1_silhouette", "view2_silhouette",
    # The live piece-count penalty. Constant under --lambda-mode fixed; under
    # dual mode its trajectory is the run's own read on how many pieces the
    # shape needs to hold --target-iou.
    "lambda_count",
    # Cumulative rewrites accepted so far, one column per kind. The piece
    # count above is their running sum (adds + splits - deletes, plus the
    # starting pieces); these say which operation produced it.
    "total_adds", "total_splits", "total_deletes",
)

PRUNING_PATH_COLUMNS = (
    "patches", "deleted_patch_index", "loss", "mean_iou", "view1_iou",
    "view2_iou", "mean_coverage", "mean_precision", "mean_spill",
)

_FLOAT_COLUMNS = {
    "step", "patches", "overlap_repaired_pairs",
    "total_adds", "total_splits", "total_deletes",
}


def _write_history_csv(path: Path, history: list[dict]) -> None:
    lines = [",".join(HISTORY_COLUMNS)]
    for entry in history:
        fields = []
        for column in HISTORY_COLUMNS:
            value = entry.get(column, 0)
            if column in _FLOAT_COLUMNS:
                fields.append(str(int(value)))
            elif column == "seconds":
                fields.append(f"{value:.3f}")
            elif column == "overlap":
                fields.append(f"{value:.8f}")
            else:
                fields.append(f"{value:.6f}")
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pruning_path_csv(path: Path, rows: list[dict]) -> None:
    lines = [",".join(PRUNING_PATH_COLUMNS)]
    for row in rows:
        fields = []
        for column in PRUNING_PATH_COLUMNS:
            value = row.get(column, 0)
            if column in {"patches", "deleted_patch_index"}:
                fields.append(str(int(value)))
            else:
                fields.append(f"{value:.6f}")
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_crossing(history: list[dict], threshold: float) -> dict | None:
    """First evaluation whose mean_iou reached ``threshold``."""
    for entry in history:
        if entry["mean_iou"] >= threshold:
            return entry
    return None


def _milestone_lines(history: list[dict]) -> list[str]:
    """Steps and seconds to each IoU milestone -- the convergence-speed record."""
    lines: list[str] = []
    for threshold in IOU_MILESTONES:
        crossing = _first_crossing(history, threshold)
        tag = f"{threshold:.2f}".replace(".", "p")
        if crossing is None:
            lines.append(f"  iou_{tag}_steps=-1")
            lines.append(f"  iou_{tag}_seconds=-1")
        else:
            lines.append(f"  iou_{tag}_steps={crossing['step']}")
            lines.append(f"  iou_{tag}_seconds={crossing['seconds']:.3f}")

    best = max((entry["mean_iou"] for entry in history), default=0.0)
    for fraction in RELATIVE_MILESTONES:
        crossing = _first_crossing(history, best * fraction)
        tag = f"{int(round(fraction * 100))}"
        if crossing is None:
            lines.append(f"  rel_{tag}pct_steps=-1")
            lines.append(f"  rel_{tag}pct_seconds=-1")
        else:
            lines.append(f"  rel_{tag}pct_steps={crossing['step']}")
            lines.append(f"  rel_{tag}pct_seconds={crossing['seconds']:.3f}")
    return lines


def _write_report(
    path: Path,
    args: argparse.Namespace,
    started_at: str,
    history: list[dict],
    stop_reason: str,
    total_seconds: float,
    setup_seconds: float,
    srd_stats: object | None,
    pruning_selected: dict | None = None,
) -> None:
    best = max(history, key=lambda entry: entry["mean_iou"]) if history else None
    best_loss = min(history, key=lambda entry: entry["loss"]) if history else None
    final = history[-1] if history else None

    ratio = (
        args.silhouette_weight / args.negative_space_weight
        if args.negative_space_weight else float("inf")
    )
    lines = [
        "Final-batch run report",
        f"started={started_at}",
        f"arm={args.arm}",
        f"seed={args.seed}",
        f"geometry={'fan_fill' if not args.no_fan_fill else 'earcut'}",
        f"target1={args.target1}",
        f"target2={args.target2}",
        f"steps={args.steps}",
        f"eval_interval={args.eval_interval}",
        f"n_patches={args.n_patches}",
        f"swept_volume_resolution={args.swept_resolution}",
        f"swept_volume_spawn_fraction={args.swept_spawn_fraction:.6g}",
        f"srd_enabled={args.arm != 'basic'}",
        f"srd_candidates={args.srd_candidates}",
        f"add_weight={args.add_weight:.6g}",
        f"delete_weight={args.delete_weight:.6g}",
        f"split_weight={args.split_weight:.6g}",
        f"disable_swept_volume_adds={args.arm == 'srd_no_swept'}",
        f"lambda_count={args.lambda_count:.6g}",
        f"lambda_mode={args.lambda_mode}",
        f"count_objective={args.count_objective}",
        f"count_target_iou={args.count_target_iou:.6g}",
        f"count_lambda={args.count_lambda:.6g}",
        f"count_power={args.count_power:.6g}",
        f"count_softplus_beta={args.count_softplus_beta:.6g}",
        f"silhouette_weight={args.silhouette_weight:.6g}",
        f"negative_space_weight={args.negative_space_weight:.6g}",
        f"weight_ratio={ratio:.6g}",
        f"overlap_mode={args.overlap_mode}",
        f"overlap_repair={args.overlap_repair}",
        f"overlap_repair_interval={args.overlap_repair_interval}",
        f"learning_rate={args.lr:.6g}",
        f"early_stop={args.early_stop}",
        f"min_steps={args.min_steps}",
        f"patience_steps={args.patience_steps}",
        f"device={args.device}",
        "",
        "Result:",
        f"  stop_reason={stop_reason}",
        f"  setup_seconds={setup_seconds:.3f}",
        f"  total_seconds={total_seconds:.3f}",
    ]

    if best is not None and best_loss is not None and final is not None:
        lines.extend([
            f"  final_step={final['step']}",
            f"  final_loss={final['loss']:.6f}",
            f"  final_mean_iou={final['mean_iou']:.6f}",
            f"  final_view1_iou={final['view1_iou']:.6f}",
            f"  final_view2_iou={final['view2_iou']:.6f}",
            f"  final_patches={final['patches']}",
            f"  final_overlap={final['overlap']:.8f}",
            f"  best_mean_iou={best['mean_iou']:.6f}",
            f"  best_mean_iou_step={best['step']}",
            f"  best_loss={best_loss['loss']:.6f}",
            f"  best_loss_step={best_loss['step']}",
            # Negative space: how much area the render covers that it should
            # not. Already inside final_mean_iou; broken out to be readable.
            f"  final_mean_spill={final['mean_spill']:.6f}",
            f"  final_view1_spill={final['view1_spill']:.6f}",
            f"  final_view2_spill={final['view2_spill']:.6f}",
            f"  final_mean_precision={final['mean_precision']:.6f}",
            f"  final_mean_coverage={final['mean_coverage']:.6f}",
            f"  best_mean_spill={min(e['mean_spill'] for e in history):.6f}",
            # Patch-count trajectory, so "how many pieces did it land with"
            # can be read against how many it ever carried.
            f"  start_patches={history[0]['patches']}",
            f"  max_patches={max(e['patches'] for e in history)}",
            f"  min_patches={min(e['patches'] for e in history)}",
            f"  mean_patches={sum(e['patches'] for e in history) / len(history):.3f}",
            # Under --lambda-mode dual this is where the multiplier settled,
            # which is the run's own estimate of how hard the shape is.
            f"  final_lambda_count={final['lambda_count']:.6g}",
        ])

    if args.count_objective and final is not None:
        # What the run was actually minimising, and whether the constraint it
        # was minimising subject to was met. The interesting comparison is
        # patches_at_threshold vs final_patches: the first is what it took to
        # reach the IoU floor, the second what survived the shedding afterwards.
        deficit = max(0.0, args.count_target_iou - final["mean_iou"])
        crossing = _first_crossing(history, args.count_target_iou)
        lines.extend([
            "",
            "Piece-count objective:",
            f"  count_target_reached={final['mean_iou'] >= args.count_target_iou}",
            f"  count_final_deficit={deficit:.6f}",
            f"  count_final_objective="
            f"{final['patches'] + args.count_lambda * deficit ** args.count_power:.6f}",
            f"  patches_at_threshold={crossing['patches'] if crossing else -1}",
            f"  step_at_threshold={crossing['step'] if crossing else -1}",
            f"  patches_shed_after_threshold="
            f"{crossing['patches'] - final['patches'] if crossing else -1}",
        ])

    if srd_stats is not None:
        lines.extend([
            # Adds and splits both grow the sculpture by one piece, but they
            # are different moves: an add drops a new piece into uncovered
            # space, a split subdivides one that is already carrying geometry.
            f"  srd_total_adds={getattr(srd_stats, 'total_adds', 0)}",
            f"  srd_total_splits={getattr(srd_stats, 'total_splits', 0)}",
            f"  srd_total_growth={getattr(srd_stats, 'total_added', 0)}",
            f"  srd_total_deletes={getattr(srd_stats, 'total_deleted', 0)}",
        ])

    if pruning_selected:
        lines.extend([
            "",
            "Pruning path (see pruning_path.csv for the full curve):",
            f"  pruned_patches={pruning_selected['patches']}",
            f"  pruned_mean_iou={pruning_selected['mean_iou']:.6f}",
            f"  pruned_loss={pruning_selected['loss']:.6f}",
            f"  pruned_mean_coverage={pruning_selected['mean_coverage']:.6f}",
            f"  pruned_mean_precision={pruning_selected['mean_precision']:.6f}",
            f"  pruned_mean_spill={pruning_selected['mean_spill']:.6f}",
            f"  pruned_threshold={pruning_selected['threshold']:.6f}",
            f"  pruned_target_reached={pruning_selected['target_reached']}",
        ])

    lines.append("")
    lines.append("Convergence:")
    lines.extend(_milestone_lines(history))
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()

    started = datetime.now()
    started_at = started.isoformat(timespec="seconds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from core.initialization import initialize_patches
    from core.optimizer import SceneOptimizer
    from core.patch import set_fan_fill
    from core.swept_volume import SweptVolume

    # Before any patch is constructed, so every mesh built in this process uses
    # the same triangulation.
    set_fan_fill(not args.no_fan_fill)

    setup_started = time.perf_counter()

    cameras = _make_scene_cameras()
    target1 = _load_target_image_with_border(args.target1)
    target2 = _load_target_image_with_border(args.target2)

    print(f"[Swept volume] building (resolution={args.swept_resolution})...", flush=True)
    swept_volume = SweptVolume.from_images(
        target1,
        target2,
        cameras,
        hanging_plane_size=args.hanging_plane_size,
        resolution=args.swept_resolution,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    patches = initialize_patches(
        mode="Experimental",
        n_patches=args.n_patches,
        reference_image=target1,
        cameras=cameras,
        device=args.device,
        seed=args.seed,
        swept_volume=swept_volume,
    )
    optimizer = SceneOptimizer(
        patches,
        cameras[0],
        cameras[1],
        target1,
        target2,
        palette=args.palette,
        lr=args.lr,
        device=args.device,
        hanging_plane_size=args.hanging_plane_size,
        hanging_plane_y=_HANGING_PLANE_Y,
        srd_config=_srd_config(args),
        swept_volume=swept_volume,
        silhouette_weight=args.silhouette_weight,
        negative_space_weight=args.negative_space_weight,
        overlap_mode=args.overlap_mode,
        overlap_repair=args.overlap_repair,
        overlap_repair_interval=args.overlap_repair_interval,
    )

    setup_seconds = time.perf_counter() - setup_started
    print(
        f"[Run] arm={args.arm}, seed={args.seed}, steps={args.steps}, "
        f"overlap={args.overlap_mode} (repair={args.overlap_repair}), "
        f"weights={args.silhouette_weight:g}/{args.negative_space_weight:g}, "
        f"setup={setup_seconds:.1f}s",
        flush=True,
    )

    max_seconds = args.max_hours * 3600.0
    optimization_started = time.perf_counter()

    history: list[dict] = []
    stop_reason = "steps"

    # Plateau tracking: a run is converged only when *neither* IoU nor loss has
    # improved for --patience-steps. Requiring both keeps a run alive when IoU
    # has saturated but the loss is still tightening the fit (or vice versa).
    best_iou = float("-inf")
    best_loss = float("inf")
    last_improvement_step = 0

    for step_index in range(1, args.steps + 1):
        step_metrics = optimizer.step(step_index, args.steps)

        is_eval = (
            step_index % args.eval_interval == 0
            or step_index == 1
            or step_index == args.steps
        )
        if not is_eval:
            continue

        metrics, _, _ = optimizer.evaluate_snapshot()
        elapsed = time.perf_counter() - optimization_started
        entry = {
            "step": step_index,
            "seconds": elapsed,
            "patches": int(metrics["patches"]),
            "overlap": float(metrics.get("overlap", 0.0)),
            "overlap_repaired_pairs": int(step_metrics.get("overlap_repaired_pairs", 0)),
            "lambda_count": (
                optimizer.srd.lambda_count if optimizer.srd is not None else 0.0
            ),
            "total_adds": (
                optimizer.srd.stats.total_adds if optimizer.srd is not None else 0
            ),
            "total_splits": (
                optimizer.srd.stats.total_splits if optimizer.srd is not None else 0
            ),
            "total_deletes": (
                optimizer.srd.stats.total_deleted if optimizer.srd is not None else 0
            ),
        }
        for column in HISTORY_COLUMNS:
            if column not in entry:
                entry[column] = float(metrics.get(column, 0.0))
        history.append(entry)

        # Refresh the CSV as we go so a job killed by the walltime still leaves
        # a usable curve behind.
        _write_history_csv(output_dir / "history.csv", history)

        # Dual ascent on the piece-count penalty. The entry above logged the
        # lambda that was in force over the interval just measured; this sets
        # the one the next interval will run under.
        if optimizer.srd is not None and args.lambda_mode == "dual":
            optimizer.srd.update_lambda_count(entry["mean_iou"])

        if entry["mean_iou"] > best_iou + args.iou_delta:
            best_iou = max(best_iou, entry["mean_iou"])
            last_improvement_step = step_index
        elif entry["loss"] < best_loss * (1.0 - args.loss_delta):
            best_loss = min(best_loss, entry["loss"])
            last_improvement_step = step_index
        best_iou = max(best_iou, entry["mean_iou"])
        best_loss = min(best_loss, entry["loss"])

        if step_index % (args.eval_interval * 4) == 0 or step_index == 1:
            print(
                f"  step {step_index}/{args.steps}: "
                f"loss={entry['loss']:.6f}, mean_iou={entry['mean_iou']:.6f}, "
                f"spill={entry['mean_spill']:.4f}, patches={entry['patches']}, "
                f"overlap={entry['overlap']:.3e}, {entry['seconds']:.1f}s",
                flush=True,
            )

        stalled_for = step_index - last_improvement_step
        if (
            args.early_stop
            and step_index >= args.min_steps
            and stalled_for >= args.patience_steps
        ):
            stop_reason = "converged"
            print(
                f"  [stop] converged at step {step_index}: no IoU/loss gain "
                f"for {stalled_for} steps",
                flush=True,
            )
            break

        if elapsed >= max_seconds:
            stop_reason = "max_hours"
            print(f"  [stop] wall-clock cap reached at step {step_index}", flush=True)
            break

    total_seconds = time.perf_counter() - optimization_started

    # Pruning path: walk the IoU-vs-pieces curve for this shape and leave the
    # model at the knee. Runs after the timed loop, so total_seconds stays
    # comparable with runs that skip it.
    pruning_selected: dict | None = None
    if args.prune_path and optimizer.srd is not None:
        prune_started = time.perf_counter()
        print(
            f"\n[Prune] walking pruning path from {len(optimizer.patches)} pieces "
            f"down to {args.prune_min_pieces} (refit {args.prune_refit_steps} "
            f"steps per deletion)",
            flush=True,
        )
        path, pruning_selected = optimizer.srd.pruning_path(
            optimizer,
            optimizer.optim,
            min_pieces=args.prune_min_pieces,
            refit_steps=args.prune_refit_steps,
            iou_tolerance=args.prune_tolerance,
            target_iou=args.prune_target_iou,
        )
        _write_pruning_path_csv(output_dir / "pruning_path.csv", path)
        if pruning_selected:
            print(
                f"[Prune] kept {pruning_selected['patches']} pieces at "
                f"mean_iou={pruning_selected['mean_iou']:.6f} "
                f"(threshold {pruning_selected['threshold']:.6f}, "
                f"reached={pruning_selected['target_reached']}), "
                f"{time.perf_counter() - prune_started:.1f}s",
                flush=True,
            )

    _write_history_csv(output_dir / "history.csv", history)
    _write_report(
        output_dir / "report.txt",
        args,
        started_at,
        history,
        stop_reason,
        total_seconds,
        setup_seconds,
        optimizer.srd.stats if optimizer.srd is not None else None,
        pruning_selected,
    )
    print(f"\n[Final] report written to {output_dir / 'report.txt'}", flush=True)

    if not args.no_renders:
        scale = max(1, args.render_scale)
        with torch.no_grad():
            render1, render2 = optimizer.renderer.render_both(
                optimizer.patches,
                cameras[0],
                cameras[1],
                (optimizer.resolution[0] * scale, optimizer.resolution[1] * scale),
            )
        stem = _render_stem(args)
        view1_path = output_dir / f"{stem}_view1.png"
        view2_path = output_dir / f"{stem}_view2.png"
        _save_render(render1.detach().cpu(), view1_path)
        _save_render(render2.detach().cpu(), view2_path)
        print(f"[Final] views exported to {view1_path.name}, {view2_path.name}", flush=True)


if __name__ == "__main__":
    main()
