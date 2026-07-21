"""Basic-vs-SRD comparison with the geometry represented by fan fill.

Same arms as submit_comparison.py ('basic' = no SRD, 'srd' = the full method),
but two things differ:

    * patches are triangulated with a vertex-0 fan rather than earcut, forced
      via core.patch.set_fan_fill so the representation is a property of the
      run instead of whichever triangulation library the environment happens
      to have installed;
    * IoU is the headline metric — it is sampled every ``--eval-interval``
      steps into a CSV and the final value is reported, instead of only the
      end-of-run metrics dump that run_ablation.py writes.

Each invocation runs a single trial (one method, one seed) so trials spread
across SLURM jobs; submit_fanfill.py fans them out.

Outputs into ``--output-dir``:
    fanfill.txt       human-readable report (parsed by submit_fanfill.py)
    iou_history.csv   step,seconds,mean_iou,view1_iou,view2_iou,loss,patches

Example:
    python compare_fanfill.py --target1 images/horse.png \
        --target2 images/circle.png --method srd --seed 0 --steps 2000 \
        --output-dir results/fanfill/horse_circle_srd_seed0
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fan-fill comparison trial and track IoU over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", required=True, help="View 2 target image path.")
    parser.add_argument(
        "--method",
        choices=("basic", "srd"),
        required=True,
        help="'basic' disables SRD entirely; 'srd' is the full method.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Trial seed.")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Sample IoU every N steps (costs one extra no-grad render).",
    )
    parser.add_argument(
        "--no-fan-fill",
        action="store_true",
        help="Control arm: keep the default earcut triangulation.",
    )

    run = parser.add_argument_group("run settings")
    run.add_argument("--n-patches", type=int, default=20)
    run.add_argument("--swept-resolution", type=int, default=256)
    run.add_argument("--lr", type=float, default=3.5e-3)
    run.add_argument("--device", default="cuda")
    run.add_argument("--palette", default="")
    run.add_argument("--hanging-plane-size", type=float, default=5.0)
    run.add_argument("--srd-interval", type=int, default=50)
    run.add_argument("--srd-candidates", type=int, default=32)
    run.add_argument("--max-hours", type=float, default=10.0, help="Hard wall-clock cap.")
    run.add_argument("--output-dir", required=True, help="Report/CSV directory.")
    return parser.parse_args()


def _write_history_csv(path: Path, history: list[dict]) -> None:
    lines = ["step,seconds,mean_iou,view1_iou,view2_iou,loss,patches"]
    for entry in history:
        lines.append(
            f"{entry['step']},{entry['seconds']:.3f},{entry['mean_iou']:.6f},"
            f"{entry['view1_iou']:.6f},{entry['view2_iou']:.6f},"
            f"{entry['loss']:.6f},{entry['patches']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    args: argparse.Namespace,
    started_at: str,
    history: list[dict],
    stop_reason: str,
    total_seconds: float,
    setup_seconds: float,
) -> None:
    best = max(history, key=lambda entry: entry["mean_iou"]) if history else None
    final = history[-1] if history else None

    lines = [
        "Fan-fill comparison report",
        f"started={started_at}",
        f"geometry={'fan_fill' if not args.no_fan_fill else 'earcut'}",
        f"method={args.method}",
        f"seed={args.seed}",
        f"target1={args.target1}",
        f"target2={args.target2}",
        f"steps={args.steps}",
        f"eval_interval={args.eval_interval}",
        f"n_patches={args.n_patches}",
        f"swept_volume_resolution={args.swept_resolution}",
        f"learning_rate={args.lr:.6g}",
        f"device={args.device}",
        "",
        "Result:",
        f"  stop_reason={stop_reason}",
        f"  setup_seconds={setup_seconds:.3f}",
        f"  total_seconds={total_seconds:.3f}",
    ]

    if best is not None and final is not None:
        lines.extend([
            f"  final_step={final['step']}",
            f"  final_mean_iou={final['mean_iou']:.6f}",
            f"  final_view1_iou={final['view1_iou']:.6f}",
            f"  final_view2_iou={final['view2_iou']:.6f}",
            f"  final_loss={final['loss']:.6f}",
            f"  final_patches={final['patches']}",
            f"  best_mean_iou={best['mean_iou']:.6f}",
            f"  best_mean_iou_step={best['step']}",
        ])

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

    # Before any patch is constructed, so every mesh built in this process
    # uses the same triangulation.
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

    srd_config = None
    if args.method == "srd":
        srd_config = {
            "enabled": True,
            "interval": args.srd_interval,
            "candidate_count": args.srd_candidates,
            "disable_swept_volume_adds": False,
            "loss_only_deletion": False,
            "disable_splitting": False,
        }

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
        srd_config=srd_config,
        swept_volume=swept_volume,
    )

    setup_seconds = time.perf_counter() - setup_started
    print(
        f"[Run] geometry={'fan_fill' if not args.no_fan_fill else 'earcut'}, "
        f"method={args.method}, seed={args.seed}, steps={args.steps}, "
        f"setup={setup_seconds:.1f}s",
        flush=True,
    )

    max_seconds = args.max_hours * 3600.0
    optimization_started = time.perf_counter()

    history: list[dict] = []
    stop_reason = "steps"

    for step_index in range(1, args.steps + 1):
        optimizer.step(step_index, args.steps)

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
            "mean_iou": float(metrics.get("mean_iou", 0.0)),
            "view1_iou": float(metrics.get("view1_iou", 0.0)),
            "view2_iou": float(metrics.get("view2_iou", 0.0)),
            "loss": float(metrics["loss"]),
            "patches": int(metrics["patches"]),
        }
        history.append(entry)

        if step_index % (args.eval_interval * 4) == 0 or step_index == 1:
            print(
                f"  step {step_index}/{args.steps}: "
                f"loss={entry['loss']:.6f}, mean_iou={entry['mean_iou']:.6f}, "
                f"patches={entry['patches']}, {entry['seconds']:.1f}s",
                flush=True,
            )

        if elapsed >= max_seconds:
            stop_reason = "max_hours"
            print(f"  [stop] wall-clock cap reached at step {step_index}", flush=True)
            break

    total_seconds = time.perf_counter() - optimization_started

    _write_history_csv(output_dir / "iou_history.csv", history)
    _write_report(
        output_dir / "fanfill.txt",
        args,
        started_at,
        history,
        stop_reason,
        total_seconds,
        setup_seconds,
    )
    print(f"\n[Fan fill] report written to {output_dir / 'fanfill.txt'}", flush=True)


if __name__ == "__main__":
    main()
