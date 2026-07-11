"""Command-line benchmark for the perspective-sculpture optimizer.

Runs SceneOptimizer headlessly (no Qt) with the same scene setup as the UI
(two cameras 90 degrees apart, hanging plane, experimental patch init) and
writes per-trial and aggregate results to a text report.

Usage
-----
    python benchmark.py --steps 200 --trials 3
    python benchmark.py --steps 500 --trials 5 \
        --silhouette-weight 2.0 --negative-space-weight 4.0 \
        --overlap-weight 0.7 --camera-bounds-weight 0.3 \
        --target1 images/circle-15.png --output benchmark.txt
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core.initialization import init_experimental
from core.optimizer import SceneOptimizer, constrain_patch_to_square_xz_bounds
from scene.camera import Camera

# Matches ui/main_window.py scene constants.
_TARGET_TRANSPARENT_BORDER_FRACTION = 0.10
_ORIGINAL_BOUNDS_SIZE = 5.0
_HANGING_PLANE_Y = (_ORIGINAL_BOUNDS_SIZE * 0.5) + 1.0
_SCENE_CAMERA_FOV_DEG = 30.22

_DEFAULT_TARGET1 = "images/465-4650566_emojione-bw-1f332-simple-christmas-tree-cartoon-hd.png"
_DEFAULT_TARGET2 = "images/illustration-of-black-cat-face-peeking-hd-transparent-png-735811696611630frpfgrklwv.png"


def _make_scene_cameras() -> list[Camera]:
    radius = 8.0
    origin = np.zeros(3, dtype=np.float32)
    shared = dict(
        target=origin,
        fov=_SCENE_CAMERA_FOV_DEG,
        aspect=4.0 / 3.0,
        near=0.35,
        far=18.0,
    )
    cam1 = Camera(
        position=np.array([0.0, 0.0, radius], dtype=np.float32),
        label="View 1",
        **shared,
    )
    cam2 = Camera(
        position=np.array([radius, 0.0, 0.0], dtype=np.float32),
        label="View 2",
        **shared,
    )
    return [cam1, cam2]


def _load_target_image_with_border(path: str) -> np.ndarray:
    """Load an image as RGBA with a transparent border, like the UI does."""
    from PIL import Image, ImageOps

    image = Image.open(path).convert("RGBA")
    border = max(1, int(round(max(image.size) * _TARGET_TRANSPARENT_BORDER_FRACTION)))
    return np.array(ImageOps.expand(image, border=border, fill=(0, 0, 0, 0)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the SceneOptimizer over repeated trials.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=200,
                        help="Optimization steps per trial.")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of independent trials (seeds base-seed, base-seed+1, ...).")

    weights = parser.add_argument_group("loss weights")
    weights.add_argument("--silhouette-weight", type=float, default=2.0)
    weights.add_argument("--negative-space-weight", type=float, default=4.0)
    weights.add_argument("--overlap-weight", type=float, default=0.7)
    weights.add_argument("--camera-bounds-weight", type=float, default=0.3)

    parser.add_argument("--target1", type=str, default=_DEFAULT_TARGET1,
                        help="View 1 target image path.")
    parser.add_argument("--target2", type=str, default=_DEFAULT_TARGET2,
                        help="View 2 target image path (pass 'none' to disable).")
    parser.add_argument("--patches", type=int, default=30,
                        help="Initial patch count.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; trial i uses seed + i.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (cpu or cuda).")
    parser.add_argument("--no-srd", action="store_true",
                        help="Disable Stochastic Rewrite Descent.")
    parser.add_argument("--output", type=str, default="benchmark.txt",
                        help="Report file path.")
    return parser.parse_args()


def _run_trial(args: argparse.Namespace, trial: int, target1: np.ndarray,
               target2: np.ndarray | None) -> dict[str, float]:
    seed = args.seed + trial
    torch.manual_seed(seed)
    cameras = _make_scene_cameras()

    patches = init_experimental(args.patches, cameras, device=args.device, seed=seed)
    half_plane = _ORIGINAL_BOUNDS_SIZE * 0.5
    with torch.no_grad():
        for patch in patches:
            constrain_patch_to_square_xz_bounds(patch, half_plane)

    srd_config = None if args.no_srd else {
        "enabled": True,
        "interval": 50,
        "rewrite_eval_steps": 5,
        "candidate_count": 64,
        "max_patches": 200,
        "min_patches": 4,
    }

    optimizer = SceneOptimizer(
        patches,
        cameras[0],
        cameras[1],
        target1,
        target2,
        lr=args.lr,
        device=args.device,
        silhouette_weight=args.silhouette_weight,
        negative_space_weight=args.negative_space_weight,
        overlap_weight=args.overlap_weight,
        camera_bounds_weight=args.camera_bounds_weight,
        hanging_plane_size=_ORIGINAL_BOUNDS_SIZE,
        hanging_plane_y=_HANGING_PLANE_Y,
        srd_config=srd_config,
    )

    metrics: dict[str, float] = {}
    min_loss = float("inf")
    start = time.perf_counter()
    for step_idx, metrics in optimizer.run(n_steps=args.steps):
        min_loss = min(min_loss, metrics["loss"])
        if step_idx % 25 == 0 or step_idx == args.steps:
            print(
                f"  trial {trial + 1} step {step_idx}/{args.steps}"
                f" loss={metrics['loss']:.6f} patches={int(metrics['patches'])}",
                flush=True,
            )
    elapsed = time.perf_counter() - start

    return {
        "seed": float(seed),
        "final_loss": metrics.get("loss", float("nan")),
        "min_loss": min_loss,
        "view1_mse": metrics.get("view1_mse", float("nan")),
        "view2_loss": metrics.get("view2_loss", float("nan")),
        "view1_silhouette": metrics.get("view1_silhouette", float("nan")),
        "view2_silhouette": metrics.get("view2_silhouette", float("nan")),
        "final_patches": metrics.get("patches", float("nan")),
        "time_s": elapsed,
        "steps_per_s": args.steps / elapsed if elapsed > 0 else float("nan"),
    }


def _mean_std(values: list[float]) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f} +/- {std:.6f}"


def _write_report(path: Path, args: argparse.Namespace,
                  results: list[dict[str, float]]) -> None:
    lines = [
        "Perspective Sculptor optimizer benchmark",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Configuration",
        "-------------",
        f"steps per trial:       {args.steps}",
        f"trials:                {args.trials}",
        f"silhouette weight:     {args.silhouette_weight}",
        f"negative space weight: {args.negative_space_weight}",
        f"overlap weight:        {args.overlap_weight}",
        f"camera bounds weight:  {args.camera_bounds_weight}",
        f"learning rate:         {args.lr}",
        f"initial patches:       {args.patches}",
        f"SRD:                   {'disabled' if args.no_srd else 'enabled'}",
        f"device:                {args.device}",
        f"base seed:             {args.seed}",
        f"target 1:              {args.target1}",
        f"target 2:              {args.target2}",
        "",
        "Per-trial results",
        "-----------------",
    ]

    header = (
        f"{'trial':>5}  {'seed':>5}  {'final_loss':>12}  {'min_loss':>12}"
        f"  {'view1_mse':>12}  {'view2_loss':>12}  {'patches':>7}"
        f"  {'time_s':>8}  {'steps/s':>8}"
    )
    lines.append(header)
    for i, r in enumerate(results):
        lines.append(
            f"{i + 1:>5}  {int(r['seed']):>5}  {r['final_loss']:>12.6f}"
            f"  {r['min_loss']:>12.6f}  {r['view1_mse']:>12.6f}"
            f"  {r['view2_loss']:>12.6f}  {int(r['final_patches']):>7}"
            f"  {r['time_s']:>8.2f}  {r['steps_per_s']:>8.2f}"
        )

    lines += [
        "",
        "Aggregate (mean +/- std)",
        "------------------------",
        f"final loss:  {_mean_std([r['final_loss'] for r in results])}",
        f"min loss:    {_mean_std([r['min_loss'] for r in results])}",
        f"view1 mse:   {_mean_std([r['view1_mse'] for r in results])}",
        f"view2 loss:  {_mean_std([r['view2_loss'] for r in results])}",
        f"time (s):    {_mean_std([r['time_s'] for r in results])}",
        f"steps/s:     {_mean_std([r['steps_per_s'] for r in results])}",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")

    target1 = _load_target_image_with_border(args.target1)
    target2 = (
        None if args.target2.lower() in ("none", "")
        else _load_target_image_with_border(args.target2)
    )

    print(
        f"Benchmark: {args.trials} trial(s) x {args.steps} step(s) on {args.device}\n"
        f"weights: silhouette={args.silhouette_weight}"
        f" negative_space={args.negative_space_weight}"
        f" overlap={args.overlap_weight}"
        f" camera_bounds={args.camera_bounds_weight}",
        flush=True,
    )

    results: list[dict[str, float]] = []
    for trial in range(args.trials):
        print(f"Trial {trial + 1}/{args.trials} (seed {args.seed + trial})", flush=True)
        results.append(_run_trial(args, trial, target1, target2))
        r = results[-1]
        print(
            f"  done: final_loss={r['final_loss']:.6f}"
            f" time={r['time_s']:.2f}s ({r['steps_per_s']:.2f} steps/s)",
            flush=True,
        )

    output = Path(args.output)
    _write_report(output, args, results)
    print(f"Report written to {output.resolve()}")


if __name__ == "__main__":
    main()
