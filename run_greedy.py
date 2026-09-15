"""Single-trial runner for the greedy swept-volume square-packing baseline.

Companion to ``run_final.py``, but this method has nothing in common with
SRD or gradient descent: candidate squares are enumerated once from the
swept volume, scored, and greedily selected by weighted marginal pixel
coverage minus a spill penalty (see ``optimizer/greedy_pack.py``). There is
no training loop, no torch optimizer, and no CUDA/nvdiffrast dependency --
this arm runs entirely on CPU.

Because everything the greedy method needs lives behind
``optimizer.greedy_pack.GreedyPackConfig``, this file cannot change the
behaviour of ``run_final.py``, ``core/optimizer.py`` or ``optimizer/srd.py``:
it only imports ``SweptVolume`` and plain image-loading helpers from them.

Outputs into ``--output-dir`` (mirroring run_final.py's naming so the same
collection conventions apply):
    report.txt      human-readable summary
    history.csv     round,candidate,score,new_view1,new_view2,spill_view1,
                     spill_view2,covered_view1,covered_view2,panels
    patches.json    the selected panels as Patch dicts (Patch.to_dict())
    <stem>_view1.png / _view2.png   final coverage silhouettes

Example:
    python run_greedy.py --target1 images/cat_face.png --target2 images/bass.png \
        --output-dir results/greedy/run
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from run_ablation import _load_target_image_with_border, _make_scene_cameras

_PROJECT_ROOT = Path(__file__).resolve().parent
_RESOLUTION = (192, 256)  # (H, W); matches core.optimizer.SceneOptimizer's default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the greedy swept-volume square-packing baseline "
                    "(no SRD, no continuous optimization).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", required=True, help="View 2 target image path.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Only used to subsample candidates when the swept "
                              "volume yields more than --max-candidates points.")
    parser.add_argument("--hanging-plane-size", type=float, default=5.0)
    parser.add_argument("--swept-resolution", type=int, default=256,
                         help="Swept-volume voxel grid resolution (same knob as run_final.py).")

    panel = parser.add_argument_group(
        "candidate geometry",
        "A candidate is a fixed-size, fixed-yaw square panel placed at a "
        "swept-volume position -- guaranteed two-view-valid by construction.",
    )
    panel.add_argument("--panel-half-size", type=float, default=0.12,
                        help="Half-width/height of every candidate square, world units.")
    panel.add_argument("--theta", type=float, default=None,
                        help="Fixed yaw (radians) for every candidate. Default: the "
                             "bisector of the two camera yaw angles (45 degrees for "
                             "the standard 90-degree two-camera rig), which is "
                             "equally foreshortened -- and so equally square-looking "
                             "-- in both views.")
    panel.add_argument("--max-candidates", type=int, default=6000,
                        help="Thin the swept-volume point cloud to at most this many "
                             "candidate positions before scoring.")

    score = parser.add_argument_group(
        "greedy scoring",
        "score = w1*(new view-1 target pixels) + w2*(new view-2 target pixels) "
        "- lambda*(new spill pixels), evaluated against the currently uncovered "
        "target masks each round.",
    )
    score.add_argument("--view1-weight", type=float, default=0.5)
    score.add_argument("--view2-weight", type=float, default=0.5)
    score.add_argument("--spill-weight", type=float, default=1.0,
                        help="lambda: cost per newly covered pixel outside the "
                             "target silhouette in either view.")
    score.add_argument("--min-gain", type=float, default=1.0,
                        help="Stop once the best remaining marginal score falls "
                             "below this many pixel-equivalents.")
    score.add_argument("--max-panels", type=int, default=400,
                        help="Hard panel budget.")
    score.add_argument("--handle-scale", type=float, default=1e-4,
                        help="Near-zero Bezier handle length for the exported "
                             "Patch outlines (straight edges, i.e. a square).")

    parser.add_argument("--output-dir", required=True, help="Report/CSV directory.")
    parser.add_argument("--no-renders", action="store_true",
                         help="Skip writing the final coverage-silhouette PNGs.")
    return parser.parse_args()


def _render_stem(args: argparse.Namespace) -> str:
    stem1 = Path(args.target1).stem
    stem2 = Path(args.target2).stem
    return f"{stem1}-{stem2}_greedy_seed{args.seed}"


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    array = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


HISTORY_COLUMNS = (
    "round", "candidate", "score", "new_view1", "new_view2",
    "spill_view1", "spill_view2", "covered_view1", "covered_view2", "panels",
)


def _write_history_csv(path: Path, history: list[dict]) -> None:
    lines = [",".join(HISTORY_COLUMNS)]
    for entry in history:
        fields = []
        for column in HISTORY_COLUMNS:
            value = entry[column]
            fields.append(f"{value:.6f}" if column == "score" else str(int(value)))
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    args: argparse.Namespace,
    started_at: str,
    theta: float,
    result,
    total_seconds: float,
    setup_seconds: float,
) -> None:
    metrics = result.metrics
    lines = [
        "Greedy swept-volume square-packing run report",
        f"started={started_at}",
        f"arm=greedy",
        f"seed={args.seed}",
        f"target1={args.target1}",
        f"target2={args.target2}",
        f"hanging_plane_size={args.hanging_plane_size:.6g}",
        f"swept_volume_resolution={args.swept_resolution}",
        f"panel_half_size={args.panel_half_size:.6g}",
        f"theta={theta:.6g}",
        f"max_candidates={args.max_candidates}",
        f"view1_weight={args.view1_weight:.6g}",
        f"view2_weight={args.view2_weight:.6g}",
        f"spill_weight={args.spill_weight:.6g}",
        f"min_gain={args.min_gain:.6g}",
        f"max_panels={args.max_panels}",
        "",
        "Result:",
        f"  setup_seconds={setup_seconds:.3f}",
        f"  total_seconds={total_seconds:.3f}",
        f"  n_candidates={result.n_candidates}",
        f"  final_patches={int(metrics['patches'])}",
        f"  final_mean_iou={metrics['mean_iou']:.6f}",
        f"  final_view1_iou={metrics['view1_iou']:.6f}",
        f"  final_view2_iou={metrics['view2_iou']:.6f}",
        f"  final_mean_coverage={metrics['mean_coverage']:.6f}",
        f"  final_view1_coverage={metrics['view1_coverage']:.6f}",
        f"  final_view2_coverage={metrics['view2_coverage']:.6f}",
        f"  final_mean_precision={metrics['mean_precision']:.6f}",
        f"  final_view1_precision={metrics['view1_precision']:.6f}",
        f"  final_view2_precision={metrics['view2_precision']:.6f}",
        f"  final_mean_spill={metrics['mean_spill']:.6f}",
        f"  final_view1_spill={metrics['view1_spill']:.6f}",
        f"  final_view2_spill={metrics['view2_spill']:.6f}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()

    started = datetime.now()
    started_at = started.isoformat(timespec="seconds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from core.optimizer import fit_image_to_resolution
    from core.swept_volume import SweptVolume
    from optimizer.greedy_pack import GreedyPackConfig, run_greedy_pack

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

    # Foreground masks at the same canvas the candidates are rasterized at,
    # so pixel coverage is directly comparable to every gradient/SRD arm.
    target1_fit = fit_image_to_resolution(target1, _RESOLUTION, "cpu")
    target2_fit = fit_image_to_resolution(target2, _RESOLUTION, "cpu")
    target1_mask = (target1_fit[..., 3].numpy() > 0.5)
    target2_mask = (target2_fit[..., 3].numpy() > 0.5)

    config = GreedyPackConfig(
        panel_half_size=args.panel_half_size,
        theta=args.theta,
        max_candidates=args.max_candidates,
        view1_weight=args.view1_weight,
        view2_weight=args.view2_weight,
        spill_weight=args.spill_weight,
        min_gain=args.min_gain,
        max_panels=args.max_panels,
        handle_scale=args.handle_scale,
    )

    setup_seconds = time.perf_counter() - setup_started
    print(
        f"[Run] arm=greedy, seed={args.seed}, panel_half_size={args.panel_half_size:g}, "
        f"weights={args.view1_weight:g}/{args.view2_weight:g}, "
        f"spill_weight={args.spill_weight:g}, setup={setup_seconds:.1f}s",
        flush=True,
    )

    np.random.seed(args.seed)
    run_started = time.perf_counter()
    result = run_greedy_pack(
        swept_volume, cameras, target1_mask, target2_mask, config,
    )
    total_seconds = time.perf_counter() - run_started

    from optimizer.greedy_pack import _bisector_theta
    theta = config.theta if config.theta is not None else _bisector_theta(cameras)

    print(
        f"  [done] {result.n_candidates} candidates, {int(result.metrics['patches'])} "
        f"panels placed, mean_iou={result.metrics['mean_iou']:.6f}, "
        f"mean_spill={result.metrics['mean_spill']:.6f}, {total_seconds:.1f}s",
        flush=True,
    )

    _write_history_csv(output_dir / "history.csv", result.history)
    _write_report(
        output_dir / "report.txt", args, started_at, theta, result,
        total_seconds, setup_seconds,
    )
    (output_dir / "patches.json").write_text(
        json.dumps([p.to_dict() for p in result.patches], indent=2),
        encoding="utf-8",
    )
    print(f"[Final] report written to {output_dir / 'report.txt'}", flush=True)

    if not args.no_renders:
        stem = _render_stem(args)
        _save_mask_png(result.coverage1, output_dir / f"{stem}_view1.png")
        _save_mask_png(result.coverage2, output_dir / f"{stem}_view2.png")
        print(f"[Final] views exported to {stem}_view1.png, {stem}_view2.png", flush=True)


if __name__ == "__main__":
    main()
