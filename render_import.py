#!/usr/bin/env python
"""Render a UI-exported design JSON to its two camera views.

``run_final.py`` writes view1/view2 PNGs at the end of an optimization run.
This does the same for a design that came out of the UI instead of out of an
optimization: it loads the pieces from an export JSON, places the same two
scene cameras the UI and the runs use, and writes both rendered views.

Because the cameras and the resolution match ``run_final.py``'s exported
views, an imported design and an optimized one can be put side by side and the
difference read as geometry rather than as framing.

nvdiffrast rasterizes on the GPU, so this has to run on a GPU node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_ablation import _make_scene_cameras

# The optimization resolution the runs use. --render-scale multiplies it, so
# the default of 4 gives the same 1024x768 exports run_final.py produces.
BASE_RESOLUTION = (192, 256)  # (height, width)


def _save_render(render, path: Path) -> None:
    """Write an RGBA render tensor/array to a PNG."""
    from PIL import Image

    array = np.asarray(render)
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGBA").save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "designs",
        nargs="+",
        help="Export JSON files to render, e.g. imports/horse1.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="imports/exports/rerender",
        help="Directory for the rendered PNGs. Defaults to a subfolder so a "
             "re-render does not overwrite the committed exports it is meant "
             "to be compared against.",
    )
    parser.add_argument(
        "--render-scale",
        type=int,
        default=4,
        help="Resolution multiplier relative to the 192x256 optimization "
             "resolution, matching run_final.py's --render-scale.",
    )
    parser.add_argument(
        "--n-per-segment",
        type=int,
        default=20,
        help="Bezier samples per spline segment. Higher is smoother; 20 is "
             "what the optimization renders with.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for the renderer.",
    )
    parser.add_argument(
        "--fan-fill",
        action="store_true",
        help="Force vertex-0 fan fill instead of earcut triangulation, which "
             "over-covers concave pieces. Without mapbox-earcut installed the "
             "renderer falls back to this anyway; passing it makes the "
             "geometry a property of the render rather than of the "
             "environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import torch

    from core.export import import_patches_from_json
    from core.patch import set_fan_fill
    from core.renderer import DiffRenderer

    # Before any patch is built, so the triangulation choice applies to the
    # geometry this run imports.
    if args.fan_fill:
        set_fan_fill(True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. nvdiffrast rasterizes on the GPU, so this "
            "has to run on a GPU node (sbatch slurm/render_imports.sh)."
        )

    scale = max(1, args.render_scale)
    resolution = (BASE_RESOLUTION[0] * scale, BASE_RESOLUTION[1] * scale)
    cameras = _make_scene_cameras()
    renderer = DiffRenderer(device=args.device, n_per_segment=args.n_per_segment)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for design in args.designs:
        path = Path(design)
        patches = import_patches_from_json(path, device=args.device)
        with torch.no_grad():
            render1, render2 = renderer.render_both(
                patches, cameras[0], cameras[1], resolution
            )
        view1_path = output_dir / f"{path.stem}_view1.png"
        view2_path = output_dir / f"{path.stem}_view2.png"
        _save_render(render1.detach().cpu(), view1_path)
        _save_render(render2.detach().cpu(), view2_path)
        print(
            f"[render] {path.name}: {len(patches)} pieces -> "
            f"{view1_path}, {view2_path} "
            f"({resolution[1]}x{resolution[0]})",
            flush=True,
        )


if __name__ == "__main__":
    main()
