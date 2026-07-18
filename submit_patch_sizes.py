"""Submit patch size comparison jobs for all image pairs.

Runs SRD vs base comparisons at patch sizes 5, 10, 20, 30 for each pair.

Example:
    python submit_patch_sizes.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def _run_patch_comparison(
    pair_name: str,
    target1: str,
    target2: str,
    patch_sizes: list[int],
    steps: int,
    trials: int,
    output_base: Path,
) -> bool:
    """Run patch size comparison for a single pair."""
    pair_output_dir = output_base / pair_name

    print(f"\n{'='*70}")
    print(f"Running patch size comparison for: {pair_name}")
    print(f"  Targets: {target1}, {target2}")
    print(f"  Patch sizes: {patch_sizes}")
    print(f"  Steps: {steps}, Trials: {trials}")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "compare_srd_patch_sizes.py"),
        "--target1", target1,
        "--target2", target2,
        "--patch-sizes", *[str(p) for p in patch_sizes],
        "--steps", str(steps),
        "--trials", str(trials),
        "--output-dir", str(pair_output_dir),
    ]

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    success = result.returncode == 0

    if success:
        print(f"[OK] Patch size comparison for {pair_name} completed successfully")
    else:
        print(f"[ERROR] Patch size comparison for {pair_name} failed with code {result.returncode}")

    return success


def main() -> None:
    # Image pairs (same as in run_all_comparisons.py)
    pairs = [
        ("fire_water", "images/fire.png", "images/water.png"),
        ("cat_face_bass", "images/cat_face.png", "images/bass.png"),
        ("horse_circle", "images/horse.png", "images/circle.png"),
        ("sun_moon", "images/sun.png", "images/moon.png"),
        ("axe_tree", "images/axe.png", "images/tree.png"),
    ]

    # Configuration
    patch_sizes = [5, 10, 20, 30]
    steps = 1000
    trials = 3

    started = datetime.now()
    output_base = (
        _PROJECT_ROOT / "comparisons"
        / f"patch_sizes_batch_{started.strftime('%Y%m%d_%H%M%S')}"
    )
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SRD Patch Size Comparison - Batch Submission")
    print(f"{'='*70}")
    print(f"Started: {started.isoformat(timespec='seconds')}")
    print(f"Image pairs: {len(pairs)}")
    print(f"Patch sizes per pair: {patch_sizes}")
    print(f"Steps: {steps}, Trials: {trials}")
    print(f"Total comparisons: {len(pairs)} pairs × {len(patch_sizes)} sizes = {len(pairs) * len(patch_sizes)}")
    print(f"Output base: {output_base}")
    print(f"{'='*70}\n")

    run_started = time.perf_counter()
    results = []

    for pair_name, target1, target2 in pairs:
        pair_started = time.perf_counter()
        success = _run_patch_comparison(
            pair_name,
            target1,
            target2,
            patch_sizes,
            steps,
            trials,
            output_base,
        )
        pair_time = time.perf_counter() - pair_started

        results.append({
            "pair": pair_name,
            "success": success,
            "time_seconds": pair_time,
        })

    total_time = time.perf_counter() - run_started

    # Write batch summary
    summary_path = output_base / "BATCH_SUMMARY.txt"
    summary_lines = [
        "SRD Patch Size Comparison - Batch Summary",
        f"Completed: {datetime.now().isoformat(timespec='seconds')}",
        f"Started: {started.isoformat(timespec='seconds')}",
        "",
        "Configuration:",
        f"  Image pairs: {len(pairs)}",
        f"  Patch sizes: {patch_sizes}",
        f"  Steps per trial: {steps}",
        f"  Trials per configuration: {trials}",
        "",
        "Results by pair:",
    ]

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        summary_lines.append(
            f"  {result['pair']:20s} {status:7s} ({result['time_seconds']:7.1f}s)"
        )

    successful = sum(1 for r in results if r["success"])
    summary_lines.extend([
        "",
        f"Completed pairs: {successful}/{len(pairs)}",
        f"Total time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)",
        "",
        "Output structure:",
        "  comparisons/patch_sizes_batch_<timestamp>/",
        "    BATCH_SUMMARY.txt",
        "    fire_water/",
        "      SUMMARY.txt",
        "      patches_5/",
        "      patches_10/",
        "      patches_20/",
        "      patches_30/",
        "    cat_face_bass/",
        "      ...",
        "    [etc for other pairs]",
        "",
        "To analyze results:",
        "  - Check BATCH_SUMMARY.txt for overall timing",
        "  - Check each pair's SUMMARY.txt for patch-size breakdown",
        "  - Check each patches_N/comparison.txt for SRD vs base metrics",
        "  - CSV files contain loss curves for detailed analysis",
    ])

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\n{'='*70}")
    print("Batch Submission Summary")
    print(f"{'='*70}")
    print(f"Successful pairs: {successful}/{len(pairs)}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
    print(f"Batch summary: {summary_path}")
    print(f"Output base: {output_base}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
