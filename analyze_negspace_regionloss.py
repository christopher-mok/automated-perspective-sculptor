"""Post-hoc pixel-level metrics for the region-loss ablation.

report.txt / history.csv already carry final_mean_iou, final_mean_spill
(false positive / target FOREGROUND area), final_mean_coverage, final_patches,
final_step and total_seconds for every run submit_negspace.py collects. Three
metrics the region-loss comparison asked for are not among them, because they
need the actual rendered pixels rather than the training-time logged floats:

  * false-positive area normalized by target BACKGROUND area (spill in
    report.txt is normalized by target FOREGROUND area instead -- a different
    denominator, not interchangeable with what was asked for here);
  * false-negative area normalized by target foreground area (equal to
    1 - coverage, but recomputed from pixels for consistency with the other
    two rather than pulled from a different column);
  * a hole-preservation score: the target's background, thresholded and
    connected-component-labeled, contains one component touching the image
    border (the true exterior) and, for shapes like the horse's legs or the
    teapot's handle, zero or more *interior* components -- holes the
    silhouette should stay out of. Restricted to just those interior pixels,
    every one of them is target background by construction, so the ordinary
    two-class IoU degenerates to 0 whenever anything spills in and is
    undefined when nothing does -- not informative. What is informative is
    the background-class version of the same ratio, i.e. how much of the
    hole's own area the render still leaves empty:

        hole_preservation = 1 - (rendered pixels inside the hole) / (hole area)

    which is exactly "IoU of the background class, restricted to interior
    background components" once you note the target's background covers the
    hole region entirely, so intersection = hole_area - spill and union =
    hole_area regardless of how much spills in. A target with no interior
    holes (e.g. a plain disc) reports NaN, not 0 or 1.

Run this against a submit_negspace.py sweep directory after the jobs finish:

    python analyze_negspace_regionloss.py --sweep-dir results/negspace/<name>

Writes <sweep-dir>/collected/region_loss_metrics.csv (one row per job) and
prints a mean +/- std summary grouped by (config, pair).
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_BORDER_FRACTION = 0.10  # must match run_ablation._TARGET_TRANSPARENT_BORDER_FRACTION

_REPORT_KEYS = (
    "arm", "seed", "view_loss", "tversky_alpha", "tversky_beta",
    "target1", "target2", "stop_reason", "final_step",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "final_patches", "final_mean_spill", "final_mean_coverage",
    "total_seconds",
)


def _grab(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _pair_config_seed(job_name: str) -> tuple[str, str, str]:
    """'neg_<pair>_<arm>_<config>[_seed<N>]' -> (pair, config, seed_suffix)."""
    from submit_ablations import IMAGE_PAIRS
    from submit_negspace import ARMS, CONFIG_TAGS

    stripped = re.sub(r"^neg_", "", job_name)
    seed_match = re.search(r"_seed(\d+)$", stripped)
    seed_suffix = seed_match.group(1) if seed_match else None
    if seed_match:
        stripped = stripped[: seed_match.start()]
    for config in sorted(CONFIG_TAGS, key=len, reverse=True):
        if stripped.endswith(f"_{config}"):
            rest = stripped[: -len(config) - 1]
            for arm in sorted(ARMS, key=len, reverse=True):
                if rest.endswith(f"_{arm}") and rest[: -len(arm) - 1] in IMAGE_PAIRS:
                    return rest[: -len(arm) - 1], config, seed_suffix
    return "-", "-", seed_suffix


def _load_target_mask(path: str, render_shape: tuple[int, int]) -> np.ndarray:
    """Reproduce optimizer.fit_image_to_resolution + foreground_mask_from_image
    on the CPU, at the resolution of an already-rendered PNG, so the mask
    lines up pixel-for-pixel with the render it is compared against.
    """
    image = Image.open(path).convert("RGBA")
    border = max(1, round(max(image.size) * _BORDER_FRACTION))
    image = ImageOps.expand(image, border=border, fill=(0, 0, 0, 0))
    arr = np.asarray(image).astype(np.float32) / 255.0

    target_h, target_w = render_shape
    src_h, src_w = arr.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = np.asarray(
        Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGBA")
        .resize((new_w, new_h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    canvas = np.zeros((target_h, target_w, 4), dtype=np.float32)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return np.clip(canvas[..., 3], 0.0, 1.0)


def _interior_hole_mask(target_binary: np.ndarray) -> np.ndarray:
    """Background pixels not 4-connected to the image border (the "holes").

    No scipy in this environment, so this floods the border-touching
    background component outward with a vectorized dilate-and-mask loop
    rather than calling ndimage.label -- the union of every *other*
    background component (there is no need to tell separate holes apart,
    only to know which background pixels are in one) is exactly what
    "target background components, dropping the one touching the border" asks
    for.
    """
    background = ~target_binary
    reached = np.zeros_like(background)
    reached[0, :] = background[0, :]
    reached[-1, :] = background[-1, :]
    reached[:, 0] = background[:, 0]
    reached[:, -1] = background[:, -1]
    while True:
        dilated = reached.copy()
        dilated[1:, :] |= reached[:-1, :]
        dilated[:-1, :] |= reached[1:, :]
        dilated[:, 1:] |= reached[:, :-1]
        dilated[:, :-1] |= reached[:, 1:]
        dilated &= background
        if np.array_equal(dilated, reached):
            break
        reached = dilated
    return background & ~reached


def _view_metrics(render_path: Path, target_path: str) -> dict[str, float]:
    render = np.asarray(Image.open(render_path).convert("RGBA")).astype(np.float32) / 255.0
    render_alpha = render[..., 3]
    render_binary = render_alpha > 0.5

    target_mask = _load_target_mask(target_path, render_alpha.shape)
    target_binary = target_mask > 0.5

    intersection = np.logical_and(render_binary, target_binary).sum()
    union = np.logical_or(render_binary, target_binary).sum()
    target_fg = int(target_binary.sum())
    target_bg = int((~target_binary).sum())
    fp = int(np.logical_and(render_binary, ~target_binary).sum())
    fn = int(np.logical_and(~render_binary, target_binary).sum())

    hole_mask = _interior_hole_mask(target_binary)
    hole_area = int(hole_mask.sum())
    if hole_area > 0:
        spilled_in_hole = int(np.logical_and(render_binary, hole_mask).sum())
        hole_preservation = 1.0 - spilled_in_hole / hole_area
    else:
        hole_preservation = float("nan")

    return {
        "recomputed_iou": intersection / union if union else 0.0,
        "fp_over_background": fp / target_bg if target_bg else float("nan"),
        "fn_over_foreground": fn / target_fg if target_fg else float("nan"),
        "hole_area": hole_area,
        "hole_preservation_iou": hole_preservation,
    }


def _stuck_at_init(job_dir: Path) -> bool | None:
    """True if a run's mean IoU barely moved from its first logged eval.

    A region-based loss has weak gradients far from any overlap, so a run
    that never leaves its random initialization is a possible outcome to
    report rather than a bug to chase. None if history.csv is missing.
    """
    history = job_dir / "history.csv"
    if not history.exists():
        return None
    lines = [line for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    header = lines[0].split(",")
    if "mean_iou" not in header:
        return None
    idx = header.index("mean_iou")
    first_iou = float(lines[1].split(",")[idx])
    last_iou = float(lines[-1].split(",")[idx])
    return (last_iou - first_iou) < 0.02


def _find_view(job_dir: Path, view: str) -> Path | None:
    matches = sorted(job_dir.glob(f"*_{view}.png"))
    return matches[0] if matches else None


def _collect(sweep_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for report_path in sorted(sweep_dir.glob("*/report.txt")):
        job_dir = report_path.parent
        text = report_path.read_text(encoding="utf-8")
        pair, config, seed_suffix = _pair_config_seed(job_dir.name)
        row: dict = {"job": job_dir.name, "pair": pair, "config": config}
        for key in _REPORT_KEYS:
            row[key] = _grab(text, key)
        if seed_suffix is not None:
            row["seed"] = seed_suffix

        target1, target2 = row.get("target1"), row.get("target2")
        for view, target in (("view1", target1), ("view2", target2)):
            render_path = _find_view(job_dir, view)
            if render_path is None or target is None:
                for metric in (
                    "recomputed_iou", "fp_over_background",
                    "fn_over_foreground", "hole_area", "hole_preservation_iou",
                ):
                    row[f"{metric}_{view}"] = "-"
                continue
            metrics = _view_metrics(render_path, target)
            for metric, value in metrics.items():
                row[f"{metric}_{view}"] = value

        stuck = _stuck_at_init(job_dir)
        row["stuck_at_init"] = "-" if stuck is None else str(stuck)
        rows.append(row)
    return rows


_CSV_COLUMNS = (
    "job", "pair", "config", "arm", "seed", "view_loss", "tversky_alpha", "tversky_beta",
    "stop_reason", "final_step", "total_seconds", "stuck_at_init",
    "final_mean_iou", "final_view1_iou", "final_view2_iou", "final_patches",
    "final_mean_spill", "final_mean_coverage",
    "recomputed_iou_view1", "recomputed_iou_view2",
    "fp_over_background_view1", "fp_over_background_view2",
    "fn_over_foreground_view1", "fn_over_foreground_view2",
    "hole_area_view1", "hole_area_view2",
    "hole_preservation_iou_view1", "hole_preservation_iou_view2",
)

_SUMMARY_METRICS = (
    "final_mean_iou", "final_patches", "final_step", "total_seconds",
    "fp_over_background_mean", "fn_over_foreground_mean", "hole_preservation_iou_mean",
)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _to_float(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _write_csv(rows: list[dict], path: Path) -> None:
    lines = ["\t".join(_CSV_COLUMNS)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(col)) for col in _CSV_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_pair(row: dict, prefix: str) -> float | None:
    values = [_to_float(row.get(f"{prefix}_view1")), _to_float(row.get(f"{prefix}_view2"))]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["config"], row["pair"]), []).append(row)

    summary: list[dict] = []
    for (config, pair), members in groups.items():
        entry: dict = {"config": config, "pair": pair, "n": len(members)}
        for metric in _SUMMARY_METRICS:
            if metric.endswith("_mean"):
                prefix = metric[: -len("_mean")]
                values = [v for v in (_mean_pair(m, prefix) for m in members) if v is not None]
            else:
                values = [v for v in (_to_float(m.get(metric)) for m in members) if v is not None]
            if values:
                mean = statistics.fmean(values)
                std = statistics.pstdev(values) if len(values) > 1 else 0.0
                entry[metric] = f"{mean:.4g} +/- {std:.4g}"
            else:
                entry[metric] = "-"
        stuck = [m["stuck_at_init"] for m in members if m["stuck_at_init"] in ("True", "False")]
        entry["stuck_at_init"] = f"{stuck.count('True')}/{len(stuck)}" if stuck else "-"
        summary.append(entry)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = _collect(args.sweep_dir)
    if not rows:
        print(f"No report.txt files found under {args.sweep_dir}")
        return

    collected = args.sweep_dir / "collected"
    collected.mkdir(parents=True, exist_ok=True)
    csv_path = collected / "region_loss_metrics.csv"
    _write_csv(rows, csv_path)

    summary = _summary(rows)
    config_order = {c: i for i, c in enumerate(("aw0p1875", "softiou", "tversky_r5p33", "tversky_sym", "tversky_r2"))}
    summary.sort(key=lambda e: (config_order.get(e["config"], 99), e["pair"]))

    print(f"{'config':<15} {'pair':<16} {'n':>2} {'stuck':>6} {'iou':>16} {'panels':>16} "
          f"{'step':>16} {'sec':>16} {'fp/bg':>16} {'fn/fg':>16} {'hole_pres':>16}")
    for entry in summary:
        print(
            f"{entry['config']:<15} {entry['pair']:<16} {entry['n']:>2} "
            f"{entry['stuck_at_init']:>6} "
            f"{entry['final_mean_iou']:>16} {entry['final_patches']:>16} "
            f"{entry['final_step']:>16} {entry['total_seconds']:>16} "
            f"{entry['fp_over_background_mean']:>16} {entry['fn_over_foreground_mean']:>16} "
            f"{entry['hole_preservation_iou_mean']:>16}"
        )

    print(f"\nWritten: {csv_path}")


if __name__ == "__main__":
    main()
