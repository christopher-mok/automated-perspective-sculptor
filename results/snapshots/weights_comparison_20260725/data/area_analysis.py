"""Foreground/background pixel accounting for the target pairs, and the
area-normalized weights equivalent to the working non-normalized 2.0/2.5.

Replicates the exact loss-side pipeline: border -> fit_image_to_resolution ->
alpha channel as the foreground mask, at the SceneOptimizer default (192, 256).
"""
import sys
sys.path.insert(0, "/oscar/home/cjmok/aps-hinge2")

import numpy as np
from PIL import Image, ImageOps

from core.optimizer import fit_image_to_resolution
from submit_ablations import IMAGE_PAIRS

BORDER_FRACTION = 0.10          # _TARGET_TRANSPARENT_BORDER_FRACTION
RESOLUTION = (192, 256)         # SceneOptimizer default; run_final never overrides
W_SIL_NN = 2.0                  # the working non-normalized weights
W_NS_NN = 2.5
WEIGHT_SUM = 4.5

IMAGES = "/oscar/home/cjmok/aps-hinge2/images/"


def foreground_fraction(name):
    image = Image.open(IMAGES + name).convert("RGBA")
    border = max(1, int(round(max(image.size) * BORDER_FRACTION)))
    arr = np.array(ImageOps.expand(image, border=border, fill=(0, 0, 0, 0)))
    fit = fit_image_to_resolution(arr, RESOLUTION, "cpu")
    mask = fit[..., 3:4].clamp(0.0, 1.0)          # foreground_mask_from_image, RGBA path
    return float(mask.mean()), float(mask.sum()), mask.numel()


def equivalent_normalized(f_fg):
    """Normalized (silhouette, negative_space) matching non-normalized 2.0/2.5.

    Non-normalized silhouette averages (alpha-mask)^2 over the WHOLE frame, so
    it already contains the background spill term. Splitting it:

        L_sil_nn = f_fg * L_sil_norm + f_bg * L_ns

    so  w_s*L_sil_nn + w_n*L_ns  ==  (w_s*f_fg)*L_sil_norm + (w_s*f_bg + w_n)*L_ns
    """
    w_sil = W_SIL_NN * f_fg
    w_ns = W_SIL_NN * (1.0 - f_fg) + W_NS_NN
    return w_sil, w_ns


PAIRS = ["water_fire", "sun_moon", "cat_face_bass", "horse_circle",
         "acm_scf", "dance_argument", "teapot_droplets", "robot_man"]

print(f"{'pair':<16} {'view':<12} {'f_fg':>7} {'f_bg':>7} "
      f"{'w_sil_eq':>9} {'w_ns_eq':>9} {'ratio':>7}")
print("-" * 76)
rows = {}
for pair in PAIRS:
    t1, t2 = IMAGE_PAIRS[pair]
    fs = []
    for view, name in (("view1 " + t1.replace(".png", ""), t1),
                       ("view2 " + t2.replace(".png", ""), t2)):
        f_fg, _, _ = foreground_fraction(name)
        fs.append(f_fg)
        w_s, w_n = equivalent_normalized(f_fg)
        print(f"{pair:<16} {view:<12} {f_fg:7.4f} {1-f_fg:7.4f} "
              f"{w_s:9.4f} {w_n:9.4f} {w_s/w_n:7.4f}")
    f_mean = float(np.mean(fs))
    w_s, w_n = equivalent_normalized(f_mean)
    rows[pair] = (f_mean, w_s, w_n, w_s / w_n)
    print(f"{pair:<16} {'MEAN':<12} {f_mean:7.4f} {1-f_mean:7.4f} "
          f"{w_s:9.4f} {w_n:9.4f} {w_s/w_n:7.4f}")
    print()

print()
print("Per-pair means, sorted by silhouette:negative-space ratio")
print(f"{'pair':<16} {'f_fg':>7} {'silhouette':>11} {'neg_space':>10} {'ratio':>7} {'sum':>6}")
print("-" * 62)
for pair, (f, w_s, w_n, r) in sorted(rows.items(), key=lambda kv: kv[1][3]):
    print(f"{pair:<16} {f:7.4f} {w_s:11.4f} {w_n:10.4f} {r:7.4f} {w_s+w_n:6.3f}")

print()
print("The two pairs that look best on non-normalized original weights:")
for pair in ("water_fire", "sun_moon"):
    f, w_s, w_n, r = rows[pair]
    print(f"  {pair:<12} f_fg={f:.4f}  ->  silhouette {w_s:.4f}, "
          f"negative_space {w_n:.4f}  (ratio {r:.4f})")
both = np.mean([rows["water_fire"][0], rows["sun_moon"][0]])
w_s, w_n = equivalent_normalized(both)
print(f"  combined     f_fg={both:.4f}  ->  silhouette {w_s:.4f}, "
      f"negative_space {w_n:.4f}  (ratio {w_s/w_n:.4f})")

print()
print("Existing CONFIGS ratios for comparison (silhouette:negative_space):")
for tag, r in (("aw0p125", 0.125), ("aw0p1875", 0.1875), ("aw0p25", 0.25)):
    neg = WEIGHT_SUM / (1.0 + r)
    print(f"  {tag:<9} ratio {r:.4f}  silhouette {WEIGHT_SUM-neg:.4f}  negative_space {neg:.4f}")
