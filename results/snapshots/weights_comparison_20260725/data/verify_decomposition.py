"""Check the claim empirically with the real loss functions.

Claim:  L_sil_nonnorm  ==  f_fg * L_sil_norm + f_bg * L_negspace
so that w_s*L_sil_nn + w_n*L_ns  ==  (w_s*f_fg)*L_sil_norm + (w_s*f_bg + w_n)*L_ns

Tested on real target masks against several plausible alpha fields, not just
one, so a coincidence at a single operating point can't pass.
"""
import sys
sys.path.insert(0, "/oscar/home/cjmok/aps-hinge2")

import numpy as np
import torch
from PIL import Image, ImageOps

from core.loss import silhouette_loss, negative_space_loss
from core.optimizer import fit_image_to_resolution

IMAGES = "/oscar/home/cjmok/aps-hinge2/images/"
W_S, W_N = 2.0, 2.5


def load_mask(name):
    im = Image.open(IMAGES + name).convert("RGBA")
    b = max(1, int(round(max(im.size) * 0.10)))
    arr = np.array(ImageOps.expand(im, border=b, fill=(0, 0, 0, 0)))
    return fit_image_to_resolution(arr, (192, 256), "cpu")[..., 3:4].clamp(0, 1)


torch.manual_seed(0)
print(f"{'target':<12} {'alpha field':<22} {'L_sil_nn':>10} {'predicted':>10} {'rel.err':>9}")
print("-" * 68)
worst = 0.0
for name in ("water.png", "fire.png", "sun.png", "moon.png", "robot.png", "man.png"):
    mask = load_mask(name)
    f_fg = float(mask.mean())
    f_bg = 1.0 - f_fg
    trials = {
        "perfect match": mask.clone(),
        "empty scene": torch.zeros_like(mask),
        "full frame": torch.ones_like(mask),
        "half-strength fg": mask * 0.5,
        "fg + 0.2 spill": (mask + 0.2 * (1 - mask)).clamp(0, 1),
        "random uniform": torch.rand_like(mask),
        "shifted mask": torch.roll(mask, shifts=12, dims=1),
    }
    for label, alpha in trials.items():
        rgba = torch.cat([alpha.expand(-1, -1, 3), alpha], dim=-1)
        l_nn = float(silhouette_loss(rgba, mask, False))
        l_norm = float(silhouette_loss(rgba, mask, True))
        l_ns = float(negative_space_loss(rgba, mask))
        pred = f_fg * l_norm + f_bg * l_ns
        err = abs(l_nn - pred) / max(l_nn, 1e-9)
        worst = max(worst, err)
        print(f"{name:<12} {label:<22} {l_nn:10.6f} {pred:10.6f} {err:9.2e}")
    print()

print(f"worst relative error across all cases: {worst:.3e}")
print()

# And the weighted-total identity, which is the thing actually being claimed.
print("Weighted total: non-normalized 2.0/2.5  vs  equivalent normalized weights")
print(f"{'target':<12} {'alpha field':<22} {'nonnorm tot':>12} {'norm tot':>12} {'rel.err':>9}")
print("-" * 72)
worst_tot = 0.0
for name in ("water.png", "sun.png", "man.png"):
    mask = load_mask(name)
    f_fg = float(mask.mean())
    w_s_eq = W_S * f_fg
    w_n_eq = W_S * (1 - f_fg) + W_N
    for label, alpha in {
        "fg + 0.2 spill": (mask + 0.2 * (1 - mask)).clamp(0, 1),
        "random uniform": torch.rand_like(mask),
        "shifted mask": torch.roll(mask, shifts=12, dims=1),
        "half-strength fg": mask * 0.5,
    }.items():
        rgba = torch.cat([alpha.expand(-1, -1, 3), alpha], dim=-1)
        total_nn = W_S * float(silhouette_loss(rgba, mask, False)) \
            + W_N * float(negative_space_loss(rgba, mask))
        total_n = w_s_eq * float(silhouette_loss(rgba, mask, True)) \
            + w_n_eq * float(negative_space_loss(rgba, mask))
        err = abs(total_nn - total_n) / max(total_nn, 1e-9)
        worst_tot = max(worst_tot, err)
        print(f"{name:<12} {label:<22} {total_nn:12.6f} {total_n:12.6f} {err:9.2e}")
print()
print(f"worst relative error on the weighted total: {worst_tot:.3e}")
