# hinge2 + restart at aw0p1875 — `neg_h2r_aw0p1875_20260725`

Arm `hinge2_restart`: the fit-then-shed count objective (`--count-objective`,
count_lambda 500, count_target_iou 0.91, FULL_STACK scheduling) with
conflict-gated restart. **No importance deletion** — that is the difference from
`neg_h2ri_nonorm1p25_20260725`, along with the weights.

Config `aw0p1875`: `--area-normalized-view-loss`, silhouette 0.7105 / negative
space 3.789, effective ratio 0.188. Swept resolution 300, 64 SRD candidates,
`--srd-min-patch-area 0.005`, `--lambda-count 0`, 20 starting patches, planar
overlap, constrained theta, seed 0, plateau stop at min 1500 / patience 300.

**Complete, 7/7.** Same seven pairs as the h2ri batch.

## Final IoU and piece count

| pair | final IoU | best IoU | final pieces | start | max | spill | step | stop |
|---|---|---|---|---|---|---|---|---|
| water_fire | 0.9180 | 0.9544 | **7** | 20 | 28 | 0.0343 | 1525 | shed_converged |
| cat_face_bass | 0.9154 | 0.9363 | **11** | 20 | 29 | 0.0431 | 1525 | shed_converged |
| robot_man | 0.9125 | 0.9378 | **10** | 20 | 30 | 0.0561 | 1500 | shed_converged |
| sun_moon | 0.9113 | 0.9504 | **16** | 20 | 33 | 0.0324 | 2025 | shed_converged |
| dance_argument | 0.8788 | 0.8798 | **31** | 20 | 33 | 0.0910 | 1950 | converged |
| teapot_droplets | 0.8249 | 0.8330 | **27** | 20 | 28 | 0.1574 | 1675 | converged |
| acm_scf | 0.8118 | 0.8173 | **28** | 20 | 32 | 0.1148 | 1500 | converged |
| **mean** | **0.8818** | 0.9013 | **18.6** | 20 | | 0.0756 | | |

The same two-population split as h2ri, and on exactly the same four pairs. Four
runs reached the target IoU, entered shedding and finished **below** their
starting count (7–16 pieces, mean 11); three never reached it, so shedding never
began and they ended at 27–31.

Note the gap between final and best IoU in the shed group — water_fire peaks at
0.9544 and finishes at 0.9180, sun_moon 0.9504 → 0.9113. **Shedding costs 2–4
points of IoU to buy the lower piece count.** That is the objective working as
specified, but it means "final IoU" understates these runs' peak quality by more
than the differences between weight configs.

## Against h2ri (nonorm1p25 + importance deletion)

| pair | aw0p1875 IoU | pieces | h2ri IoU | pieces | ΔIoU |
|---|---|---|---|---|---|
| water_fire | 0.9180 | 7 | 0.9353 | 9 | −0.0173 |
| cat_face_bass | 0.9154 | 11 | 0.9318 | 10 | −0.0164 |
| robot_man | 0.9125 | 10 | 0.9151 | 11 | −0.0026 |
| sun_moon | 0.9113 | 16 | 0.9139 | 18 | −0.0026 |
| dance_argument | 0.8788 | 31 | 0.8767 | 29 | +0.0021 |
| teapot_droplets | 0.8249 | 27 | 0.7987 | 35 | **+0.0262** |
| acm_scf | 0.8118 | 28 | 0.8913 | 44 | **−0.0795** |
| **mean** | **0.8818** | **18.6** | **0.8947** | **22.3** | **−0.0129** |

h2ri leads by 1.3 points on average, but **this comparison moves two variables at
once** — weights (aw0p1875 vs nonorm1p25) and importance deletion (absent vs
present). Nothing here attributes the difference to either.

acm_scf drives most of it: −8.0 points, and h2ri got there with 44 pieces against
28. Excluding acm_scf the means are within 0.2 points. On the other side,
aw0p1875 reaches a **lower mean piece count** (18.6 vs 22.3) on every shed pair
but cat_face_bass, so it is shedding harder for slightly less IoU.

## Contents

- `collected/summary.tsv` — 27 metrics per run
- `collected/curves.csv` — per-step traces
- `collected/by_config_arm.tsv` — means over pairs
- `collected/views/` — final exported renders, 14 PNGs (7 pairs x 2 views)
- `videos/<pair>.mp4` — the full optimization as video, both views side by side
  (1024x384, h264, 20 fps), assembled from the `--render-every 10` frames
- `manifest.tsv` — job IDs, weights, output dirs

The 2350 source frames stay on Oscar under each run's `renders/`; the videos
carry the same content at about a tenth the size.

### Rebuilding the videos

```
module load ffmpeg/7.1-7dmq
ffmpeg -framerate 20 \
  -pattern_type glob -i '<run>/renders/step*_view1.png' \
  -pattern_type glob -i '<run>/renders/step*_view2.png' \
  -filter_complex "[0:v][1:v]hstack=inputs=2" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p videos/<pair>.mp4
```
