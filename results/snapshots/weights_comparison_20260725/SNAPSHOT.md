# Weight comparison — SRD family, 2026-07-25

Collect run across every finished job. **The headline is that weights are not
the dominant variable.**

## Read this first: the lambda confound

The sweeps split into two clusters, and the split is by `--lambda-count`, not
by weights:

| `--lambda-count` | sweeps | pieces | mean IoU |
|---|---|---|---|
| **0** | minarea_nonorm4, aw0p1875_lam0, awrec, aw0p25_lam0, ORIGINAL | 28–30 | 0.88–0.92 |
| 0.05 (default) | neg_20260724, restart, importnet, aw0p125, srd_nonorm1p25 | 16–17 | 0.77–0.85 |

`--lambda-count` is the per-piece penalty in SRD's acceptance test: at 0.05
every add must beat the loss by more than the penalty, so adds are rejected and
runs settle at ~17 pieces instead of ~29, losing 5–10 IoU points. That effect is
larger than any weight change measured here. **Any ranking that pools the two
clusters is measuring lambda, not weights** — the pooled table in
`data/pooled_and_branch_comparison.txt` is included only to show that.

Everything below is restricted to the `lambda-count 0` family.

## Weights, matched

On the three pairs every `lambda-count 0` sweep has finished — cat_face_bass,
sun_moon, water_fire. "eff. ratio" is the normalized-equivalent
silhouette:negative-space ratio at f_fg = 0.153, which puts normalized and
non-normalized configs on one axis (derivation in
`data/foreground_area_analysis.txt`).

| config | eff. ratio | arm | IoU | spill | coverage | precision | pieces |
|---|---|---|---|---|---|---|---|
| **aw0p1875** | 0.188 | srd_restart | **0.9608** | 0.0220 | 0.982 | 0.978 | 26.7 |
| nonorm1p25 *(ORIGINAL branch)* | 0.073 | srd_restart | 0.9566 | 0.0200 | 0.976 | 0.980 | 26.3 |
| aw0p25 | 0.250 | srd_restart | 0.9539 | 0.0338 | 0.986 | 0.967 | 25.3 |
| aw0p073 | 0.073 | **srd (no restart)** | 0.9423 | 0.0219 | 0.963 | 0.978 | 27.0 |
| nonorm4 | 0.032 | srd_restart | 0.9346 | **0.0084** | 0.942 | **0.991** | 26.0 |

The tradeoff is monotone in the effective ratio: more silhouette weight buys
coverage and costs spill. `nonorm4` at 0.032 has the lowest spill in the whole
study and the lowest IoU; `aw0p25` at 0.250 has the most coverage and the worst
spill. **`aw0p1875` is the best IoU/spill compromise.**

Per-pair, best in each column starred:

| pair | aw0p073 | aw0p1875 | aw0p25 | nonorm1p25 | nonorm4 |
|---|---|---|---|---|---|
| cat_face_bass | 0.9477 | **0.9504\*** | 0.9392 | 0.9501 | 0.9452 |
| sun_moon | 0.9320 | **0.9685\*** | 0.9496 | 0.9511 | 0.9225 |
| water_fire | 0.9472 | 0.9636 | **0.9728\*** | 0.9685 | 0.9360 |

Spill, same runs:

| pair | aw0p073 | aw0p1875 | aw0p25 | nonorm1p25 | nonorm4 |
|---|---|---|---|---|---|
| cat_face_bass | 0.0144 | 0.0288 | 0.0448 | 0.0229 | **0.0103\*** |
| sun_moon | 0.0268 | 0.0152 | 0.0357 | 0.0273 | **0.0087\*** |
| water_fire | 0.0245 | 0.0220 | 0.0210 | 0.0099 | **0.0060\*** |

### aw0p073 did not win, and the test wasn't clean

`aw0p073` is the config derived from the pixel analysis to reproduce what
non-normalized 2.0/2.5 computes. It placed 4th of 5 on IoU. It matches the
original weights on spill (0.0219 vs 0.0200) and precision (0.978 vs 0.980) but
trails on IoU (0.9423 vs 0.9566), so the derivation is half-confirmed at best.

The caveat is real: **`aw0p073` is the only arm here without conflict restart**
(it was submitted as plain full SRD by request, `--arm srd`), and every other
row is `srd_restart`. The gap may be the missing restart rather than the
weights. Only 5 of its 8 jobs have finished.

## Original branch vs main — not answerable yet

The naive comparison looks decisive. ORIGINAL beats `main/neg_srd_nonorm1p25` on
all six shared pairs:

| pair | ORIGINAL | main | ORIGINAL pieces | main pieces |
|---|---|---|---|---|
| water_fire | 0.9685 | 0.8562 | 24 | 17 |
| sun_moon | 0.9511 | 0.8636 | 30 | 16 |
| cat_face_bass | 0.9501 | 0.8966 | 25 | 16 |
| horse_circle | 0.9067 | 0.8692 | 27 | 18 |
| teapot_droplets | 0.8456 | 0.7202 | 33 | 19 |
| acm_scf | 0.8029 | 0.6536 | 31 | 19 |

But that main sweep runs `lambda-count 0.05` and `--arm srd` with no restart.
The piece counts give it away — 16–19 against 24–33. That is the lambda
confound plus an arm difference, not a branch difference.

On the matched table above, ORIGINAL's `nonorm1p25` places **2nd of 5, within
0.4 points of the winner** — competitive with main, not ahead of it.

**No main-branch run exists at ORIGINAL's exact config** (nonorm1p25 +
srd_restart + lam0 + minarea 0.005 + swept 300 + 64 candidates), so branch
effect and settings cannot be separated from what has been run. That one config
on main, 8 jobs, is the only thing that would answer the question.

## Baselines, no SRD (both 11/11 complete)

| | mean IoU | best | worst |
|---|---|---|---|
| 20 pieces | **0.7951** | water_fire 0.9218 | teapot_droplets 0.6369 |
| 5 pieces | **0.5583** | horse_circle 0.8450 | siggraph_sigchi 0.2728 |

Dropping 20→5 pieces costs **24 IoU points**. horse_circle degrades most
gracefully (0.870→0.845, 2.5 points — a circle is nearly a single piece);
siggraph_sigchi and acm_scf collapse (−50 and −36). All 22 converged on plateau,
none hit the ceiling.

Note the 20-piece *no-SRD* baseline at 0.795 beats several full-SRD sweeps in
the lambda-0.05 cluster. More evidence that lambda was crippling those runs.

## Limits

The matched set is **3 pairs, and they are the three easiest**. acm_scf,
teapot_droplets, dance_argument and horse_circle are not yet common to all five
sweeps. Absolute numbers are optimistic and the ranking may shift as the hard
pairs land. 10 jobs were still running when this was cut.

## Contents

- `weights/` — final exported views, `<pair>__<config>_view<N>.png`, 3 pairs x
  5 configs x 2 views
- `baseline/` — final views for all 11 pairs at 20 and 5 pieces,
  `<pair>__<20pc|5pc>_view<N>.png`
- `data/` — the three analysis outputs as text, the scripts that produced them
  (including `verify_decomposition.py`, which checks the normalization algebra
  numerically against the real loss functions), and the raw `summary.tsv` for
  every sweep used

---

## Visual: weights, view 1

Ordered left to right by effective ratio, most negative-space-weighted first.

| pair | nonorm4 (0.032) | aw0p073 (0.073) | nonorm1p25 ORIG (0.073) | aw0p1875 (0.188) | aw0p25 (0.250) |
|---|---|---|---|---|---|
| cat_face_bass | ![](weights/cat_face_bass__nonorm4_view1.png) | ![](weights/cat_face_bass__aw0p073_view1.png) | ![](weights/cat_face_bass__nonorm1p25-ORIGINAL_view1.png) | ![](weights/cat_face_bass__aw0p1875_view1.png) | ![](weights/cat_face_bass__aw0p25_view1.png) |
| sun_moon | ![](weights/sun_moon__nonorm4_view1.png) | ![](weights/sun_moon__aw0p073_view1.png) | ![](weights/sun_moon__nonorm1p25-ORIGINAL_view1.png) | ![](weights/sun_moon__aw0p1875_view1.png) | ![](weights/sun_moon__aw0p25_view1.png) |
| water_fire | ![](weights/water_fire__nonorm4_view1.png) | ![](weights/water_fire__aw0p073_view1.png) | ![](weights/water_fire__nonorm1p25-ORIGINAL_view1.png) | ![](weights/water_fire__aw0p1875_view1.png) | ![](weights/water_fire__aw0p25_view1.png) |

## Visual: weights, view 2

| pair | nonorm4 (0.032) | aw0p073 (0.073) | nonorm1p25 ORIG (0.073) | aw0p1875 (0.188) | aw0p25 (0.250) |
|---|---|---|---|---|---|
| cat_face_bass | ![](weights/cat_face_bass__nonorm4_view2.png) | ![](weights/cat_face_bass__aw0p073_view2.png) | ![](weights/cat_face_bass__nonorm1p25-ORIGINAL_view2.png) | ![](weights/cat_face_bass__aw0p1875_view2.png) | ![](weights/cat_face_bass__aw0p25_view2.png) |
| sun_moon | ![](weights/sun_moon__nonorm4_view2.png) | ![](weights/sun_moon__aw0p073_view2.png) | ![](weights/sun_moon__nonorm1p25-ORIGINAL_view2.png) | ![](weights/sun_moon__aw0p1875_view2.png) | ![](weights/sun_moon__aw0p25_view2.png) |
| water_fire | ![](weights/water_fire__nonorm4_view2.png) | ![](weights/water_fire__aw0p073_view2.png) | ![](weights/water_fire__nonorm1p25-ORIGINAL_view2.png) | ![](weights/water_fire__aw0p1875_view2.png) | ![](weights/water_fire__aw0p25_view2.png) |

## Visual: piece count, no SRD

20 pieces against 5, view 1 then view 2, sorted by how much the drop costs.

| pair | 20pc v1 | 5pc v1 | 20pc v2 | 5pc v2 | IoU 20→5 |
|---|---|---|---|---|---|
| horse_circle | ![](baseline/horse_circle__20pc_view1.png) | ![](baseline/horse_circle__5pc_view1.png) | ![](baseline/horse_circle__20pc_view2.png) | ![](baseline/horse_circle__5pc_view2.png) | 0.870 → 0.845 |
| cat_face_bass | ![](baseline/cat_face_bass__20pc_view1.png) | ![](baseline/cat_face_bass__5pc_view1.png) | ![](baseline/cat_face_bass__20pc_view2.png) | ![](baseline/cat_face_bass__5pc_view2.png) | 0.908 → 0.730 |
| robot_man | ![](baseline/robot_man__20pc_view1.png) | ![](baseline/robot_man__5pc_view1.png) | ![](baseline/robot_man__20pc_view2.png) | ![](baseline/robot_man__5pc_view2.png) | 0.905 → 0.736 |
| water_fire | ![](baseline/water_fire__20pc_view1.png) | ![](baseline/water_fire__5pc_view1.png) | ![](baseline/water_fire__20pc_view2.png) | ![](baseline/water_fire__5pc_view2.png) | 0.922 → 0.683 |
| dancer_guitar | ![](baseline/dancer_guitar__20pc_view1.png) | ![](baseline/dancer_guitar__5pc_view1.png) | ![](baseline/dancer_guitar__20pc_view2.png) | ![](baseline/dancer_guitar__5pc_view2.png) | 0.763 → 0.730 |
| crane_crab | ![](baseline/crane_crab__20pc_view1.png) | ![](baseline/crane_crab__5pc_view1.png) | ![](baseline/crane_crab__20pc_view2.png) | ![](baseline/crane_crab__5pc_view2.png) | 0.723 → 0.562 |
| sun_moon | ![](baseline/sun_moon__20pc_view1.png) | ![](baseline/sun_moon__5pc_view1.png) | ![](baseline/sun_moon__20pc_view2.png) | ![](baseline/sun_moon__5pc_view2.png) | 0.810 → 0.472 |
| dance_argument | ![](baseline/dance_argument__20pc_view1.png) | ![](baseline/dance_argument__5pc_view1.png) | ![](baseline/dance_argument__20pc_view2.png) | ![](baseline/dance_argument__5pc_view2.png) | 0.792 → 0.402 |
| teapot_droplets | ![](baseline/teapot_droplets__20pc_view1.png) | ![](baseline/teapot_droplets__5pc_view1.png) | ![](baseline/teapot_droplets__20pc_view2.png) | ![](baseline/teapot_droplets__5pc_view2.png) | 0.637 → 0.425 |
| acm_scf | ![](baseline/acm_scf__20pc_view1.png) | ![](baseline/acm_scf__5pc_view1.png) | ![](baseline/acm_scf__20pc_view2.png) | ![](baseline/acm_scf__5pc_view2.png) | 0.640 → 0.284 |
| siggraph_sigchi | ![](baseline/siggraph_sigchi__20pc_view1.png) | ![](baseline/siggraph_sigchi__5pc_view1.png) | ![](baseline/siggraph_sigchi__20pc_view2.png) | ![](baseline/siggraph_sigchi__5pc_view2.png) | 0.777 → 0.273 |
