# Baseline batches (no SRD) — 2026-07-25

Three `--arm basic` sweeps, all complete. No SRD means no adds, splits or
deletes: piece count is fixed at initialization and every run ends where it
started.

Shared settings: original weights (silhouette 2.0 / negative space 2.5,
non-normalized, via `--weight-ratio 0.8 --weight-sum 4.5`), planar overlap,
`--render-every 10 --render-every-scale 2`, early stop on plateau at min 1500 /
patience 300, 20000-step ceiling, seed 0.

| sweep | pieces | pairs | jobs | swept res |
|---|---|---|---|---|
| `baseline_20pc_20260725` | 20 | all 11 | 11/11 | 256 |
| `baseline_5pc_20260725` | 5 | all 11 | 11/11 | 256 |
| `baseline_h2pc_20260725` | per-pair, seeded from hinge2 | 7 | 7/7 | 300 |

Every job in all three converged on plateau; none hit the step ceiling.

## Piece count sweep, 20 vs 5

| | mean IoU | best | worst |
|---|---|---|---|
| 20 pieces | **0.7951** | water_fire 0.9218 | teapot_droplets 0.6369 |
| 5 pieces | **0.5583** | horse_circle 0.8450 | siggraph_sigchi 0.2728 |

Dropping 20 → 5 costs **24 IoU points** on average. horse_circle barely notices
(0.870 → 0.845; a circle is nearly one piece). siggraph_sigchi and acm_scf
collapse (−50, −36) — two detailed silhouettes competing for the same geometry
cannot be expressed in five pieces.

## The real question: does hinge2's advantage come from its piece count?

`baseline_h2pc` seeds each baseline run at the piece count the
`hinge2_restart_importnet` run converged to for that pair. If hinge2's result
were just a consequence of landing on a good piece count, a baseline handed that
same count should match it.

It does not.

| pair | pieces | baseline IoU | hinge2 IoU | delta | hinge2 stop |
|---|---|---|---|---|---|
| water_fire | 9 | 0.8064 | 0.9353 | **−0.1289** | shed_converged |
| cat_face_bass | 10 | 0.8543 | 0.9318 | **−0.0775** | shed_converged |
| robot_man | 11 | 0.8721 | 0.9151 | **−0.0430** | shed_converged |
| sun_moon | 18 | 0.8046 | 0.9139 | **−0.1093** | shed_converged |
| dance_argument | 29 | 0.8768 | 0.8767 | **+0.0001** | converged |
| teapot_droplets | 35 | 0.7806 | 0.7987 | −0.0181 | converged |
| acm_scf | 44 | 0.8449 | 0.8913 | −0.0464 | converged |
| **mean** | | **0.8343** | **0.8947** | **−0.0604** | |

**hinge2 beats the piece-count-matched baseline by 6.0 IoU points.** The
advantage is not the piece count — it is how the pieces got there.

### The split tracks `stop_reason` exactly

Sorting by hinge2's stop reason rather than by pair separates the effect:

| hinge2 outcome | pairs | mean delta |
|---|---|---|
| `shed_converged` (target reached, shedding ran) | water_fire, cat_face_bass, robot_man, sun_moon | **−0.0897** |
| `converged` (target never reached, no shedding) | dance_argument, teapot_droplets, acm_scf | **−0.0215** |

Where the fit-then-shed schedule actually ran — grow past 20 pieces, then shed
back to 9–18 — hinge2 is 9 points ahead of a baseline started at the same
endpoint. Where shedding never began, the gap nearly vanishes, and on
dance_argument it is zero to four decimal places.

That is the cleanest evidence so far that the *path* matters: pieces that were
grown to 29–38 and then shed are placed better than pieces initialized directly
at the final count. A baseline at 9 pieces has nine pieces' worth of swept-volume
initialization to work with; hinge2's nine are survivors of a much larger
population.

### Piece count helps the baseline on hard pairs

Comparing `h2pc` against the fixed-20 baseline on the same pairs shows the other
half of the picture:

| pair | pieces | h2pc IoU | 20pc IoU | 5pc IoU |
|---|---|---|---|---|
| water_fire | 9 | 0.8064 | 0.9218 | 0.6829 |
| cat_face_bass | 10 | 0.8543 | 0.9081 | 0.7301 |
| robot_man | 11 | 0.8721 | 0.9049 | 0.7361 |
| sun_moon | 18 | 0.8046 | 0.8097 | 0.4719 |
| dance_argument | 29 | **0.8768** | 0.7916 | 0.4020 |
| teapot_droplets | 35 | **0.7806** | 0.6369 | 0.4250 |
| acm_scf | 44 | **0.8449** | 0.6403 | 0.2839 |

Below 20 pieces the baseline loses ground, as expected. Above 20 it gains a
lot — acm_scf +20 points at 44 pieces, teapot +14 at 35, dance_argument +9 at
29. The three hard pairs were piece-starved at 20, which is worth knowing
independently: **the standard 20-piece baseline understates what no-SRD
optimization can do on detailed pairs.**

## Caveats

- `baseline_h2pc` ran at `--swept-resolution 300` to match the hinge2 batch it
  is compared against; `baseline_20pc` and `baseline_5pc` ran at the 256
  default. The cross-sweep column above therefore carries a small
  initialization-fidelity confound.
- Single seed (0) throughout. No error bars; a 0.0001 delta is not meaningfully
  distinguishable from a small real one.
- horse_circle is absent from `h2pc` because its hinge2 run had not finished
  when the counts were read.
- hinge2's counts for the three `converged` pairs are not counts it *chose* —
  those runs stalled at or near their own maximum without shedding. acm_scf's 44
  is more than double its start.

## Files

Per sweep, under `<sweep>/collected/`: `summary.tsv` (per-run metrics),
`curves.csv` (per-step traces), `means_by_group.tsv`, `convergence.tsv` (IoU
milestone crossings). `manifest.tsv` at the sweep root records job IDs and, for
`h2pc`, the per-pair `n_patches`. Render frames and per-run `history.csv` stay
on Oscar.
