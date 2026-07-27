# Constrained vs. unconstrained comparison

Single-paragraph draft for the results section. Numbers computed from the
five pairs in the figure; per-pair arithmetic at the bottom.

---

Constraining the piece count removes slightly more than half the panels --
140 across the five pairs against 65, a mean of 28.0 against 13.0, or
$2.2\times$ -- at a cost of 1.9 IoU points on average (0.947 against 0.928).
The unconstrained run scores higher on every pair, but the margin varies by
more than an order of magnitude, from 0.001 on sun/moon, where the two are
effectively tied, to 0.037 on robot/man, and that spread is the more
informative result. Measured per panel removed, the IoU cost of the
constraint ranges from 0.00008 to 0.00247, ordering the pairs by how much
thin structure the subject carries: a sun and a crescent are near-convex
regions whose area sits almost entirely in one mass, so panels can be merged
nearly for free, whereas the antenna, arms, and leg gap of a robot are thin,
low-area features that each need a panel of their own, and these are what the
constrained run gives up first. Thin features are simultaneously the cheapest
thing an area-based objective can surrender and the most expensive thing for
a viewer to lose, so the constrained robot reads as less legible than its
3.7-point deficit suggests while the constrained moon is indistinguishable
from the unconstrained one -- an argument for exposing the constraint as a
control rather than adopting it as a default, to be relaxed, or paired with a
fidelity term that prices structure above area, on subjects assembled from
thin appendages.

---

## Per-pair arithmetic

| pair | unc. IoU | unc. panels | con. IoU | con. panels | dIoU | panels removed | dIoU per panel |
|---|---|---|---|---|---|---|---|
| sun_moon | 0.944 | 35 | 0.943 | 22 | 0.001 | 13 | 0.00008 |
| cat_face_bass | 0.950 | 30 | 0.932 | 10 | 0.018 | 20 | 0.00090 |
| horse_circle | 0.931 | 25 | 0.915 | 13 | 0.016 | 12 | 0.00133 |
| droplets_fire | 0.960 | 24 | 0.935 | 9 | 0.025 | 15 | 0.00167 |
| robot_man | 0.952 | 26 | 0.915 | 11 | 0.037 | 15 | 0.00247 |
| **mean** | **0.947** | **28.0** | **0.928** | **13.0** | **0.019** | **15.0** | |

Panel ratio from totals: 140 / 65 = 2.15x (reported as 2.2x).
Mean of per-pair ratios is 2.31x -- avoid, it overweights cat_face_bass.
Reduction: 53.6% of panels removed.
