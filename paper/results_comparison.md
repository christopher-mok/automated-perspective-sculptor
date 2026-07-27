# Constrained vs. unconstrained comparison

Draft prose for the results table. Numbers computed from the five pairs in
the figure; see the table at the bottom for the per-pair arithmetic.

---

Table~\ref{tab:constrained} compares the two settings on five image pairs.
Constraining the piece count removes slightly more than half the panels --
140 across the five pairs against 65, a mean of 28.0 against 13.0, or
$2.2\times$ -- at a cost of 1.9 IoU points on average (0.947 against 0.928).
The unconstrained run scores higher on every pair, but the size of that
margin varies by more than an order of magnitude, from 0.001 on sun/moon,
where the two are effectively tied, to 0.037 on robot/man.

The aggregate hides the more useful pattern. Measured per panel removed, the
IoU cost of the constraint spans a factor of thirty across the five pairs,
from 0.00008 on sun/moon to 0.00247 on robot/man, and it orders the pairs by
how much thin structure the subject carries. A sun and a crescent are
near-convex regions whose area sits almost entirely in one mass, so panels
can be merged nearly for free. A robot and a standing figure are the
opposite: the antenna, the arms, the gap between the legs are thin, low-area
features, each of which needs a panel of its own, and these are what the
constrained run gives up first.

This is the failure mode of an area-based objective in its plainest form.
Thin features are simultaneously the cheapest thing IoU can surrender, since
they cover few pixels, and the most expensive thing for a viewer to lose,
since they carry much of the subject's identity. The constrained robot
therefore reads as noticeably less legible than a 3.7-point IoU deficit
would suggest, while the constrained moon, 0.001 behind, is
indistinguishable from the unconstrained one. We read this as an argument
for exposing the constraint as a control rather than adopting it as a
default: on subjects that are mostly compact mass it halves the piece count
at no visible cost, and on subjects assembled from thin appendages it should
either be relaxed or paired with a fidelity term that prices structure above
area.

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
