# Discussion

## Applications

We parameterize each piece as a closed cubic Bézier outline, but nothing in the
formulation requires that choice. The two-view rendering objective is agnostic
to how a piece's geometry is produced, and freezing the control-point
parameters while leaving the rigid pose free recovers a *fixed-primitive*
variant of the method: the optimizer arranges a supplied inventory of shapes —
rectangles, discs, letterforms, or a stock of laser-cuttable parts — rather
than sculpting outlines of its own. The limiting case of this is the practice
that motivated us in the first place. Artists who build perspective sculptures
routinely assemble them from everyday objects, whose geometry is not theirs to
edit, and the same pipeline applies with arbitrary 3D meshes substituted for
flat panels.

That regime is where adaptive structure matters most. When the shape
parameters are frozen, gradient descent can only slide a fixed multiset of
pieces around; the continuous parameters no longer control *what* the
silhouette is able to express, only where its content sits. The only remaining
lever is the composition of the inventory itself, which is precisely the
discrete add/delete/restore search SRD performs. A deformable-panel run can
often recover from a poor initial piece count by reshaping what it already
has, and treats SRD as a refinement; a fixed-primitive run cannot, and depends
on it. (Splitting is the exception — it presumes a divisible outline and has
no counterpart for a rigid mesh, so the rewrite budget would redistribute over
the remaining kinds.)

A second direction relaxes opacity rather than shape. If pieces are
semi-transparent, a region covered by $k$ overlapping panels of transmittance
$t$ transmits $t^k$, so overlap depth becomes directly readable as tone and the
target can be a grayscale image instead of a binary mask. This inverts the role
overlap plays in our formulation: we currently penalize intersecting pieces, so
that a panel occluding another is a configuration to be repaired, whereas under
translucency it is the mechanism by which mid-tones are produced, and the
overlap term would have to be replaced by something that schedules depth rather
than discourages it. The optimization is correspondingly harder. For uniformly
absorbing gray panels the composite depends only on how many layers cover a
pixel and not on their order, but for colored gels the compositing is
order-dependent and subtractive, so the depth permutation of the pieces enters
the objective as a discrete variable alongside the rewrites SRD already
searches.

## Limitations

**Piece-count control cannot distinguish essential detail.** Once the fit
clears the target IoU, our count objective
$J = n + \lambda\,\mathrm{hinge}(\tau - \mathrm{IoU})^{p}$ is flat in image
fidelity, so every candidate rewrite is scored purely on the pieces it costs.
Any deletion that leaves mean IoU above $\tau$ is therefore accepted, however
important the deleted piece is to the subject — and the constant-price
`lambda_count` variant is worse, since it keeps buying piece reductions with
fidelity no matter how good the fit already is. The deeper cause is that IoU is
an area measure: it charges the same for a fixed number of mislabeled pixels
regardless of how they are distributed. A slight erosion spread around the
entire boundary and an entirely missing limb can cost identical IoU, but they
are not perceptually interchangeable. Losing a small, semantically necessary
part — a leg, a beak, a handle — frequently reads worse than a silhouette that
is uniformly underfilled but structurally complete, which is the failure we
observe when trimming to the minimum acceptable IoU. Addressing this requires a
fidelity term that is sensitive to *where* the error falls, such as weighting
the deficit by distance to the shape's medial axis, or penalizing changes in
connected-component count and topology, so that thin or isolated structures are
priced above their pixel area.

**The edge-on constraint is defined against the camera axis, not the
per-piece viewing ray.** We replace the soft edge-on penalty with a hard
post-step constraint that projects each piece's yaw to at least $15^\circ$ from
either camera's yaw. Because a piece's orientation is a single yaw and the
constraint is stated against one nominal axis per camera, it is exact only for
pieces on the optical axis. Under perspective the ray from the camera to a
piece near the edge of the frame deviates from that axis by up to half the
field of view, which for our cameras is $19.8^\circ$ horizontally — larger than
the margin itself. A piece that drifts to the periphery of the frustum can thus
satisfy the constraint and still present very nearly edge-on to the ray that
actually images it. The per-piece projected-area penalty removes the most
degenerate of these, but pieces large enough to clear the minimum projected
area while severely foreshortened survive. The correction is to constrain the
angle between the piece's normal and its own view ray rather than the camera
yaw. This is cheap to evaluate, but it makes the feasible band depend on the
piece's position, so it can no longer be applied as an independent post-step
clamp on orientation: the projection would have to account for the center
update in the same step, or be reintroduced as a soft term.
