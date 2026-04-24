---
name: Patch and pipeline design (updated)
description: Core data structures and optimization loop for the anamorphic sculpture — spline-based patches, not rectangles
type: project
---

Patch design is spline-based, not rectangular.  The earlier rectangle+angle design is obsolete and will be replaced.

**Why:** Spline patches are physically accurate laser-cut shapes that can be extruded into 3D meshes and rendered differentiably.

**How to apply:** When rewriting core/patch.py, core/initialization.py, core/optimizer.py, core/renderer.py — use the structures below, not the old rectangle model.

---

## Patch struct

- 5 Bezier curve control points (defines the outline of the cut piece as a closed spline)
- theta: Y-axis rotation (same as before)

## Bezier control point struct (all differentiable torch params)

- `x, y, z` — 3D position of the control point
- `handle_scale` — scalar, scales both handles symmetrically
- `handle_rotation` — scalar (radians), rotates both handles around the control point
- `next_control_point` — reference to next point in the chain
- `prev_control_point` — reference to previous point in the chain

The two handles per control point are derived from handle_scale and handle_rotation (symmetric/mirrored handles, like Illustrator's smooth nodes).

## Optimization loop

1. Accept user-chosen colors (discrete palette, not continuous RGB)
2. Preprocess input image: map pixel values to the nearest user-selected color
3. Initialize n patches to cover the image (grid or SAM)
4. Loop until converged:
   a. Sample points along each patch's spline (differentiable w.r.t. control points via autodiff)
   b. Extrude sampled spline outline → 3D triangle mesh
   c. Render mesh with nvdiffrast from both camera angles
   d. MSE loss between rendered image and preprocessed target
   e. Adam step on all control point parameters (x, y, z, handle_scale, handle_rotation) and theta

## Post-optimization (export)

- For each piece: compute center of mass → find two string attachment positions that balance it
- Generate a hanging grid above the pieces
- Export SVG cut files (one per piece, flat unextruded outline)
