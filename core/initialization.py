"""Patch initialization strategies.

Each function returns a list of Patch objects positioned in the XZ plane
(y=0) ready to be passed to SceneOptimizer.

Strategies
----------
init_grid   : Patches on a regular grid — fast, deterministic.
init_random : Patches at random positions — good for breaking symmetry.
init_sam    : Use Meta's Segment Anything Model to seed patch positions from
              a reference image (requires the ``segment-anything`` package).
"""
