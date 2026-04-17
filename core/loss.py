"""Loss functions for the optimization loop.

All functions operate on (H, W, C) float32 tensors in [0, 1].

View 2 supports two modes:
  - MSE   : direct pixel reconstruction against a target image.
  - SDS   : Score Distillation Sampling from a text prompt using a
             HuggingFace diffusion pipeline (requires diffusers).
"""
