"""Differentiable renderer using nvdiffrast.

Converts a list of Patch objects into a rasterized image from a given camera
viewpoint. The entire forward pass is differentiable, so gradients flow back
into patch positions, rotations, sizes, and albedo values.

nvdiffrast installation
-----------------------
  pip install git+https://github.com/NVlabs/nvdiffrast

On machines without CUDA, nvdiffrast falls back to an OpenGL rasterizer
(RasterizeGLContext). On macOS this is the only available backend.

NOTE: nvdiffrast's GL context and PyQt6's GL context must share the same
thread. Run rendering inside the OptimizationWorker (a QThread), *not* on
the main thread, and do *not* call any PyQt6 GL functions from that thread.
"""
