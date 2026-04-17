"""Patch primitive — the basic unit of the anamorphic sculpture.

A Patch is a flat rectangular piece of laser-cut material positioned in 3D
space. Its learnable parameters (position, orientation, albedo) are torch
tensors so the optimizer can differentiate through geometry and color jointly.

Rotation is stored as a 3-vector (axis * angle in radians, Rodrigues form).
This avoids gimbal lock and keeps the parameterization unconstrained.
"""
