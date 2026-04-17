"""Optimization loop for the anamorphic sculpture.

SceneOptimizer wraps the render → loss → backward → step cycle.  It is a
plain Python object with no Qt dependency so it can be used from scripts,
notebooks, or the UI worker thread equally.

Typical use (from a script)
---------------------------
    optimizer = SceneOptimizer(patches, renderer, cam1, cam2, t1, t2, lr=1e-3)
    for step, metrics in optimizer.run(n_steps=500):
        print(step, metrics["loss"])

Typical use (from the UI worker)
---------------------------------
    See ui/worker.py — the worker wraps ``run()`` in a QThread and emits
    Qt signals for each step.
"""
