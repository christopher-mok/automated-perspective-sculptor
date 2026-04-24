"""Qt worker for running differentiable optimization off the main thread."""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.optimizer import SceneOptimizer


class OptimizationWorker(QThread):
    """Run SceneOptimizer in a thread and emit viewport-safe mesh snapshots."""

    step_completed = pyqtSignal(int, object, object)  # step, metrics, meshes
    failed = pyqtSignal(str)
    optimization_finished = pyqtSignal(object)

    def __init__(
        self,
        *,
        patches: list,
        cameras: list,
        target1: object,
        target2: object | None,
        palette: object,
        lr: float,
        n_steps: int,
        view2_loss: str,
        sds_prompt: str,
        device: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._patches = patches
        self._cameras = cameras
        self._target1 = target1
        self._target2 = target2
        self._palette = palette
        self._lr = lr
        self._n_steps = n_steps
        self._view2_loss = view2_loss
        self._sds_prompt = sds_prompt
        self._device = device
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            optimizer = SceneOptimizer(
                self._patches,
                self._cameras[0],
                self._cameras[1],
                self._target1,
                self._target2,
                palette=self._palette,
                lr=self._lr,
                view2_loss=self._view2_loss,
                sds_prompt=self._sds_prompt,
                device=self._device,
            )

            last_metrics: dict[str, float] = {}
            for step_idx, metrics in optimizer.run(self._n_steps):
                if self._stop_requested:
                    break
                last_metrics = metrics
                self.step_completed.emit(step_idx, metrics, optimizer.mesh_snapshot())

            self.optimization_finished.emit(last_metrics)
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(details)
