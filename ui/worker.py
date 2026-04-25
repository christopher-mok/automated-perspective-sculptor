"""Qt worker for running differentiable optimization off the main thread."""

from __future__ import annotations

import traceback
import time

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
        initial_temperature: float,
        temperature_schedule: str,
        enable_patch_restarts: bool,
        restart_interval: int,
        run_until_convergence: bool,
        convergence_threshold: float,
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
        self._initial_temperature = initial_temperature
        self._temperature_schedule = temperature_schedule
        self._enable_patch_restarts = enable_patch_restarts
        self._restart_interval = restart_interval
        self._run_until_convergence = run_until_convergence
        self._convergence_threshold = convergence_threshold
        self._view2_loss = view2_loss
        self._sds_prompt = sds_prompt
        self._device = device
        self._stop_requested = False
        self._pause_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True
        self._pause_requested = False

    def set_paused(self, paused: bool) -> None:
        self._pause_requested = paused

    def _wait_if_paused(self) -> None:
        while self._pause_requested and not self._stop_requested:
            time.sleep(0.05)

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
                initial_temperature=self._initial_temperature,
                temperature_schedule=self._temperature_schedule,
                enable_patch_restarts=self._enable_patch_restarts,
                restart_interval=self._restart_interval,
                view2_loss=self._view2_loss,
                sds_prompt=self._sds_prompt,
                device=self._device,
            )

            last_metrics: dict[str, float] = {}
            if self._run_until_convergence:
                step_idx = 0
                while not self._stop_requested:
                    self._wait_if_paused()
                    if self._stop_requested:
                        break
                    step_idx += 1
                    last_metrics = optimizer.step(step_idx, self._n_steps)
                    self.step_completed.emit(
                        step_idx,
                        last_metrics,
                        optimizer.mesh_snapshot(),
                    )
                    loss = last_metrics.get("loss", float("inf"))
                    if loss <= self._convergence_threshold:
                        break
            else:
                for step_idx in range(1, self._n_steps + 1):
                    self._wait_if_paused()
                    if self._stop_requested:
                        break
                    last_metrics = optimizer.step(step_idx, self._n_steps)
                    self.step_completed.emit(
                        step_idx,
                        last_metrics,
                        optimizer.mesh_snapshot(),
                    )

            self.optimization_finished.emit(last_metrics)
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(details)
