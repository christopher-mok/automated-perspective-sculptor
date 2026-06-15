"""Qt worker for running differentiable optimization off the main thread."""

from __future__ import annotations

from pathlib import Path
import traceback
import time

import numpy as np
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
        run_until_convergence: bool,
        convergence_threshold: float,
        view2_loss: str,
        sds_prompt: str,
        device: str,
        hanging_plane_size: float,
        hanging_plane_y: float,
        srd_config: dict[str, object] | None,
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
        self._run_until_convergence = run_until_convergence
        self._convergence_threshold = convergence_threshold
        self._view2_loss = view2_loss
        self._sds_prompt = sds_prompt
        self._device = device
        self._hanging_plane_size = hanging_plane_size
        self._hanging_plane_y = hanging_plane_y
        self._srd_config = srd_config
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
                view2_loss=self._view2_loss,
                sds_prompt=self._sds_prompt,
                device=self._device,
                hanging_plane_size=self._hanging_plane_size,
                hanging_plane_y=self._hanging_plane_y,
                srd_config=self._srd_config,
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


class BenchmarkWorker(QThread):
    """Run the built-in image-pair benchmark and persist its final outputs."""

    pair_started = pyqtSignal(int, int, str)
    pair_completed = pyqtSignal(str, object, object)
    failed = pyqtSignal(str)
    benchmark_finished = pyqtSignal(str, object)

    def __init__(
        self,
        *,
        cameras: list,
        image_pairs: list[tuple[str, Path, Path]],
        output_dir: Path,
        n_patches: int,
        init_mode: str,
        sam_variant: str,
        palette: object,
        lr: float,
        n_steps: int,
        trial_seeds: list[int],
        device: str,
        hanging_plane_size: float,
        hanging_plane_y: float,
        srd_config: dict[str, object] | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cameras = cameras
        self._image_pairs = image_pairs
        self._output_dir = output_dir
        self._n_patches = n_patches
        self._init_mode = init_mode
        self._sam_variant = sam_variant
        self._palette = palette
        self._lr = lr
        self._n_steps = n_steps
        self._trial_seeds = trial_seeds
        self._device = device
        self._hanging_plane_size = hanging_plane_size
        self._hanging_plane_y = hanging_plane_y
        self._srd_config = srd_config
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    @staticmethod
    def _save_render(render: object, path: Path) -> None:
        from PIL import Image

        array = np.asarray(render)
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(array, mode="RGBA").save(path)

    def run(self) -> None:
        try:
            from core.initialization import initialize_patches
            from ui.main_window import _load_target_image_with_border

            self._output_dir.mkdir(parents=True, exist_ok=True)
            results: list[tuple[str, int, int, dict[str, float]]] = []
            total_runs = len(self._image_pairs) * len(self._trial_seeds)
            run_index = 0

            for trial_number, seed in enumerate(self._trial_seeds, start=1):
                for label, target1_path, target2_path in self._image_pairs:
                    if self._stop_requested:
                        return

                    np.random.seed(seed)
                    import torch

                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    run_index += 1
                    run_label = f"{label}_trial{trial_number}"
                    self.pair_started.emit(run_index, total_runs, run_label)
                    target1 = _load_target_image_with_border(str(target1_path))
                    target2 = _load_target_image_with_border(str(target2_path))
                    patches = initialize_patches(
                        mode=self._init_mode,
                        n_patches=self._n_patches,
                        reference_image=target1,
                        sam_variant=self._sam_variant,
                        cameras=self._cameras,
                        device=self._device,
                        seed=seed,
                    )
                    optimizer = SceneOptimizer(
                        patches,
                        self._cameras[0],
                        self._cameras[1],
                        target1,
                        target2,
                        palette=self._palette,
                        lr=self._lr,
                        view2_loss="mse",
                        device=self._device,
                        hanging_plane_size=self._hanging_plane_size,
                        hanging_plane_y=self._hanging_plane_y,
                        srd_config=self._srd_config,
                    )

                    for step_index in range(1, self._n_steps + 1):
                        if self._stop_requested:
                            return
                        optimizer.step(step_index, self._n_steps)

                    metrics, render1, render2 = optimizer.evaluate_snapshot()
                    self._save_render(
                        render1,
                        self._output_dir / f"{run_label}_view1.png",
                    )
                    self._save_render(
                        render2,
                        self._output_dir / f"{run_label}_view2.png",
                    )
                    results.append((label, trial_number, seed, metrics))
                    self.pair_completed.emit(
                        run_label,
                        metrics,
                        optimizer.mesh_snapshot(),
                    )

            report_lines = [
                "Perspective Sculptor benchmark",
                f"trials={len(self._trial_seeds)}",
                f"seeds={','.join(str(seed) for seed in self._trial_seeds)}",
                f"steps={self._n_steps}",
                f"learning_rate={self._lr:.6g}",
                f"srd_enabled={self._srd_config is not None and bool(self._srd_config.get('enabled', False))}",
                "",
            ]
            for label, trial_number, seed, metrics in results:
                report_lines.append(
                    f"{label} trial={trial_number} seed={seed}: "
                    f"loss={metrics['loss']:.6f}"
                )
            report_lines.append("")
            for label, _target1, _target2 in self._image_pairs:
                losses = [
                    metrics["loss"]
                    for result_label, _trial, _seed, metrics in results
                    if result_label == label
                ]
                report_lines.append(
                    f"{label} mean_loss={float(np.mean(losses)):.6f}"
                )
            report_path = self._output_dir / "benchmark.txt"
            report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            self.benchmark_finished.emit(str(report_path), results)
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(details)
