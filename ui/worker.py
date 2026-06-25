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
        swept_volume: object | None,
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
        self._swept_volume = swept_volume
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
                swept_volume=self._swept_volume,
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
        simulated_annealing: bool,
        swept_volume_resolution: int,
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
        self._simulated_annealing = simulated_annealing
        self._swept_volume_resolution = swept_volume_resolution
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    @staticmethod
    def _save_render(render: object, path: Path) -> None:
        from PIL import Image

        array = np.asarray(render)
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(array, mode="RGBA").save(path)

    def _write_loss_outputs(
        self,
        loss_samples: list[tuple[str, int, int, int, float]],
    ) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        csv_path = self._output_dir / "loss_history.csv"
        csv_lines = ["pair,trial,seed,step,loss"]
        for label, trial_number, seed, step, loss in loss_samples:
            csv_lines.append(
                f"{label},{trial_number},{seed},{step},{loss:.9f}"
            )
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

        plot_paths: list[Path] = []
        for label, _target1, _target2 in self._image_pairs:
            label_samples = [
                (step, loss)
                for sample_label, _trial, _seed, step, loss in loss_samples
                if sample_label == label
            ]
            if not label_samples:
                continue

            steps = sorted({step for step, _loss in label_samples})
            mean_losses = []
            for step in steps:
                losses = [
                    loss
                    for sample_step, loss in label_samples
                    if sample_step == step
                ]
                mean_losses.append(float(np.mean(losses)))

            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            ax.plot(steps, mean_losses, linewidth=2.0)
            ax.set_title(f"{label.replace('_', ' + ')} Average Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            plot_path = self._output_dir / f"{label}_average_loss.png"
            fig.savefig(plot_path)
            plt.close(fig)
            plot_paths.append(plot_path)

        return [csv_path, *plot_paths]

    def run(self) -> None:
        try:
            from core.initialization import initialize_patches
            from core.swept_volume import SweptVolume
            from ui.main_window import _load_target_image_with_border

            self._output_dir.mkdir(parents=True, exist_ok=True)
            results: list[tuple[str, int, int, dict[str, float]]] = []
            loss_samples: list[tuple[str, int, int, int, float]] = []
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
                    trial_started = time.perf_counter()
                    target1 = _load_target_image_with_border(str(target1_path))
                    target2 = _load_target_image_with_border(str(target2_path))
                    last_progress = -1

                    def _on_swept_progress(done: int, total: int) -> None:
                        nonlocal last_progress
                        percent = int(round((done / max(total, 1)) * 100))
                        if percent >= last_progress + 10 or done == total:
                            last_progress = percent
                            self.pair_started.emit(
                                run_index,
                                total_runs,
                                f"{run_label}_swept_volume_{percent}%",
                            )
                            print(
                                f"[Swept volume] {run_label}: "
                                f"{done}/{total} slices ({percent}%)"
                            )

                    swept_volume = SweptVolume.from_images(
                        target1,
                        target2,
                        self._cameras,
                        hanging_plane_size=self._hanging_plane_size,
                        resolution=self._swept_volume_resolution,
                        progress_callback=_on_swept_progress,
                    )
                    patches = initialize_patches(
                        mode=self._init_mode,
                        n_patches=self._n_patches,
                        reference_image=target1,
                        sam_variant=self._sam_variant,
                        cameras=self._cameras,
                        device=self._device,
                        seed=seed,
                        swept_volume=swept_volume,
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
                        simulated_annealing=self._simulated_annealing,
                        swept_volume=swept_volume,
                    )

                    optimization_started = time.perf_counter()
                    for step_index in range(1, self._n_steps + 1):
                        if self._stop_requested:
                            return
                        step_metrics = optimizer.step(step_index, self._n_steps)
                        if step_index % 10 == 0:
                            loss_samples.append((
                                label,
                                trial_number,
                                seed,
                                step_index,
                                float(step_metrics.get("loss", 0.0)),
                            ))
                    optimization_seconds = time.perf_counter() - optimization_started
                    final_deletion_stats = {
                        "tiny_deleted": 0.0,
                        "loss_improving_deleted": 0.0,
                        "evaluated": 0.0,
                    }
                    if optimizer.srd is not None and optimizer.srd.enabled:
                        final_deletion_stats = optimizer.srd.final_deletion_pass(
                            optimizer,
                            optimizer.optim,
                            self._n_steps,
                        )

                    metrics, render1, render2 = optimizer.evaluate_snapshot()
                    metrics["benchmark_final_tiny_deleted"] = final_deletion_stats[
                        "tiny_deleted"
                    ]
                    metrics["benchmark_final_loss_deleted"] = final_deletion_stats[
                        "loss_improving_deleted"
                    ]
                    metrics["benchmark_final_delete_evaluated"] = final_deletion_stats[
                        "evaluated"
                    ]
                    self._save_render(
                        render1,
                        self._output_dir / f"{run_label}_view1.png",
                    )
                    self._save_render(
                        render2,
                        self._output_dir / f"{run_label}_view2.png",
                    )
                    metrics["trial_seconds"] = time.perf_counter() - trial_started
                    metrics["average_step_seconds"] = (
                        optimization_seconds / self._n_steps
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
                f"simulated_annealing={self._simulated_annealing}",
                f"swept_volume_resolution={self._swept_volume_resolution}",
                "",
            ]
            for label, trial_number, seed, metrics in results:
                report_lines.append(
                    f"{label} trial={trial_number} seed={seed}: "
                    f"loss={metrics['loss']:.6f}, "
                    f"trial_seconds={metrics['trial_seconds']:.3f}, "
                    f"average_step_seconds={metrics['average_step_seconds']:.6f}, "
                    f"final_tiny_deleted={metrics['benchmark_final_tiny_deleted']:.0f}, "
                    f"final_loss_deleted={metrics['benchmark_final_loss_deleted']:.0f}, "
                    f"final_delete_evaluated={metrics['benchmark_final_delete_evaluated']:.0f}"
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
            loss_output_paths = self._write_loss_outputs(loss_samples)
            report_lines.append("")
            report_lines.append("loss_outputs:")
            for path in loss_output_paths:
                report_lines.append(f"- {path.name}")
            report_path = self._output_dir / "benchmark.txt"
            report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            self.benchmark_finished.emit(str(report_path), results)
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(details)
