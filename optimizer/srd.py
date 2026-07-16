"""Stochastic Rewrite Descent for adaptive patch structure."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from core.patch import ControlPoint, Patch
from core.swept_volume import _project_points

if TYPE_CHECKING:
    from core.swept_volume import SweptVolume
    from scene.camera import Camera


RewriteKind = Literal["add", "restore", "delete", "split"]


@dataclass
class SRDStats:
    added: int = 0
    deleted: int = 0
    total_added: int = 0
    total_deleted: int = 0
    active: int = 0
    evaluated: int = 0
    promising: int = 0
    accepted: int = 0


@dataclass
class RewriteCandidate:
    kind: RewriteKind
    position: np.ndarray | None = None
    patch_index: int | None = None
    history_index: int | None = None
    patch_state: dict | None = None
    improvement: float = 0.0
    applied_index: int | None = None
    reason: str = ""

    @property
    def label(self) -> str:
        if self.kind == "delete":
            return f"DeletePatch({self.patch_index})"
        if self.kind == "split":
            return f"SplitPatch({self.patch_index})"
        if self.kind == "restore":
            return f"RestorePatch({self.history_index})"
        return "AddPatch"


def _patch_parameters(patches: Sequence[Patch]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for patch in patches:
        params.extend([patch.center, patch.theta])
        for cp in patch.control_points:
            cp.z.requires_grad_(False)
            cp.z.grad = None
            params.extend([cp.x, cp.y, cp.handle_scale, cp.handle_rotation])
    return params


def _small_default_patch(
    position: np.ndarray,
    device: str,
    albedo: Sequence[float],
    creation_step: int,
    label: str,
    radius: float = 0.05,
) -> Patch:
    """Create a small regular-pentagon patch for SRD additions."""
    control_points: list[ControlPoint] = []
    handle_scale = radius * (4.0 / 3.0) * math.tan(math.pi / Patch.N_CONTROL_POINTS)
    for idx in range(Patch.N_CONTROL_POINTS):
        angle = 2.0 * math.pi * idx / Patch.N_CONTROL_POINTS - math.pi / 2.0
        control_points.append(ControlPoint(
            x=radius * math.cos(angle),
            y=radius * math.sin(angle),
            z=0.0,
            handle_scale=handle_scale,
            handle_rotation=angle + math.pi / 2.0,
            device=device,
        ))
    patch = Patch(
        control_points=control_points,
        center=position.tolist(),
        theta=0.0,
        albedo=list(albedo),
        device=device,
        label=label,
    )
    patch.creation_step = creation_step
    patch.self_intersect_counter = 0
    return patch


class StochasticRewriteDescent:
    """Sample candidate add/delete rewrites and accept useful compatible ones."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        interval: int = 50,
        lambda_count: float = 0.05,
        lambda_area: float = 0.05,
        min_patch_area: float = 0.01,
        max_patches: int = 200,
        min_patches: int = 4,
        max_additions: int = 3,
        max_deletions: int = 3,
        cooldown_steps: int = 30,
        candidate_count: int = 64,
        scene_box_size: float = 5.0,
        rewrite_eval_steps: int = 4,
        no_contribution_alpha: float = 1e-5,
        no_effect_image_delta: float = 1e-6,
        rule_violation_tol: float = 1e-4,
        swept_volume: "SweptVolume | None" = None,
        swept_volume_spawn_fraction: float = 0.75,
        disable_swept_volume_adds: bool = False,
        loss_only_deletion: bool = False,
        disable_splitting: bool = False,
    ) -> None:
        self.enabled = enabled
        self.interval = interval
        self.lambda_count = lambda_count
        self.lambda_area = lambda_area
        self.min_patch_area = min_patch_area
        self.max_patches = max_patches
        self.min_patches = min_patches
        self.max_additions = max_additions
        self.max_deletions = max_deletions
        self.cooldown_steps = cooldown_steps
        self.candidate_count = candidate_count
        self.scene_box_size = scene_box_size
        self.rewrite_eval_steps = rewrite_eval_steps
        self.no_contribution_alpha = no_contribution_alpha
        self.no_effect_image_delta = no_effect_image_delta
        self.rule_violation_tol = rule_violation_tol
        self.swept_volume = swept_volume
        self.swept_volume_spawn_fraction = float(np.clip(swept_volume_spawn_fraction, 0.0, 1.0))
        # Ablation switches: disable swept-volume-guided additions, restrict
        # deletion to loss-improving rewrites only, and disable splitting.
        self.disable_swept_volume_adds = bool(disable_swept_volume_adds)
        self.loss_only_deletion = bool(loss_only_deletion)
        self.disable_splitting = bool(disable_splitting)
        self._swept_point_order = np.empty(0, dtype=np.int64)
        self._swept_point_cursor = 0
        self.deleted_history: list[dict] = []
        self.stats = SRDStats()

    def step(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        current_loss: float,
        cameras: Sequence["Camera"],
        targets: tuple[torch.Tensor, torch.Tensor | None],
        current_step: int,
    ) -> SRDStats:
        """Run one SRD rewrite pass when the interval says it is time."""
        self.stats.added = 0
        self.stats.deleted = 0
        self.stats.evaluated = 0
        self.stats.promising = 0
        self.stats.accepted = 0
        self.stats.active = len(model.patches)

        if not self.enabled or self.interval <= 0 or current_step % self.interval != 0:
            return self.stats

        tiny_deletes = (
            [] if self.loss_only_deletion else self._tiny_area_delete_rewrites(model)
        )
        if tiny_deletes:
            self._apply_rewrites(model, optimizer, tiny_deletes, current_step)
            self.stats.accepted += len(tiny_deletes)
            self.stats.active = len(model.patches)

        mandatory_deletes = (
            [] if self.loss_only_deletion
            else self._mandatory_delete_rewrites(model, current_step)
        )
        if mandatory_deletes:
            self._apply_rewrites(model, optimizer, mandatory_deletes, current_step)
            self.stats.accepted += len(mandatory_deletes)
            self.stats.active = len(model.patches)
            with torch.no_grad():
                render1, render2 = model.renderer.render_both(
                    model.patches,
                    model.camera1,
                    model.camera2,
                    model.render_resolutions,
                )
                current_loss_tensor, _ = model._loss_from_renders(render1, render2, model.patches)
                current_loss = float(current_loss_tensor.detach().cpu())

        uncovered_masks = self._uncovered_target_masks(model)
        candidates = self._sample_rewrites(model, current_step, uncovered_masks)
        scored: list[RewriteCandidate] = []
        for candidate in candidates:
            improvement = self.evaluate_rewrite(model, optimizer, candidate, current_loss)
            self.stats.evaluated += 1
            if improvement > 0.0:
                candidate.improvement = improvement
                scored.append(candidate)

        self.stats.promising = len(scored)
        accepted = self._select_compatible(scored)
        self._apply_rewrites(model, optimizer, accepted, current_step)
        self.stats.accepted += len(accepted)
        self.stats.active = len(model.patches)

        print(
            f"SRD step {current_step}: evaluated {self.stats.evaluated} candidates, "
            f"{self.stats.promising} promising, accepted {self.stats.accepted}"
        )
        for candidate in accepted:
            patch_ref = candidate.applied_index if candidate.applied_index is not None else candidate.patch_index
            print(f"  Accepted {candidate.label} at patch {patch_ref}, improvement={candidate.improvement:.6f}")
        for candidate in [*tiny_deletes, *mandatory_deletes]:
            patch_ref = candidate.applied_index if candidate.applied_index is not None else candidate.patch_index
            print(f"  Mandatory {candidate.label} at patch {patch_ref}, reason={candidate.reason}")
        print(
            f"  Total patches: {len(model.patches)}, total adds: {self.stats.total_added}, "
            f"total deletes: {self.stats.total_deleted}"
        )
        return self.stats

    def _tiny_area_delete_rewrites(self, model) -> list[RewriteCandidate]:
        """Delete all below-threshold pieces at the start of a rewrite step."""
        if len(model.patches) <= 1:
            return []

        areas = [float(patch.compute_area().detach().cpu()) for patch in model.patches]
        indices = [idx for idx, area in enumerate(areas) if area < self.min_patch_area]
        if len(indices) >= len(model.patches):
            keep_idx = min(range(len(areas)), key=lambda idx: areas[idx])
            indices = [idx for idx in indices if idx != keep_idx]

        return [
            RewriteCandidate(
                kind="delete",
                patch_index=idx,
                reason=f"Area {areas[idx]:.6f} below minimum {self.min_patch_area}",
            )
            for idx in indices
        ]

    def final_deletion_pass(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        current_step: int,
    ) -> dict[str, float]:
        """Greedily delete tiny or loss-worsening pieces at the end of a run."""
        stats = {
            "tiny_deleted": 0.0,
            "loss_improving_deleted": 0.0,
            "evaluated": 0.0,
        }
        if not self.enabled or len(model.patches) <= 1:
            return stats

        tiny_deletes = (
            [] if self.loss_only_deletion else self._tiny_area_delete_rewrites(model)
        )
        if tiny_deletes:
            self._apply_rewrites(model, optimizer, tiny_deletes, current_step)
            optimizer = model.optim
            stats["tiny_deleted"] = float(len(tiny_deletes))

        current_loss = self._current_model_loss(model)
        patch_index = 0
        while patch_index < len(model.patches) and len(model.patches) > 1:
            stats["evaluated"] += 1.0
            loss_without_patch = self._loss_without_patch(model, patch_index)
            if loss_without_patch < current_loss:
                rewrite = RewriteCandidate(
                    kind="delete",
                    patch_index=patch_index,
                    improvement=current_loss - loss_without_patch,
                    reason="final deletion pass improved loss",
                )
                self._apply_single(
                    model,
                    rewrite,
                    current_step=current_step,
                    tentative=False,
                )
                optimizer = self._rebuild_optimizer(model, optimizer)
                model.optim = optimizer
                model._post_step_constraints()
                current_loss = loss_without_patch
                stats["loss_improving_deleted"] += 1.0
                continue
            patch_index += 1

        model._post_step_constraints()
        return stats

    def _current_model_loss(self, model) -> float:
        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            loss, _ = model._loss_from_renders(render1, render2, model.patches)
        return float(loss.detach().cpu())

    def _loss_without_patch(self, model, patch_index: int) -> float:
        remaining = [
            patch
            for idx, patch in enumerate(model.patches)
            if idx != patch_index
        ]
        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                remaining,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            loss, _ = model._loss_from_renders(render1, render2, remaining)
        return float(loss.detach().cpu())

    def _mandatory_delete_rewrites(self, model, current_step: int) -> list[RewriteCandidate]:
        """Find patches that must be deleted before stochastic SRD scoring."""
        if len(model.patches) <= self.min_patches:
            return []

        deletes: list[RewriteCandidate] = []
        available_deletions = max(0, len(model.patches) - self.min_patches)
        for idx, patch in enumerate(model.patches):
            if len(deletes) >= min(self.max_deletions, available_deletions):
                break
            if current_step - int(getattr(patch, "creation_step", 0)) < self.cooldown_steps:
                continue

            reason = self._mandatory_delete_reason(model, idx)
            if reason:
                deletes.append(RewriteCandidate(
                    kind="delete",
                    patch_index=idx,
                    improvement=0.0,
                    reason=reason,
                ))
        return deletes

    def _mandatory_delete_reason(self, model, patch_index: int) -> str:
        reasons: list[str] = []
        if self._patch_is_entirely_above_hanging_plane(model, patch_index):
            reasons.append("entirely above hanging plane")
        if self._patch_violates_rules(model, patch_index):
            reasons.append("rule violation")
        if not self._patch_contributes_to_either_image(model, patch_index):
            reasons.append("no image contribution")
        if not self._deleting_patch_changes_image(model, patch_index):
            reasons.append("delete has no image effect")
        return ", ".join(reasons)

    def _patch_is_entirely_above_hanging_plane(self, model, patch_index: int) -> bool:
        patch = model.patches[patch_index]
        plane_y = float(getattr(model, "hanging_plane_y", 3.5))
        n_per_segment = int(getattr(model.renderer, "n_per_segment", 20))
        with torch.no_grad():
            verts, _ = patch.extruded_mesh_world(n_per_segment=n_per_segment)
            if verts.numel() == 0:
                return False
            return bool(torch.all(verts[:, 1] > plane_y).item())

    def _patch_violates_rules(self, model, patch_index: int) -> bool:
        patch = model.patches[patch_index]
        params = [patch.center, patch.theta]
        for cp in patch.control_points:
            params.extend([cp.x, cp.y, cp.z, cp.handle_scale, cp.handle_rotation])
        if any(not torch.isfinite(param.detach()).all().item() for param in params):
            return True
        if any(abs(float(cp.z.detach().cpu())) > self.rule_violation_tol for cp in patch.control_points):
            return True
        if any(float(cp.handle_scale.detach().cpu()) <= 0.0 for cp in patch.control_points):
            return True
        return patch.is_self_intersecting()

    def _patch_contributes_to_either_image(self, model, patch_index: int) -> bool:
        patch = model.patches[patch_index]
        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                [patch],
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            alpha1 = render1[..., 3].amax() if render1.shape[-1] >= 4 else render1[..., :3].amax()
            alpha2 = render2[..., 3].amax() if render2.shape[-1] >= 4 else render2[..., :3].amax()
        return (
            float(alpha1.detach().cpu()) > self.no_contribution_alpha
            or float(alpha2.detach().cpu()) > self.no_contribution_alpha
        )

    def _deleting_patch_changes_image(self, model, patch_index: int) -> bool:
        with torch.no_grad():
            full1, full2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            remaining = [patch for idx, patch in enumerate(model.patches) if idx != patch_index]
            without1, without2 = model.renderer.render_both(
                remaining,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            delta = torch.maximum(
                (full1 - without1).abs().amax(),
                (full2 - without2).abs().amax(),
            )
        return float(delta.detach().cpu()) > self.no_effect_image_delta

    def evaluate_rewrite(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        rewrite: RewriteCandidate,
        current_loss: float,
    ) -> float:
        """Tentatively apply a rewrite, run local lookahead steps, then score it."""
        saved_patches, saved_optimizer_state = self._save_state(model, optimizer)
        try:
            self._apply_single(model, rewrite, current_step=0, tentative=True)
            model.optim = self._rebuild_optimizer(model, optimizer)

            n_steps = self._rewrite_eval_steps(rewrite)
            for _ in range(n_steps):
                valid_shape_states = model._capture_patch_shape_states()
                model.optim.zero_grad(set_to_none=True)
                render1, render2 = model.renderer.render_both(
                    model.patches,
                    model.camera1,
                    model.camera2,
                    model.render_resolutions,
                )
                loss, _ = model._loss_from_renders(render1, render2, model.patches)
                loss.backward()
                model.optim.step()
                model.optim.zero_grad(set_to_none=True)
                model._post_step_constraints(valid_shape_states, model.optim)

            with torch.no_grad():
                render1_new, render2_new = model.renderer.render_both(
                    model.patches,
                    model.camera1,
                    model.camera2,
                    model.render_resolutions,
                )
                new_loss, _ = model._loss_from_renders(render1_new, render2_new, model.patches)
            return current_loss - float(new_loss.detach().cpu())
        finally:
            self._restore_state(model, optimizer, saved_patches, saved_optimizer_state)

    def _rewrite_eval_steps(self, rewrite: RewriteCandidate) -> int:
        """Use a slightly longer local lookahead for growth rewrites."""
        if rewrite.kind in ("add", "restore", "split"):
            return max(1, self.rewrite_eval_steps)
        return 1

    def _sample_rewrites(
        self,
        model,
        current_step: int,
        uncovered_masks: tuple[np.ndarray, np.ndarray | None] | None = None,
    ) -> list[RewriteCandidate]:
        candidates: list[RewriteCandidate] = []
        add_budget = int(round(self.candidate_count * 0.35))
        delete_budget = int(round(self.candidate_count * 0.15))
        split_budget = self.candidate_count - add_budget - delete_budget
        if self.disable_splitting:
            add_budget += split_budget
            split_budget = 0

        if len(model.patches) >= self.max_patches:
            add_budget = 0
            split_budget = 0
            delete_budget = self.candidate_count
        if len(model.patches) <= self.min_patches:
            delete_budget = 0
            add_budget = self.candidate_count - split_budget

        for _ in range(add_budget):
            if np.random.random() < 0.35 and self.deleted_history:
                hist_idx = int(np.random.randint(0, len(self.deleted_history)))
                candidates.append(RewriteCandidate(
                    kind="restore",
                    history_index=hist_idx,
                    patch_state=copy.deepcopy(self.deleted_history[hist_idx]),
                ))
            else:
                if (
                    self.swept_volume is not None
                    and not self.disable_swept_volume_adds
                    and np.random.random() < self.swept_volume_spawn_fraction
                ):
                    position = self._sample_guided_swept_volume_position(
                        model,
                        uncovered_masks,
                    )
                    if position is None:
                        continue
                else:
                    position = np.random.uniform(
                        -self.scene_box_size * 0.5,
                        self.scene_box_size * 0.5,
                        size=3,
                    ).astype(np.float32)
                candidates.append(RewriteCandidate(kind="add", position=position))

        eligible_delete_indices = [
            idx for idx, patch in enumerate(model.patches)
            if current_step - int(getattr(patch, "creation_step", 0)) >= self.cooldown_steps
            and (
                self.loss_only_deletion
                or float(patch.compute_area().detach().cpu()) <= self.min_patch_area
            )
        ]
        for _ in range(delete_budget):
            if not eligible_delete_indices:
                break
            idx = int(np.random.choice(eligible_delete_indices))
            candidates.append(RewriteCandidate(kind="delete", patch_index=idx))

        eligible_split_indices = [
            idx for idx, patch in enumerate(model.patches)
            if current_step - int(getattr(patch, "creation_step", 0)) >= self.cooldown_steps
        ]
        for _ in range(split_budget):
            if not eligible_split_indices or len(model.patches) + 1 > self.max_patches:
                break
            idx = self._sample_split_index(model, eligible_split_indices)
            candidates.append(RewriteCandidate(kind="split", patch_index=idx))

        np.random.shuffle(candidates)
        return candidates[:self.candidate_count]

    def _uncovered_target_masks(self, model) -> tuple[np.ndarray, np.ndarray | None] | None:
        if self.swept_volume is None or self.disable_swept_volume_adds:
            return None

        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )

        target1 = model.target1_mask.detach().cpu().numpy().squeeze() > 0.5
        alpha1 = self._render_alpha_mask(render1)
        uncovered1 = target1 & ~alpha1

        uncovered2 = None
        if model.target2_mask is not None:
            target2 = model.target2_mask.detach().cpu().numpy().squeeze() > 0.5
            alpha2 = self._render_alpha_mask(render2)
            uncovered2 = target2 & ~alpha2

        return uncovered1, uncovered2

    @staticmethod
    def _render_alpha_mask(render: torch.Tensor, threshold: float = 0.05) -> np.ndarray:
        image = render.detach().cpu()
        if image.shape[-1] >= 4:
            alpha = image[..., 3]
        else:
            alpha = image[..., :3].amax(dim=-1)
        return alpha.numpy() > threshold

    def _sample_guided_swept_volume_position(
        self,
        model,
        uncovered_masks: tuple[np.ndarray, np.ndarray | None] | None,
        max_attempts: int = 128,
    ) -> np.ndarray | None:
        if self.swept_volume is None:
            return None
        if uncovered_masks is None or not self._has_uncovered_cells(uncovered_masks):
            return self._sample_swept_volume_position()

        best_position: np.ndarray | None = None
        best_score = -np.inf
        for _ in range(max(1, int(max_attempts))):
            position = self._sample_swept_volume_position()
            if not self._point_hits_uncovered(model, position, uncovered_masks):
                continue
            score = float(self.swept_volume.score_points(position[None, :])[0])
            if score > best_score:
                best_score = score
                best_position = position

        return best_position

    @staticmethod
    def _has_uncovered_cells(uncovered_masks: tuple[np.ndarray, np.ndarray | None]) -> bool:
        uncovered1, uncovered2 = uncovered_masks
        return bool(np.any(uncovered1) or (uncovered2 is not None and np.any(uncovered2)))

    def _point_hits_uncovered(
        self,
        model,
        point: np.ndarray,
        uncovered_masks: tuple[np.ndarray, np.ndarray | None],
    ) -> bool:
        uncovered1, uncovered2 = uncovered_masks
        pts = np.asarray(point, dtype=np.float32).reshape((1, 3))

        x1, y1, valid1 = _project_points(pts, model.camera1, uncovered1.shape)
        hit1 = bool(valid1[0] and uncovered1[y1[0], x1[0]])

        hit2 = False
        if uncovered2 is not None:
            x2, y2, valid2 = _project_points(pts, model.camera2, uncovered2.shape)
            hit2 = bool(valid2[0] and uncovered2[y2[0], x2[0]])

        return hit1 or hit2

    def _sample_swept_volume_position(self) -> np.ndarray:
        if self.swept_volume is None:
            raise ValueError("Swept-volume sampling requires a swept volume.")
        n_points = len(self.swept_volume.points)
        if n_points <= 0:
            raise ValueError("Cannot sample from an empty swept volume.")
        if (
            len(self._swept_point_order) != n_points
            or self._swept_point_cursor >= n_points
        ):
            self._swept_point_order = np.random.permutation(n_points)
            self._swept_point_cursor = 0
        point_index = int(self._swept_point_order[self._swept_point_cursor])
        self._swept_point_cursor += 1
        return self.swept_volume.sample_point_at_index(point_index)

    def _sample_split_index(self, model, eligible_indices: Sequence[int]) -> int:
        """Sample split candidates with larger patches more likely."""
        areas = np.array([
            max(1e-8, float(model.patches[idx].compute_area().detach().cpu()))
            for idx in eligible_indices
        ], dtype=np.float64)
        weights = areas / areas.sum()
        return int(np.random.choice(list(eligible_indices), p=weights))

    def _select_compatible(self, candidates: Sequence[RewriteCandidate]) -> list[RewriteCandidate]:
        accepted: list[RewriteCandidate] = []
        touched_indices: set[int] = set()
        restored_history: set[int] = set()
        additions = 0
        deletions = 0

        for candidate in sorted(candidates, key=lambda c: c.improvement, reverse=True):
            if candidate.kind == "delete":
                if candidate.patch_index is None or candidate.patch_index in touched_indices:
                    continue
                if deletions >= self.max_deletions:
                    continue
                touched_indices.add(candidate.patch_index)
                deletions += 1
                accepted.append(candidate)
                continue

            if candidate.kind == "split":
                if candidate.patch_index is None or candidate.patch_index in touched_indices:
                    continue
                if additions >= self.max_additions:
                    continue
                touched_indices.add(candidate.patch_index)
                additions += 1
                accepted.append(candidate)
                continue

            if additions >= self.max_additions:
                continue
            if candidate.kind == "restore":
                if candidate.history_index is None or candidate.history_index in restored_history:
                    continue
                restored_history.add(candidate.history_index)
            additions += 1
            accepted.append(candidate)

        return accepted

    def _apply_rewrites(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        rewrites: Sequence[RewriteCandidate],
        current_step: int,
    ) -> None:
        if not rewrites:
            return

        indexed_rewrites = [r for r in rewrites if r.kind in ("delete", "split")]
        for rewrite in sorted(indexed_rewrites, key=lambda r: r.patch_index or 0, reverse=True):
            self._apply_single(model, rewrite, current_step=current_step, tentative=False)
        restored_history_indices: list[int] = []
        for rewrite in [r for r in rewrites if r.kind not in ("delete", "split")]:
            self._apply_single(model, rewrite, current_step=current_step, tentative=False)
            if rewrite.kind == "restore" and rewrite.history_index is not None:
                restored_history_indices.append(rewrite.history_index)
        for history_index in sorted(set(restored_history_indices), reverse=True):
            if 0 <= history_index < len(self.deleted_history):
                self.deleted_history.pop(history_index)

        model.optim = self._rebuild_optimizer(model, optimizer)
        model._post_step_constraints()

    def _apply_single(
        self,
        model,
        rewrite: RewriteCandidate,
        *,
        current_step: int,
        tentative: bool,
    ) -> None:
        if rewrite.kind == "delete":
            if rewrite.patch_index is None or rewrite.patch_index >= len(model.patches):
                return
            patch = model.patches.pop(rewrite.patch_index)
            rewrite.applied_index = rewrite.patch_index
            if not tentative:
                self.deleted_history.append(patch.to_dict())
                self.stats.deleted += 1
                self.stats.total_deleted += 1
                pos = patch.center.detach().cpu().numpy()
                print(
                    f"[SRD rewrite] deleted patch={rewrite.patch_index}, "
                    f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                    f"reason={rewrite.reason or 'accepted rewrite'}"
                )
            return

        if rewrite.kind == "split":
            if rewrite.patch_index is None or rewrite.patch_index >= len(model.patches):
                return
            if len(model.patches) + 1 > self.max_patches:
                return
            patch = model.patches.pop(rewrite.patch_index)
            child_a, child_b = patch.split_down_middle(creation_step=current_step)
            if child_a.is_self_intersecting() or child_b.is_self_intersecting():
                model.patches.insert(rewrite.patch_index, patch)
                return
            model.patches.extend([child_a, child_b])
            rewrite.applied_index = rewrite.patch_index
            if not tentative:
                self.deleted_history.append(patch.to_dict())
                self.stats.added += 1
                self.stats.total_added += 1
                print(
                    f"[SRD rewrite] split patch={rewrite.patch_index}, "
                    f"children={len(model.patches) - 2},{len(model.patches) - 1}, "
                    f"improvement={rewrite.improvement:.6f}"
                )
            return

        if rewrite.kind == "restore" and rewrite.patch_state is not None:
            patch = Patch.from_dict(copy.deepcopy(rewrite.patch_state), device=model.device)
            if patch.is_self_intersecting():
                return
            patch.creation_step = current_step
            model.patches.append(patch)
            rewrite.applied_index = len(model.patches) - 1
            if not tentative:
                self.stats.added += 1
                self.stats.total_added += 1
                pos = patch.center.detach().cpu().numpy()
                print(
                    f"[SRD rewrite] restored patch={rewrite.applied_index}, "
                    f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                    f"improvement={rewrite.improvement:.6f}"
                )
            return

        if rewrite.position is None:
            return
        if len(model.patches) >= self.max_patches:
            return
        palette_color = [1.0, 1.0, 1.0]
        patch = _small_default_patch(
            rewrite.position,
            model.device,
            palette_color,
            current_step,
            label=f"patch_{len(model.patches):04d}",
        )
        model.patches.append(patch)
        rewrite.applied_index = len(model.patches) - 1
        if not tentative:
            self.stats.added += 1
            self.stats.total_added += 1
            pos = patch.center.detach().cpu().numpy()
            print(
                f"[SRD rewrite] added patch={rewrite.applied_index}, "
                f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                f"improvement={rewrite.improvement:.6f}"
            )

    def _save_state(self, model, optimizer: torch.optim.Optimizer) -> tuple[list[dict], dict]:
        return [patch.to_dict() for patch in model.patches], copy.deepcopy(optimizer.state_dict())

    def _restore_state(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        patch_states: Sequence[dict],
        optimizer_state: dict,
    ) -> None:
        model.patches[:] = [
            Patch.from_dict(copy.deepcopy(state), device=model.device)
            for state in patch_states
        ]
        model.optim = self._rebuild_optimizer(model, optimizer)
        try:
            model.optim.load_state_dict(optimizer_state)
        except ValueError:
            pass
        model._post_step_constraints()

    def _rebuild_optimizer(self, model, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        defaults = optimizer.defaults.copy()
        return optimizer.__class__(_patch_parameters(model.patches), **defaults)
