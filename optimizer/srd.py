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

if TYPE_CHECKING:
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


def _near_zero_patch(
    position: np.ndarray,
    device: str,
    albedo: Sequence[float],
    creation_step: int,
    label: str,
) -> Patch:
    control_points: list[ControlPoint] = []
    radius = 0.001
    for idx in range(Patch.N_CONTROL_POINTS):
        angle = 2.0 * math.pi * idx / Patch.N_CONTROL_POINTS - math.pi / 2.0
        control_points.append(ControlPoint(
            x=radius * math.cos(angle),
            y=radius * math.sin(angle),
            z=0.0,
            handle_scale=0.01,
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
        min_patch_area: float = 0.001,
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

        mandatory_deletes = self._mandatory_delete_rewrites(model, current_step)
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

        candidates = self._sample_rewrites(model, current_step)
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
        for candidate in mandatory_deletes:
            patch_ref = candidate.applied_index if candidate.applied_index is not None else candidate.patch_index
            print(f"  Mandatory {candidate.label} at patch {patch_ref}, reason={candidate.reason}")
        print(
            f"  Total patches: {len(model.patches)}, total adds: {self.stats.total_added}, "
            f"total deletes: {self.stats.total_deleted}"
        )
        return self.stats

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
        if self._patch_violates_rules(model, patch_index):
            reasons.append("rule violation")
        if not self._patch_contributes_to_either_image(model, patch_index):
            reasons.append("no image contribution")
        if not self._deleting_patch_changes_image(model, patch_index):
            reasons.append("delete has no image effect")
        return ", ".join(reasons)

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
        with torch.no_grad():
            _, components = model._loss_from_renders(
                *model.renderer.render_both(
                    [patch],
                    model.camera1,
                    model.camera2,
                    model.render_resolutions,
                ),
                [patch],
            )
        return (
            float(components["camera_bounds"].detach().cpu()) > self.rule_violation_tol
            or float(components["visibility"].detach().cpu()) > self.rule_violation_tol
        )

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
                model._post_step_constraints()

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

    def _sample_rewrites(self, model, current_step: int) -> list[RewriteCandidate]:
        candidates: list[RewriteCandidate] = []
        add_budget = int(round(self.candidate_count * 0.35))
        delete_budget = int(round(self.candidate_count * 0.15))
        split_budget = self.candidate_count - add_budget - delete_budget

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
                position = np.random.uniform(
                    -self.scene_box_size * 0.5,
                    self.scene_box_size * 0.5,
                    size=3,
                ).astype(np.float32)
                candidates.append(RewriteCandidate(kind="add", position=position))

        eligible_delete_indices = [
            idx for idx, patch in enumerate(model.patches)
            if current_step - int(getattr(patch, "creation_step", 0)) >= self.cooldown_steps
            and float(patch.compute_area().detach().cpu()) <= self.min_patch_area
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
            return

        if rewrite.kind == "split":
            if rewrite.patch_index is None or rewrite.patch_index >= len(model.patches):
                return
            if len(model.patches) + 1 > self.max_patches:
                return
            patch = model.patches.pop(rewrite.patch_index)
            child_a, child_b = patch.split_down_middle(creation_step=current_step)
            model.patches.extend([child_a, child_b])
            rewrite.applied_index = rewrite.patch_index
            if not tentative:
                self.deleted_history.append(patch.to_dict())
                self.stats.added += 1
                self.stats.total_added += 1
            return

        if rewrite.kind == "restore" and rewrite.patch_state is not None:
            patch = Patch.from_dict(copy.deepcopy(rewrite.patch_state), device=model.device)
            patch.creation_step = current_step
            model.patches.append(patch)
            rewrite.applied_index = len(model.patches) - 1
            if not tentative:
                self.stats.added += 1
                self.stats.total_added += 1
            return

        if rewrite.position is None:
            return
        if len(model.patches) >= self.max_patches:
            return
        palette_color = model.palette[-1].detach().cpu().numpy().tolist()
        patch = _near_zero_patch(
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

    def _save_state(self, model, optimizer: torch.optim.Optimizer) -> tuple[list[dict], dict]:
        return [patch.to_dict() for patch in model.patches], copy.deepcopy(optimizer.state_dict())

    def _restore_state(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        patch_states: Sequence[dict],
        optimizer_state: dict,
    ) -> None:
        model.patches = [Patch.from_dict(copy.deepcopy(state), device=model.device) for state in patch_states]
        model.optim = self._rebuild_optimizer(model, optimizer)
        try:
            model.optim.load_state_dict(optimizer_state)
        except ValueError:
            pass
        model._post_step_constraints()

    def _rebuild_optimizer(self, model, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        defaults = optimizer.defaults.copy()
        return optimizer.__class__(_patch_parameters(model.patches), **defaults)
