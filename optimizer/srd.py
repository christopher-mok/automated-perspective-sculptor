"""Stochastic Rewrite Descent for adaptive patch structure."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from core.patch import ControlPoint, Patch

if TYPE_CHECKING:
    from core.optimizer import SceneOptimizer


RewriteType = Literal["add", "delete", "restore"]


@dataclass
class SRDConfig:
    enabled: bool = True
    lr: float = 1e-3
    propose_every: int = 50
    proposal_trigger: Literal["step", "stagnation"] = "step"
    proposal_patience: int = 10
    proposal_rel_loss: float = 1e-4
    proposal_steps: int = 5
    num_candidates: int = 64
    better_abs_eps: float = 1e-5
    better_rel_eps: float = 1e-3
    parallel_accept: bool = True
    w_simplicity: float = 0.1
    cleanup_interval: int = 25
    max_patches: int = 200
    min_patches: int = 4
    max_handle_scale: float = 2.0
    scene_min_x: float = -2.5
    scene_max_x: float = 2.5
    scene_min_y: float = -2.5
    scene_max_y: float = 2.5
    scene_min_z: float = -2.5
    scene_max_z: float = 2.5


@dataclass
class SRDStats:
    active: int = 0
    added: int = 0
    deleted: int = 0
    mandatory_deleted: int = 0
    total_adds: int = 0
    total_deletes: int = 0
    total_mandatory_deletes: int = 0
    evaluated: int = 0
    promising: int = 0
    accepted: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "srd_active_patches": float(self.active),
            "srd_added": float(self.added),
            "srd_deleted": float(self.deleted),
            "srd_mandatory_deleted": float(self.mandatory_deleted),
            "srd_total_adds": float(self.total_adds),
            "srd_total_deletes": float(self.total_deletes),
            "srd_total_mandatory_deletes": float(self.total_mandatory_deletes),
            "srd_evaluated": float(self.evaluated),
            "srd_promising": float(self.promising),
            "srd_accepted": float(self.accepted),
        }


@dataclass
class RewriteSpec:
    type: RewriteType
    index: int | None = None
    patch_state: dict | None = None
    history_index: int | None = None
    reason: str = ""


def patch_parameters(patches: Sequence[Patch]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for patch in patches:
        params.extend([patch.center, patch.theta])
        for cp in patch.control_points:
            cp.z.requires_grad_(False)
            cp.z.grad = None
            params.extend([cp.x, cp.y, cp.handle_scale, cp.handle_rotation])
    return params


def deep_copy_patches(patches: Sequence[Patch], device: str) -> list[Patch]:
    copied = [Patch.from_dict(copy.deepcopy(patch.to_dict()), device=device) for patch in patches]
    for patch in copied:
        for param in patch.parameters():
            param.requires_grad_(True)
    return copied


def save_patch_state(patch: Patch) -> dict:
    return copy.deepcopy(patch.to_dict())


def create_patch_from_saved(state: dict, device: str) -> Patch:
    return Patch.from_dict(copy.deepcopy(state), device=device)


def create_near_zero_patch(
    *,
    position: Sequence[float],
    device: str,
    albedo: Sequence[float],
    creation_step: int,
    label: str,
) -> Patch:
    radius = 0.001
    control_points: list[ControlPoint] = []
    for i in range(Patch.N_CONTROL_POINTS):
        angle = 2.0 * math.pi * i / Patch.N_CONTROL_POINTS - math.pi / 2.0
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
        center=list(position),
        theta=0.0,
        albedo=list(albedo),
        device=device,
        label=label,
    )
    patch.creation_step = creation_step
    patch.self_intersect_counter = 0
    return patch


class SRDOptimizer:
    """Hybrid optimizer that proposes add/delete rewrites around gradient descent."""

    def __init__(self, model: "SceneOptimizer", cameras, targets, config: SRDConfig) -> None:
        self.model = model
        self.cameras = cameras
        self.targets = targets
        self.config = config

        self.optimizer = torch.optim.Adam(patch_parameters(model.patches), lr=config.lr)

        self.propose_every = config.propose_every
        self.proposal_trigger = config.proposal_trigger
        self.proposal_patience = config.proposal_patience
        self.proposal_rel_loss = config.proposal_rel_loss
        self.proposal_steps = config.proposal_steps
        self.num_candidates = config.num_candidates
        self.better_abs_eps = config.better_abs_eps
        self.better_rel_eps = config.better_rel_eps
        self.parallel_accept = config.parallel_accept
        self.w_simplicity = config.w_simplicity
        self.cleanup_interval = config.cleanup_interval

        self.loss_ema: float | None = None
        self.patience_counter = 0
        self.loss_history: list[float] = []
        self.deleted_patches: list[dict] = []

        self.total_adds = 0
        self.total_deletes = 0
        self.total_mandatory_deletes = 0
        self.stats = SRDStats(active=len(model.patches))

    def sync_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(patch_parameters(self.model.patches), lr=self.config.lr)
        self.model.optim = self.optimizer

    def step(self, current_step: int) -> dict[str, float]:
        self.stats.added = 0
        self.stats.deleted = 0
        self.stats.mandatory_deleted = 0
        self.stats.evaluated = 0
        self.stats.promising = 0
        self.stats.accepted = 0

        if self.cleanup_interval > 0 and current_step % self.cleanup_interval == 0:
            self._mandatory_cleanup(current_step)

        if self._should_propose(current_step):
            self._rewrite_step(current_step)

        metrics = self._continuous_step()
        self._project_to_valid(self.model.patches)
        self._update_self_intersect_counters()
        self._update_loss_history(metrics["loss"])

        self.stats.active = len(self.model.patches)
        metrics.update(self.stats.as_dict())
        return metrics

    def _continuous_step(self) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        render1, render2 = self.model.renderer.render_both(
            self.model.patches,
            self.model.camera1,
            self.model.camera2,
            self.model.resolution,
        )
        loss, components = self.model._loss_from_renders(render1, render2, self.model.patches)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model._post_step_constraints()
        metrics = self.model._metrics_from_components(loss, components)
        return metrics

    def _compute_task_loss_for_patches(self, patches: Sequence[Patch]) -> torch.Tensor:
        render1, render2 = self.model.renderer.render_both(
            list(patches),
            self.model.camera1,
            self.model.camera2,
            self.model.resolution,
        )
        loss, _ = self.model._loss_from_renders(render1, render2, patches)
        return loss

    def _compute_score_for_patches(self, patches: Sequence[Patch]) -> float:
        with torch.no_grad():
            task_loss = self._compute_task_loss_for_patches(patches)
        return float(task_loss.detach().cpu()) + self.w_simplicity * len(patches)

    def _should_propose(self, current_step: int) -> bool:
        if not self.config.enabled:
            return False
        if self.proposal_trigger == "step":
            return self.propose_every > 0 and current_step % self.propose_every == 0
        if self.loss_ema is None:
            return False
        current = self.loss_history[-1] if self.loss_history else float("inf")
        if self.loss_ema - current < self.proposal_rel_loss * self.loss_ema:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
        if self.patience_counter >= self.proposal_patience:
            self.patience_counter = 0
            return True
        return False

    def _rewrite_step(self, current_step: int) -> None:
        base_score = self._compute_score_for_patches(self.model.patches)
        proposals = self._generate_proposals(current_step)
        scored: list[tuple[list[Patch], RewriteSpec, float, float]] = []

        for proposal_patches, spec in proposals:
            self._briefly_optimize(proposal_patches)
            proposal_score = self._compute_score_for_patches(proposal_patches)
            improvement = base_score - proposal_score
            scored.append((proposal_patches, spec, improvement, proposal_score))
            self.stats.evaluated += 1

        accepted = [
            (patches, spec, improvement)
            for patches, spec, improvement, _score in scored
            if improvement > self.better_abs_eps or improvement > self.better_rel_eps * abs(base_score)
        ]
        self.stats.promising = len(accepted)

        if not accepted:
            print(f"SRD step {current_step}: {len(proposals)} candidates, none accepted")
            return

        selected = self._greedy_select_compatible(accepted) if self.parallel_accept else [
            max(accepted, key=lambda item: item[2])
        ]
        changed = self._apply_rewrites(selected, current_step)
        self.stats.accepted = len(selected) if changed else 0
        if changed:
            self.sync_optimizer()

        print(
            f"SRD step {current_step}: {len(proposals)} candidates, "
            f"{len(accepted)} promising, {self.stats.accepted} accepted"
        )
        for _patches, spec, improvement in selected:
            print(f"  Accepted {spec.type} patch {spec.index if spec.index is not None else 'new'}, improvement={improvement:.6f}")
        print(
            f"  Total patches: {len(self.model.patches)}, adds: {self.total_adds}, deletes: {self.total_deletes}"
        )

    def _briefly_optimize(self, patches: Sequence[Patch]) -> None:
        params = patch_parameters(patches)
        if not params:
            return
        temp_optimizer = torch.optim.Adam(params, lr=self.config.lr)
        for _ in range(max(0, self.proposal_steps)):
            temp_optimizer.zero_grad(set_to_none=True)
            loss = self._compute_task_loss_for_patches(patches)
            loss.backward()
            temp_optimizer.step()
            temp_optimizer.zero_grad(set_to_none=True)
            self._project_to_valid(patches)

    def _generate_proposals(self, current_step: int) -> list[tuple[list[Patch], RewriteSpec]]:
        proposals: list[tuple[list[Patch], RewriteSpec]] = []
        num_patches = len(self.model.patches)

        for _ in range(self.num_candidates):
            roll = random.random()

            if roll < 0.4 and num_patches > self.config.min_patches:
                idx = random.randint(0, num_patches - 1)
                patch = self.model.patches[idx]
                if current_step - int(getattr(patch, "creation_step", 0)) < 30:
                    continue
                proposal = deep_copy_patches(self.model.patches, self.model.device)
                del proposal[idx]
                proposals.append((proposal, RewriteSpec(type="delete", index=idx)))
                continue

            if roll < 0.8 and num_patches < self.config.max_patches:
                proposal = deep_copy_patches(self.model.patches, self.model.device)
                position = [
                    random.uniform(self.config.scene_min_x, self.config.scene_max_x),
                    random.uniform(self.config.scene_min_y, self.config.scene_max_y),
                    random.uniform(self.config.scene_min_z, self.config.scene_max_z),
                ]
                proposal.append(create_near_zero_patch(
                    position=position,
                    device=self.model.device,
                    albedo=self._default_albedo(),
                    creation_step=current_step,
                    label=f"patch_{len(proposal):04d}",
                ))
                proposals.append((proposal, RewriteSpec(
                    type="add",
                    index=len(proposal) - 1,
                    patch_state=save_patch_state(proposal[-1]),
                )))
                continue

            if self.deleted_patches and num_patches < self.config.max_patches:
                hist_idx = random.randrange(len(self.deleted_patches))
                proposal = deep_copy_patches(self.model.patches, self.model.device)
                restored = create_patch_from_saved(self.deleted_patches[hist_idx], self.model.device)
                restored.creation_step = current_step
                proposal.append(restored)
                proposals.append((proposal, RewriteSpec(
                    type="restore",
                    index=len(proposal) - 1,
                    history_index=hist_idx,
                    patch_state=save_patch_state(restored),
                )))

        return proposals

    def _greedy_select_compatible(
        self,
        accepted: list[tuple[list[Patch], RewriteSpec, float]],
    ) -> list[tuple[list[Patch], RewriteSpec, float]]:
        accepted.sort(key=lambda x: x[2], reverse=True)
        selected: list[tuple[list[Patch], RewriteSpec, float]] = []
        used_indices: set[int] = set()

        for patches, spec, improvement in accepted:
            idx = spec.index
            if spec.type == "delete" and idx is not None and idx in used_indices:
                continue
            selected.append((patches, spec, improvement))
            if idx is not None:
                used_indices.add(idx)
        return selected

    def _apply_rewrites(
        self,
        selected: Sequence[tuple[list[Patch], RewriteSpec, float]],
        current_step: int,
    ) -> bool:
        changed = False
        deletes = sorted(
            [(spec, improvement) for _patches, spec, improvement in selected if spec.type == "delete"],
            key=lambda item: item[0].index if item[0].index is not None else -1,
            reverse=True,
        )
        for spec, _improvement in deletes:
            idx = spec.index
            if idx is None or idx >= len(self.model.patches) or len(self.model.patches) <= self.config.min_patches:
                continue
            self.deleted_patches.append(save_patch_state(self.model.patches[idx]))
            self.deleted_patches = self.deleted_patches[-100:]
            del self.model.patches[idx]
            self.total_deletes += 1
            self.stats.deleted += 1
            changed = True

        for proposal_patches, spec, _improvement in selected:
            if spec.type not in ("add", "restore") or len(self.model.patches) >= self.config.max_patches:
                continue
            source_idx = spec.index
            if source_idx is not None and 0 <= source_idx < len(proposal_patches):
                patch = Patch.from_dict(save_patch_state(proposal_patches[source_idx]), device=self.model.device)
            elif spec.patch_state is not None:
                patch = create_patch_from_saved(spec.patch_state, self.model.device)
            else:
                continue
            patch.creation_step = current_step
            patch.self_intersect_counter = 0
            self.model.patches.append(patch)
            self.total_adds += 1
            self.stats.added += 1
            changed = True

        if changed:
            self.model._post_step_constraints()
        return changed

    def _mandatory_cleanup(self, current_step: int) -> None:
        if len(self.model.patches) <= self.config.min_patches:
            return

        to_delete: list[tuple[int, str]] = []
        for i, patch in enumerate(self.model.patches):
            if len(self.model.patches) - len(to_delete) <= self.config.min_patches:
                break
            reason = self._mandatory_delete_reason(i, patch, current_step)
            if reason is not None:
                to_delete.append((i, reason))

        for idx, reason in sorted(to_delete, key=lambda x: x[0], reverse=True):
            print(f"MANDATORY DELETE patch {idx}: {reason}")
            self.deleted_patches.append(save_patch_state(self.model.patches[idx]))
            self.deleted_patches = self.deleted_patches[-100:]
            del self.model.patches[idx]
            self.total_mandatory_deletes += 1
            self.stats.mandatory_deleted += 1

        if to_delete:
            self.sync_optimizer()

    def _mandatory_delete_reason(self, index: int, patch: Patch, current_step: int) -> str | None:
        for param in patch.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return "NaN/Inf in parameters"

        for cp in patch.control_points:
            if float(cp.handle_scale.detach().cpu()) <= 0.0:
                return "Non-positive handle scale"

        if int(getattr(patch, "self_intersect_counter", 0)) > 30:
            return "Persistent self-intersection (30+ steps)"

        if current_step - int(getattr(patch, "creation_step", 0)) < 30:
            return None

        if self._is_cheap_candidate_for_removal(index):
            if not self._deleting_patch_changes_image(index):
                return "Redundant - removing changes nothing in either view"
            if not self._patch_contributes_to_loss(index):
                return "Zero contribution to loss"
        return None

    def _is_cheap_candidate_for_removal(self, index: int) -> bool:
        areas = [float(p.compute_area().detach().cpu()) for p in self.model.patches]
        if not areas:
            return False
        threshold = float(np.percentile(np.array(areas, dtype=np.float32), 25.0))
        return areas[index] <= threshold

    def _deleting_patch_changes_image(self, index: int) -> bool:
        with torch.no_grad():
            full1, full2 = self.model.renderer.render_both(
                self.model.patches,
                self.model.camera1,
                self.model.camera2,
                self.model.resolution,
            )
            remaining = [patch for i, patch in enumerate(self.model.patches) if i != index]
            without1, without2 = self.model.renderer.render_both(
                remaining,
                self.model.camera1,
                self.model.camera2,
                self.model.resolution,
            )
            diff = torch.maximum((full1 - without1).abs().max(), (full2 - without2).abs().max())
        return float(diff.detach().cpu()) >= 1e-4

    def _patch_contributes_to_loss(self, index: int) -> bool:
        loss_with = self._compute_score_for_patches(self.model.patches)
        remaining = [patch for i, patch in enumerate(self.model.patches) if i != index]
        loss_without = self._compute_score_for_patches(remaining)
        return abs(loss_with - loss_without) >= 1e-6

    def _project_to_valid(self, patches: Sequence[Patch]) -> None:
        with torch.no_grad():
            for patch in patches:
                patch.center.data[0].clamp_(self.config.scene_min_x, self.config.scene_max_x)
                patch.center.data[1].clamp_(self.config.scene_min_y, self.config.scene_max_y)
                patch.center.data[2].clamp_(self.config.scene_min_z, self.config.scene_max_z)
                for cp in patch.control_points:
                    cp.z.data.zero_()
                    cp.handle_scale.data.clamp_(min=0.01, max=self.config.max_handle_scale)
            if patches is self.model.patches:
                self.model._post_step_constraints()

    def _update_self_intersect_counters(self) -> None:
        for patch in self.model.patches:
            if patch.is_self_intersecting():
                patch.self_intersect_counter += 1
            else:
                patch.self_intersect_counter = 0

    def _update_loss_history(self, loss_val: float) -> None:
        self.loss_history.append(loss_val)
        alpha = 0.01
        if self.loss_ema is None:
            self.loss_ema = loss_val
        else:
            self.loss_ema = alpha * loss_val + (1 - alpha) * self.loss_ema

    def _default_albedo(self) -> list[float]:
        if not self.model.patches:
            return [1.0, 1.0, 1.0]
        return self.model.patches[0].albedo.detach().cpu().clamp(0.0, 1.0).numpy().tolist()
