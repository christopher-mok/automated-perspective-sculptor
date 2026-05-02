"""JSON export helpers for optimized patch pieces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, request

import torch

from core.patch import ControlPoint, Patch

EXPORT_JSON_PATH = Path("exports/pieces.json")


def _tensor_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _rgb_to_hex(rgb: list[float]) -> str:
    channels = [max(0, min(255, int(round(c * 255.0)))) for c in rgb[:3]]
    return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}"


def _control_point_dict(cp: ControlPoint) -> dict[str, Any]:
    handle_in = cp.handle_in().detach().cpu().numpy()
    handle_out = cp.handle_out().detach().cpu().numpy()
    return {
        "x": _tensor_scalar(cp.x),
        "y": _tensor_scalar(cp.y),
        "handleIn": {
            "x": float(handle_in[0]),
            "y": float(handle_in[1]),
        },
        "handleOut": {
            "x": float(handle_out[0]),
            "y": float(handle_out[1]),
        },
    }


def patch_to_piece_dict(
    patch: Patch,
    piece_id: str,
    *,
    piece_scale: float = 1.0,
) -> dict[str, Any]:
    center = patch.center.detach().cpu().numpy()
    color = patch.albedo.detach().cpu().clamp(0.0, 1.0).numpy().tolist()
    return {
        "id": piece_id,
        "position": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        },
        "scale": float(piece_scale),
        "theta": _tensor_scalar(patch.theta),
        "color": _rgb_to_hex(color),
        "controlPoints": [
            _control_point_dict(cp)
            for cp in patch.control_points
        ],
    }


def build_export_payload(
    patches: list[Patch],
    *,
    scale: float = 1.0,
    piece_scale: float = 1.0,
    id_prefix: str = "P",
    id_width: int = 2,
) -> dict[str, Any]:
    pieces = [
        patch_to_piece_dict(
            patch,
            f"{id_prefix}{idx:0{id_width}d}",
            piece_scale=piece_scale,
        )
        for idx, patch in enumerate(patches, start=1)
    ]
    return {
        "scale": float(scale),
        "pieces": pieces,
    }


def write_export_json(
    payload: dict[str, Any],
    *,
    indent: int = 2,
) -> Path:
    path = EXPORT_JSON_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=indent)}\n", encoding="utf-8")
    return path


def export_patches_to_json(
    patches: list[Patch],
    *,
    scale: float = 1.0,
    piece_scale: float = 1.0,
    id_prefix: str = "P",
    id_width: int = 2,
    indent: int = 2,
) -> Path:
    payload = build_export_payload(
        patches,
        scale=scale,
        piece_scale=piece_scale,
        id_prefix=id_prefix,
        id_width=id_width,
    )
    return write_export_json(payload, indent=indent)


def send_export_payload(
    payload: dict[str, Any],
    endpoint: str,
    *,
    timeout_s: float = 15.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST an export payload to an API endpoint."""
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    body_bytes = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=body_bytes, headers=req_headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            try:
                response_body: Any = json.loads(response_text) if response_text else None
            except json.JSONDecodeError:
                response_body = response_text
            return {
                "ok": True,
                "status": int(response.status),
                "body": response_body,
            }
    except error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        try:
            response_body = json.loads(response_text) if response_text else None
        except json.JSONDecodeError:
            response_body = response_text
        return {
            "ok": False,
            "status": int(exc.code),
            "body": response_body,
        }
    except error.URLError as exc:
        return {
            "ok": False,
            "status": None,
            "body": str(exc.reason),
        }
