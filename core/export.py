"""JSON export helpers for optimized patch pieces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib import error, request
from xml.sax.saxutils import escape

import numpy as np
import torch

from core.patch import ControlPoint, Patch

EXPORT_JSON_PATH = Path("exports/pieces.json")
EXPORT_GRID_SVG_PATH = Path("exports/grid.svg")
EXPORT_PIECES_SVG_PATH = Path("exports/pieces.svg")
STRING_CONNECTION_IDS = ("left", "right")
SCENE_UNIT_TO_INCH = 2.4
INCH_TO_SCENE_UNIT = 1.0 / SCENE_UNIT_TO_INCH
SVG_RED = "rgb(255,0,0)"
SVG_BLUE = "rgb(0,0,255)"
SVG_BLACK = "rgb(0,0,0)"
SVG_STROKE_IN = 0.01
SVG_HOLE_RADIUS_IN = 0.035
SVG_FONT_SIZE_IN = 0.16
SVG_SMALL_FONT_SIZE_IN = 0.12


def _tensor_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _rgb_to_hex(rgb: list[float]) -> str:
    channels = [max(0, min(255, int(round(c * 255.0)))) for c in rgb[:3]]
    return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}"


def _hex_to_rgb(value: str) -> list[float]:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError(f"Expected #rrggbb colour, got {value!r}.")
    return [
        int(text[idx:idx + 2], 16) / 255.0
        for idx in (0, 2, 4)
    ]


def _control_point_dict(
    cp: ControlPoint,
    *,
    offset_xy: torch.Tensor | None = None,
) -> dict[str, Any]:
    handle_in = cp.handle_in().detach().cpu().numpy()
    handle_out = cp.handle_out().detach().cpu().numpy()
    point_xy = torch.stack([cp.x, cp.y]).detach().cpu()
    if offset_xy is not None:
        point_xy = point_xy - offset_xy.detach().cpu()
    return {
        "x": float(point_xy[0]),
        "y": float(point_xy[1]),
        "handleIn": {
            "x": float(handle_in[0]),
            "y": float(handle_in[1]),
        },
        "handleOut": {
            "x": float(handle_out[0]),
            "y": float(handle_out[1]),
        },
    }


def _xyz_dict(point: torch.Tensor | list[float] | tuple[float, float, float]) -> dict[str, float]:
    if isinstance(point, torch.Tensor):
        values = point.detach().cpu().numpy().tolist()
    else:
        values = list(point)
    return {
        "x": float(values[0]),
        "y": float(values[1]),
        "z": float(values[2]),
    }


def _xy_dict(point: torch.Tensor | list[float] | tuple[float, float]) -> dict[str, float]:
    if isinstance(point, torch.Tensor):
        values = point.detach().cpu().numpy().tolist()
    else:
        values = list(point)
    return {
        "x": float(values[0]),
        "y": float(values[1]),
    }


def _fmt(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _piece_label(index: int) -> str:
    return f"P{index:02d}"


def _svg_document(width_in: float, height_in: float, body: list[str]) -> str:
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{_fmt(width_in)}in" height="{_fmt(height_in)}in" '
            f'viewBox="0 0 {_fmt(width_in)} {_fmt(height_in)}">'
        ),
        *body,
        "</svg>",
        "",
    ])


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _export_anchor_xy(patch: Patch) -> torch.Tensor:
    points = [
        torch.stack([cp.x, cp.y])
        for cp in patch.control_points
    ]
    return torch.stack(points, dim=0).mean(dim=0).detach()


def _anchored_export_center(patch: Patch, anchor_xy: torch.Tensor) -> torch.Tensor:
    local_anchor = torch.stack([
        anchor_xy.to(device=patch.center.device, dtype=patch.center.dtype)[0],
        anchor_xy.to(device=patch.center.device, dtype=patch.center.dtype)[1],
        torch.zeros((), device=patch.center.device, dtype=patch.center.dtype),
    ])
    return (patch.rotation_matrix() @ local_anchor) + patch.center


def _patch_string_connections(patch: Patch) -> list[dict[str, Any]]:
    raw_connections = getattr(patch, "string_connections", None)
    if not raw_connections:
        return []
    return list(raw_connections)


def _piece_string_connection_dict(
    connection: dict[str, Any],
    *,
    anchor_xy: torch.Tensor,
) -> dict[str, Any]:
    local_point = connection.get("pieceLocalPoint", {})
    local_xy = torch.tensor(
        [
            float(local_point.get("x", 0.0)),
            float(local_point.get("y", 0.0)),
        ],
        dtype=anchor_xy.dtype,
    ) - anchor_xy.detach().cpu()
    center_of_mass = connection.get("pieceCenterOfMass", {})
    center_xy = torch.tensor(
        [
            float(center_of_mass.get("x", 0.0)),
            float(center_of_mass.get("y", 0.0)),
        ],
        dtype=anchor_xy.dtype,
    ) - anchor_xy.detach().cpu()
    return {
        "id": str(connection["id"]),
        "placementMethod": str(connection.get("placementMethod", "")),
        "pieceCenterOfMass": _xy_dict(center_xy),
        "pieceLocalPoint": _xy_dict(local_xy),
        "pieceWorldPoint": connection["pieceWorldPoint"],
        "boardConnectionId": str(connection["boardConnectionId"]),
        "boardPoint": connection["boardPoint"],
        "stringId": str(connection["stringId"]),
        "length": float(connection["length"]),
    }


def patch_to_piece_dict(
    patch: Patch,
    piece_id: str,
    *,
    piece_scale: float = 1.0,
) -> dict[str, Any]:
    anchor_xy = _export_anchor_xy(patch)
    center = _anchored_export_center(patch, anchor_xy).detach().cpu().numpy()
    color = patch.albedo.detach().cpu().clamp(0.0, 1.0).numpy().tolist()
    piece = {
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
            _control_point_dict(cp, offset_xy=anchor_xy)
            for cp in patch.control_points
        ],
    }
    string_connections = [
        _piece_string_connection_dict(connection, anchor_xy=anchor_xy)
        for connection in _patch_string_connections(patch)
    ]
    if string_connections:
        piece["stringConnections"] = string_connections
    return piece


def _payload_string_mappings(patches: list[Patch], piece_ids: list[str]) -> list[dict[str, Any]]:
    strings: list[dict[str, Any]] = []
    for patch, piece_id in zip(patches, piece_ids):
        for connection in _patch_string_connections(patch):
            strings.append({
                "id": str(connection["stringId"]),
                "pieceId": piece_id,
                "pieceConnectionId": str(connection["id"]),
                "pieceWorldPoint": connection["pieceWorldPoint"],
                "boardConnectionId": str(connection["boardConnectionId"]),
                "boardPoint": connection["boardPoint"],
                "length": float(connection["length"]),
            })
    return strings


def build_export_payload(
    patches: list[Patch],
    *,
    scale: float = 1.0,
    hanging_plane_size: float | None = None,
    piece_scale: float = 1.0,
    id_prefix: str = "P",
    id_width: int = 2,
) -> dict[str, Any]:
    piece_ids = [
        f"{id_prefix}{idx:0{id_width}d}"
        for idx in range(1, len(patches) + 1)
    ]
    pieces = [
        patch_to_piece_dict(
            patch,
            piece_id,
            piece_scale=piece_scale,
        )
        for patch, piece_id in zip(patches, piece_ids)
    ]
    payload: dict[str, Any] = {
        "scale": float(scale),
        "pieces": pieces,
    }
    if hanging_plane_size is not None:
        payload["hangingPlaneSize"] = float(hanging_plane_size)
    string_mappings = _payload_string_mappings(patches, piece_ids)
    if string_mappings:
        payload["strings"] = string_mappings
        payload["hangingBoardConnections"] = [
            {
                "id": mapping["boardConnectionId"],
                "point": mapping["boardPoint"],
                "stringId": mapping["id"],
                "pieceId": mapping["pieceId"],
                "pieceConnectionId": mapping["pieceConnectionId"],
            }
            for mapping in string_mappings
        ]
    return payload


def _polygon_centroid_xy(xy: torch.Tensor) -> torch.Tensor:
    nxt = torch.roll(xy, shifts=-1, dims=0)
    cross = xy[:, 0] * nxt[:, 1] - nxt[:, 0] * xy[:, 1]
    area_twice = cross.sum()
    if float(area_twice.abs()) <= 1e-8:
        return xy.mean(dim=0)
    centroid = torch.stack([
        ((xy[:, 0] + nxt[:, 0]) * cross).sum(),
        ((xy[:, 1] + nxt[:, 1]) * cross).sum(),
    ]) / (3.0 * area_twice)
    return centroid


def _upper_boundary_point_at_x(xy: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
    target = float(target_x.detach().cpu())
    candidates: list[torch.Tensor] = []
    eps = 1e-7
    for idx in range(len(xy)):
        p0 = xy[idx]
        p1 = xy[(idx + 1) % len(xy)]
        x0 = float(p0[0].detach().cpu())
        x1 = float(p1[0].detach().cpu())
        if abs(x1 - x0) <= eps:
            if abs(target - x0) <= eps:
                candidates.extend([p0, p1])
            continue

        if target < min(x0, x1) - eps or target > max(x0, x1) + eps:
            continue

        t = max(0.0, min(1.0, (target - x0) / (x1 - x0)))
        candidates.append(p0 + (p1 - p0) * xy.new_tensor(t))

    if candidates:
        return max(candidates, key=lambda point: float(point[1].detach().cpu()))

    y_range = float((xy[:, 1].max() - xy[:, 1].min()).detach().cpu())
    top_band = max(y_range * 0.03, 1e-4)
    max_y = xy[:, 1].max()
    top_points = xy[(max_y - xy[:, 1]) <= top_band]
    if len(top_points) == 0:
        top_points = xy
    distances = (top_points[:, 0] - target_x).abs()
    return top_points[int(torch.argmin(distances))]


def _top_edge_string_local_points(patch: Patch, n_per_segment: int = 96) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pts = patch.sample_spline_local(n_per_segment).detach()
    xy = pts[:, :2]
    if len(xy) < 3:
        raise ValueError(f"Patch {patch.label!r} has too few outline points for string placement.")

    center = _polygon_centroid_xy(xy)
    min_x = xy[:, 0].min()
    max_x = xy[:, 0].max()
    width = max_x - min_x
    if float(width.detach().cpu()) <= 1e-5:
        raise ValueError(f"Patch {patch.label!r} is too narrow for stable string placement.")

    left_anchor = _upper_boundary_point_at_x(xy, min_x + width * 0.25)
    right_anchor = _upper_boundary_point_at_x(xy, min_x + width * 0.75)
    if float(left_anchor[0].detach().cpu()) > float(right_anchor[0].detach().cpu()):
        left_anchor, right_anchor = right_anchor, left_anchor
    return center, left_anchor, right_anchor


def add_strings_to_patches(
    patches: list[Patch],
    *,
    hanging_plane_y: float,
    id_prefix: str = "P",
    id_width: int = 2,
) -> int:
    """Attach two vertical hanging strings to each current patch."""
    total = 0
    for piece_idx, patch in enumerate(patches, start=1):
        piece_id = f"{id_prefix}{piece_idx:0{id_width}d}"
        connections: list[dict[str, Any]] = []
        center_xy, left_xy, right_xy = _top_edge_string_local_points(patch)
        for connection_id, local_xy in zip(STRING_CONNECTION_IDS, (left_xy, right_xy)):
            local_point = torch.stack([
                local_xy[0].to(device=patch.center.device, dtype=patch.center.dtype),
                local_xy[1].to(device=patch.center.device, dtype=patch.center.dtype),
                torch.zeros((), device=patch.center.device, dtype=patch.center.dtype),
            ])
            world_point = patch.local_to_world(local_point.unsqueeze(0))[0]
            board_point = world_point.detach().clone()
            board_point[1] = board_point.new_tensor(float(hanging_plane_y))
            length = abs(float(board_point[1].detach().cpu()) - float(world_point[1].detach().cpu()))
            string_id = f"S{piece_idx:0{id_width}d}_{connection_id}"
            board_connection_id = f"B{piece_idx:0{id_width}d}_{connection_id}"
            connections.append({
                "id": connection_id,
                "pieceId": piece_id,
                "placementMethod": "upper_boundary_edge",
                "pieceCenterOfMass": _xy_dict(center_xy),
                "pieceLocalPoint": _xy_dict(local_xy),
                "pieceWorldPoint": _xyz_dict(world_point),
                "boardConnectionId": board_connection_id,
                "boardPoint": _xyz_dict(board_point),
                "stringId": string_id,
                "length": length,
            })
        patch.string_connections = connections
        total += len(connections)
    return total


def _connections_by_id(patch: Patch) -> dict[str, dict[str, Any]]:
    return {
        str(connection.get("id", "")): connection
        for connection in _patch_string_connections(patch)
    }


def _require_string_connections(patches: list[Patch]) -> None:
    missing = [
        _piece_label(idx)
        for idx, patch in enumerate(patches, start=1)
        if any(connection_id not in _connections_by_id(patch) for connection_id in STRING_CONNECTION_IDS)
    ]
    if missing:
        raise ValueError(
            "String connections are missing for: "
            f"{', '.join(missing)}. Add strings before SVG export."
        )


def _board_svg_point(point: dict[str, Any], half_size: float) -> tuple[float, float]:
    x_in = (float(point["x"]) + half_size) * SCENE_UNIT_TO_INCH
    y_in = (half_size - float(point["z"])) * SCENE_UNIT_TO_INCH
    return x_in, y_in


def export_grid_svg(
    patches: list[Patch],
    *,
    hanging_plane_size: float,
    output_path: str | Path = EXPORT_GRID_SVG_PATH,
) -> Path:
    """Export the hanging board connection layout as an inches-scaled SVG."""
    _require_string_connections(patches)
    size_in = float(hanging_plane_size) * SCENE_UNIT_TO_INCH
    half_size = float(hanging_plane_size) * 0.5
    body: list[str] = [
        (
            f'<rect x="0" y="0" width="{_fmt(size_in)}" height="{_fmt(size_in)}" '
            f'fill="none" stroke="{SVG_RED}" stroke-width="{_fmt(SVG_STROKE_IN)}"/>'
        )
    ]

    for idx, patch in enumerate(patches, start=1):
        label = _piece_label(idx)
        connections = _connections_by_id(patch)
        points: dict[str, tuple[float, float]] = {}
        for connection_id in STRING_CONNECTION_IDS:
            points[connection_id] = _board_svg_point(
                connections[connection_id]["boardPoint"],
                half_size,
            )

        left = points["left"]
        right = points["right"]
        mid_x = 0.5 * (left[0] + right[0])
        mid_y = 0.5 * (left[1] + right[1])
        text_y = min(max(mid_y - 0.10, SVG_FONT_SIZE_IN), size_in - SVG_FONT_SIZE_IN * 0.4)
        body.append(f'<g id="{escape(label)}_strings">')
        body.append(
            f'<line x1="{_fmt(left[0])}" y1="{_fmt(left[1])}" '
            f'x2="{_fmt(right[0])}" y2="{_fmt(right[1])}" '
            f'stroke="{SVG_BLACK}" stroke-width="{_fmt(SVG_STROKE_IN)}" fill="none"/>'
        )
        body.append(
            f'<text x="{_fmt(mid_x)}" y="{_fmt(text_y)}" fill="{SVG_BLACK}" '
            f'font-family="monospace" font-size="{_fmt(SVG_FONT_SIZE_IN)}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )
        for connection_id, marker in (("left", "L"), ("right", "R")):
            x, y = points[connection_id]
            label_y = min(max(y + SVG_SMALL_FONT_SIZE_IN * 1.6, SVG_SMALL_FONT_SIZE_IN), size_in)
            body.append(
                f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(SVG_HOLE_RADIUS_IN)}" '
                f'fill="{SVG_BLACK}"/>'
            )
            body.append(
                f'<text x="{_fmt(x)}" y="{_fmt(label_y)}" fill="{SVG_BLACK}" '
                f'font-family="monospace" font-size="{_fmt(SVG_SMALL_FONT_SIZE_IN)}" '
                f'text-anchor="middle">{marker}</text>'
            )
        body.append("</g>")

    return _write_text(Path(output_path), _svg_document(size_in, size_in, body))


def _piece_dict_sample_outline(piece: dict[str, Any], n_per_segment: int = 64) -> torch.Tensor:
    cps = piece["controlPoints"]
    t_values = torch.linspace(0.0, 1.0, n_per_segment + 1)[:-1]
    samples: list[torch.Tensor] = []
    for idx, cp0 in enumerate(cps):
        cp1 = cps[(idx + 1) % len(cps)]
        p0 = torch.tensor([float(cp0["x"]), float(cp0["y"])], dtype=torch.float32)
        p1 = p0 + torch.tensor(
            [float(cp0["handleOut"]["x"]), float(cp0["handleOut"]["y"])],
            dtype=torch.float32,
        )
        p3 = torch.tensor([float(cp1["x"]), float(cp1["y"])], dtype=torch.float32)
        p2 = p3 + torch.tensor(
            [float(cp1["handleIn"]["x"]), float(cp1["handleIn"]["y"])],
            dtype=torch.float32,
        )
        t = t_values.unsqueeze(1)
        samples.append(
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3
        )
    return torch.cat(samples, dim=0)


def _piece_svg_point(
    point: torch.Tensor,
    min_xy: torch.Tensor,
    max_y: float,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float]:
    x_in = offset_x + (float(point[0]) - float(min_xy[0])) * SCENE_UNIT_TO_INCH
    y_in = offset_y + (max_y - float(point[1])) * SCENE_UNIT_TO_INCH
    return x_in, y_in


def _piece_dict_svg_path_data(
    piece: dict[str, Any],
    min_xy: torch.Tensor,
    max_y: float,
    offset_x: float,
    offset_y: float,
) -> str:
    commands: list[str] = []
    cps = piece["controlPoints"]
    first = torch.tensor([float(cps[0]["x"]), float(cps[0]["y"])], dtype=torch.float32)
    x0, y0 = _piece_svg_point(first, min_xy, max_y, offset_x, offset_y)
    commands.append(f"M {_fmt(x0)} {_fmt(y0)}")

    for idx, cp0 in enumerate(cps):
        cp1 = cps[(idx + 1) % len(cps)]
        p0 = torch.tensor([float(cp0["x"]), float(cp0["y"])], dtype=torch.float32)
        p1 = p0 + torch.tensor(
            [float(cp0["handleOut"]["x"]), float(cp0["handleOut"]["y"])],
            dtype=torch.float32,
        )
        p3 = torch.tensor([float(cp1["x"]), float(cp1["y"])], dtype=torch.float32)
        p2 = p3 + torch.tensor(
            [float(cp1["handleIn"]["x"]), float(cp1["handleIn"]["y"])],
            dtype=torch.float32,
        )
        x1, y1 = _piece_svg_point(p1, min_xy, max_y, offset_x, offset_y)
        x2, y2 = _piece_svg_point(p2, min_xy, max_y, offset_x, offset_y)
        x3, y3 = _piece_svg_point(p3, min_xy, max_y, offset_x, offset_y)
        commands.append(
            f"C {_fmt(x1)} {_fmt(y1)} {_fmt(x2)} {_fmt(y2)} {_fmt(x3)} {_fmt(y3)}"
        )

    commands.append("Z")
    return " ".join(commands)


def _patch_outline_svg_path_data(
    outline_xy: torch.Tensor,
    min_xy: torch.Tensor,
    max_y: float,
    offset_x: float,
    offset_y: float,
) -> str:
    if outline_xy.ndim != 2 or outline_xy.shape[1] != 2 or len(outline_xy) < 2:
        raise ValueError("Patch outline must contain at least 2 XY points.")
    commands: list[str] = []
    x0, y0 = _piece_svg_point(outline_xy[0], min_xy, max_y, offset_x, offset_y)
    commands.append(f"M {_fmt(x0)} {_fmt(y0)}")
    for point in outline_xy[1:]:
        x, y = _piece_svg_point(point, min_xy, max_y, offset_x, offset_y)
        commands.append(f"L {_fmt(x)} {_fmt(y)}")
    commands.append("Z")
    return " ".join(commands)


def _patch_component_outlines_svg_path_data(
    outlines_xy: list[torch.Tensor],
    min_xy: torch.Tensor,
    max_y: float,
    offset_x: float,
    offset_y: float,
) -> str:
    commands: list[str] = []
    for outline_xy in outlines_xy:
        if outline_xy.ndim != 2 or outline_xy.shape[1] != 2 or len(outline_xy) < 2:
            continue
        x0, y0 = _piece_svg_point(outline_xy[0], min_xy, max_y, offset_x, offset_y)
        commands.append(f"M {_fmt(x0)} {_fmt(y0)}")
        for point in outline_xy[1:]:
            x, y = _piece_svg_point(point, min_xy, max_y, offset_x, offset_y)
            commands.append(f"L {_fmt(x)} {_fmt(y)}")
        commands.append("Z")
    if not commands:
        raise ValueError("No outline components were available for SVG path export.")
    return " ".join(commands)


def _polyline_self_intersections(points: np.ndarray, eps: float = 1e-9) -> int:
    """Count strict self-intersections in a closed polyline."""
    if len(points) < 4:
        return 0

    def _cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        ab = b - a
        cd = d - c
        ac = c - a
        ad = d - a
        ca = a - c
        cb = b - c

        o1 = _cross(ab, ac)
        o2 = _cross(ab, ad)
        o3 = _cross(cd, ca)
        o4 = _cross(cd, cb)
        return ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
            (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
        )

    count = 0
    n = len(points)
    for i in range(n):
        a0 = points[i]
        a1 = points[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent segments (including wraparound) are allowed to meet.
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            b0 = points[j]
            b1 = points[(j + 1) % n]
            if _segments_intersect(a0, a1, b0, b1):
                count += 1
    return count


def _radial_envelope_outline(points: np.ndarray, n_rays: int = 1440) -> np.ndarray:
    """Build a non-self-intersecting outer envelope from a sampled outline.

    The viewport mesh face is a centroid fan. This radial envelope follows the
    furthest intersection along each ray from the same centroid, matching the
    visually filled boundary without Bezier crossover artifacts.
    """
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return points

    centroid = points.mean(axis=0)
    seg_start = points
    seg_end = np.roll(points, -1, axis=0)
    seg_delta = seg_end - seg_start

    out: list[np.ndarray] = []
    eps = 1e-9
    n_rays = max(int(n_rays), 360)

    for ray_idx in range(n_rays):
        theta = 2.0 * math.pi * ray_idx / n_rays
        ray = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        best_t = -1.0
        best_point: np.ndarray | None = None

        for start, delta in zip(seg_start, seg_delta):
            denom = float(ray[0] * delta[1] - ray[1] * delta[0])
            if abs(denom) < eps:
                continue
            offset = start - centroid
            t = float((offset[0] * delta[1] - offset[1] * delta[0]) / denom)
            u = float((offset[0] * ray[1] - offset[1] * ray[0]) / denom)
            if t >= 0.0 and -1e-6 <= u <= 1.0 + 1e-6 and t > best_t:
                best_t = t
                best_point = centroid + t * ray

        if best_point is not None:
            out.append(best_point)

    if len(out) < 3:
        return points

    dedup: list[np.ndarray] = [out[0]]
    for point in out[1:]:
        if float(np.linalg.norm(point - dedup[-1])) > 1e-6:
            dedup.append(point)
    if len(dedup) > 2 and float(np.linalg.norm(dedup[0] - dedup[-1])) <= 1e-6:
        dedup.pop()
    return np.asarray(dedup, dtype=np.float32)


def _patch_viewport_outline_components(
    patch: Patch,
    *,
    n_per_segment: int,
    raster_max_dim_px: int = 3072,
    pad_px: int = 6,
) -> list[torch.Tensor]:
    """Trace visible piece components from the same mesh the viewport draws."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Viewport-matching SVG export requires OpenCV (`cv2`).") from exc

    with torch.no_grad():
        vertices_world, faces = patch.extruded_mesh_world(n_per_segment=n_per_segment)
        center = patch.center.detach().cpu().numpy()
        rotation = patch.rotation_matrix().detach().cpu().numpy()

    verts_world = vertices_world.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy().astype(np.int32)

    # Inverse local_to_world (row-vector form): local = (world - center) @ R
    verts_local = (verts_world - center[None, :]) @ rotation
    xy = verts_local[:, :2].astype(np.float64)
    min_x, min_y = float(xy[:, 0].min()), float(xy[:, 1].min())
    max_x, max_y = float(xy[:, 0].max()), float(xy[:, 1].max())
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    max_dim = max(span_x, span_y)
    scale = max(1.0, (float(raster_max_dim_px) - 2.0 * pad_px) / max_dim)

    width_px = max(32, int(math.ceil(span_x * scale)) + 2 * pad_px + 1)
    height_px = max(32, int(math.ceil(span_y * scale)) + 2 * pad_px + 1)
    mask = np.zeros((height_px, width_px), dtype=np.uint8)

    def _xy_to_px(points_xy: np.ndarray) -> np.ndarray:
        px_x = (points_xy[:, 0] - min_x) * scale + pad_px
        px_y = (max_y - points_xy[:, 1]) * scale + pad_px
        return np.stack([px_x, px_y], axis=1)

    for tri in faces_np:
        tri_xy = xy[tri]
        # Side-wall faces collapse toward zero area in local XY; skip them so
        # raster tracing reflects the face silhouette instead of pixel bridges.
        tri_area = abs(
            (tri_xy[1, 0] - tri_xy[0, 0]) * (tri_xy[2, 1] - tri_xy[0, 1])
            - (tri_xy[1, 1] - tri_xy[0, 1]) * (tri_xy[2, 0] - tri_xy[0, 0])
        ) * 0.5
        if tri_area <= 1e-10:
            continue
        tri_px = np.round(_xy_to_px(tri_xy)).astype(np.int32)
        cv2.fillConvexPoly(mask, tri_px, 255)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [torch.from_numpy(xy.astype(np.float32))]

    contour_areas = [float(cv2.contourArea(contour)) for contour in contours]
    max_area = max(contour_areas) if contour_areas else 0.0
    min_area = max(4.0, max_area * 1e-4)

    components: list[tuple[float, torch.Tensor]] = []
    for contour, area in zip(contours, contour_areas):
        if area < min_area:
            continue
        pts_px = contour[:, 0, :].astype(np.float64)
        x_local = (pts_px[:, 0] - pad_px) / scale + min_x
        y_local = max_y - ((pts_px[:, 1] - pad_px) / scale)
        pts_local = np.stack([x_local, y_local], axis=1).astype(np.float32)
        components.append((area, torch.from_numpy(pts_local)))

    if not components:
        return [torch.from_numpy(xy.astype(np.float32))]

    # Stable ordering: largest visible component first.
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _area, component in components]


def export_pieces_svg(
    patches: list[Patch],
    *,
    output_path: str | Path = EXPORT_PIECES_SVG_PATH,
    sheet_width_in: float = 12.0,
    samples_per_segment: int = 20,
) -> Path:
    """Export all cut piece outlines and string holes as an inches-scaled SVG.

    The outline always follows the viewport-like filled shape by projecting
    the sampled spline through a centroid-fan envelope (the same visual model
    used by the viewport front-face fill).
    """
    _require_string_connections(patches)
    if samples_per_segment < 8:
        raise ValueError("samples_per_segment must be at least 8 for stable SVG export.")

    margin = 0.35
    label_gap = 0.30
    piece_infos: list[dict[str, Any]] = []
    max_piece_width = 0.0
    for idx, patch in enumerate(patches, start=1):
        label = _piece_label(idx)
        outlines_xy = _patch_viewport_outline_components(
            patch,
            n_per_segment=samples_per_segment,
        )
        if not outlines_xy:
            raise ValueError(f"Patch {label} did not produce any viewport outline components.")
        all_xy = torch.cat(outlines_xy, dim=0)
        if not torch.isfinite(all_xy).all():
            raise ValueError(
                f"Patch {label} contains non-finite viewport geometry. "
                "Try reducing optimization aggressiveness or resetting this piece."
            )
        min_xy = all_xy.min(dim=0).values
        max_xy = all_xy.max(dim=0).values
        width = max(float(max_xy[0] - min_xy[0]) * SCENE_UNIT_TO_INCH, SVG_HOLE_RADIUS_IN * 4.0)
        height = max(float(max_xy[1] - min_xy[1]) * SCENE_UNIT_TO_INCH, SVG_HOLE_RADIUS_IN * 4.0)
        max_piece_width = max(max_piece_width, width)
        piece_infos.append({
            "index": idx,
            "label": label,
            "patch": patch,
            "outlines_xy": outlines_xy,
            "min_xy": min_xy,
            "max_xy": max_xy,
            "width": width,
            "height": height,
        })

    sheet_width = max(float(sheet_width_in), max_piece_width + margin * 2.0)
    cursor_x = margin
    cursor_y = margin
    row_height = 0.0
    for info in piece_infos:
        cell_width = info["width"] + margin
        cell_height = info["height"] + label_gap + margin
        if cursor_x + cell_width > sheet_width and cursor_x > margin:
            cursor_x = margin
            cursor_y += row_height + margin
            row_height = 0.0
        info["offset_x"] = cursor_x
        info["offset_y"] = cursor_y
        cursor_x += cell_width
        row_height = max(row_height, cell_height)
    sheet_height = max(cursor_y + row_height + margin, margin * 2.0)

    body: list[str] = []
    for info in piece_infos:
        patch = info["patch"]
        label = info["label"]
        min_xy = info["min_xy"]
        max_xy = info["max_xy"]
        offset_x = float(info["offset_x"])
        offset_y = float(info["offset_y"])
        connections = _connections_by_id(patch)
        center_x = offset_x + info["width"] * 0.5
        center_y = offset_y + info["height"] * 0.5
        body.append(f'<g id="{escape(label)}">')
        body.append(
            f'<path d="{_patch_component_outlines_svg_path_data(info["outlines_xy"], min_xy, float(max_xy[1]), offset_x, offset_y)}" '
            f'fill="none" '
            f'stroke="{SVG_RED}" stroke-width="{_fmt(SVG_STROKE_IN)}"/>'
        )
        body.append(
            f'<text x="{_fmt(center_x)}" y="{_fmt(center_y)}" fill="{SVG_BLACK}" '
            f'font-family="monospace" font-size="{_fmt(SVG_FONT_SIZE_IN)}" '
            f'font-weight="bold" text-anchor="middle" dominant-baseline="middle">'
            f'{escape(label)}</text>'
        )

        for connection_id, marker in (("left", "L"), ("right", "R")):
            local_point = connections[connection_id]["pieceLocalPoint"]
            x_in = offset_x + (float(local_point["x"]) - float(min_xy[0])) * SCENE_UNIT_TO_INCH
            y_in = offset_y + (float(max_xy[1]) - float(local_point["y"])) * SCENE_UNIT_TO_INCH
            body.append(
                f'<circle cx="{_fmt(x_in)}" cy="{_fmt(y_in)}" r="{_fmt(SVG_HOLE_RADIUS_IN)}" '
                f'fill="{SVG_BLACK}"/>'
            )
            label_y = max(y_in - SVG_SMALL_FONT_SIZE_IN * 0.7, SVG_SMALL_FONT_SIZE_IN)
            body.append(
                f'<text x="{_fmt(x_in)}" y="{_fmt(label_y)}" fill="{SVG_BLACK}" '
                f'font-family="monospace" font-size="{_fmt(SVG_SMALL_FONT_SIZE_IN)}" '
                f'text-anchor="middle">{marker}</text>'
            )

        bottom_label_y = offset_y + info["height"] + SVG_SMALL_FONT_SIZE_IN * 1.5
        body.append(
            f'<text x="{_fmt(center_x)}" y="{_fmt(bottom_label_y)}" fill="{SVG_BLACK}" '
            f'font-family="monospace" font-size="{_fmt(SVG_SMALL_FONT_SIZE_IN)}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )
        body.append("</g>")

    return _write_text(Path(output_path), _svg_document(sheet_width, sheet_height, body))


def _piece_control_point_to_model(cp: dict[str, Any], device: str) -> ControlPoint:
    if "x" not in cp or "y" not in cp:
        raise ValueError("Each control point must include x and y.")

    handle_out = cp.get("handleOut") or {}
    handle_x = float(handle_out.get("x", 0.0))
    handle_y = float(handle_out.get("y", 0.0))
    handle_scale = math.hypot(handle_x, handle_y)
    handle_rotation = math.atan2(handle_y, handle_x) if handle_scale > 1e-8 else 0.0

    return ControlPoint(
        x=float(cp["x"]),
        y=float(cp["y"]),
        z=float(cp.get("z", 0.0)),
        handle_scale=max(handle_scale, 1e-6),
        handle_rotation=handle_rotation,
        device=device,
    )


def piece_dict_to_patch(piece: dict[str, Any], *, device: str = "cpu") -> Patch:
    """Build a Patch from one exported piece JSON object."""
    position = piece.get("position")
    if not isinstance(position, dict):
        raise ValueError("Each piece must include a position object.")

    control_points = piece.get("controlPoints")
    if not isinstance(control_points, list) or len(control_points) != Patch.N_CONTROL_POINTS:
        raise ValueError(
            f"Each piece must include exactly {Patch.N_CONTROL_POINTS} controlPoints."
        )

    color = piece.get("color", "#ffffff")
    if isinstance(color, str):
        albedo = _hex_to_rgb(color)
    elif isinstance(color, list):
        albedo = [float(channel) for channel in color[:3]]
    else:
        albedo = [1.0, 1.0, 1.0]

    patch = Patch(
        control_points=[
            _piece_control_point_to_model(cp, device)
            for cp in control_points
        ],
        center=[
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
        ],
        theta=float(piece.get("theta", 0.0)),
        albedo=albedo,
        device=device,
        label=str(piece.get("id", "")),
    )
    string_connections = piece.get("stringConnections")
    if isinstance(string_connections, list):
        patch.string_connections = [
            {
                "id": str(connection.get("id", "")),
                "pieceId": str(piece.get("id", "")),
                "placementMethod": str(connection.get("placementMethod", "")),
                "pieceCenterOfMass": connection.get("pieceCenterOfMass", {}),
                "pieceLocalPoint": connection.get("pieceLocalPoint", {}),
                "pieceWorldPoint": connection.get("pieceWorldPoint", {}),
                "boardConnectionId": str(connection.get("boardConnectionId", "")),
                "boardPoint": connection.get("boardPoint", {}),
                "stringId": str(connection.get("stringId", "")),
                "length": float(connection.get("length", 0.0)),
            }
            for connection in string_connections
        ]
    return patch


def patches_from_export_payload(payload: dict[str, Any], *, device: str = "cpu") -> list[Patch]:
    """Build patches from a pieces.json-style export payload."""
    pieces = payload.get("pieces")
    if not isinstance(pieces, list):
        raise ValueError("Import JSON must contain a pieces list.")
    if not pieces:
        raise ValueError("Import JSON does not contain any pieces.")
    return [
        piece_dict_to_patch(piece, device=device)
        for piece in pieces
    ]


def import_patches_from_json(path: str | Path, *, device: str = "cpu") -> list[Patch]:
    """Load patches from a previous pieces JSON export."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Import JSON root must be an object.")
    return patches_from_export_payload(payload, device=device)


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
    hanging_plane_size: float | None = None,
    piece_scale: float = 1.0,
    id_prefix: str = "P",
    id_width: int = 2,
    indent: int = 2,
) -> Path:
    payload = build_export_payload(
        patches,
        scale=scale,
        hanging_plane_size=hanging_plane_size,
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


def send_export_json_file(
    endpoint: str = "http://localhost:5173/api/import",
    *,
    timeout_s: float = 15.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST exports/pieces.json as application/json (curl --data-binary equivalent)."""
    if not EXPORT_JSON_PATH.exists():
        return {
            "ok": False,
            "status": None,
            "body": f"Missing export file: {EXPORT_JSON_PATH}",
        }

    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    body_bytes = EXPORT_JSON_PATH.read_bytes()

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
