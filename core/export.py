"""JSON export helpers for optimized patch pieces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib import error, request
from xml.sax.saxutils import escape

import torch

from core.patch import ControlPoint, Patch

EXPORT_JSON_PATH = Path("exports/pieces.json")
EXPORT_GRID_SVG_PATH = Path("exports/grid.svg")
EXPORT_PIECES_SVG_PATH = Path("exports/pieces.svg")
EXPORT_PIECES_PNG_PATH = Path("exports/pieces.png")
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


def _balanced_string_local_points(patch: Patch, n_per_segment: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pts = patch.sample_spline_local(n_per_segment).detach()
    xy = pts[:, :2]
    if len(xy) < 3:
        raise ValueError(f"Patch {patch.label!r} has too few outline points for string placement.")

    center = _polygon_centroid_xy(xy)
    leftmost = xy[int(torch.argmin(xy[:, 0]))]
    rightmost = xy[int(torch.argmax(xy[:, 0]))]
    if torch.linalg.norm(leftmost - rightmost) <= 1e-5:
        raise ValueError(f"Patch {patch.label!r} is too narrow for stable string placement.")

    left_anchor = 0.5 * (center + leftmost)
    right_anchor = 0.5 * (center + rightmost)
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
        center_xy, left_xy, right_xy = _balanced_string_local_points(patch)
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
                "placementMethod": "center_of_mass_to_extreme_midpoint",
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


def export_pieces_svg(
    patches: list[Patch],
    *,
    output_path: str | Path = EXPORT_PIECES_SVG_PATH,
    sheet_width_in: float = 12.0,
) -> Path:
    """Export all cut piece outlines and string holes as an inches-scaled SVG."""
    _require_string_connections(patches)
    margin = 0.35
    label_gap = 0.30
    piece_infos: list[dict[str, Any]] = []
    max_piece_width = 0.0
    for idx, patch in enumerate(patches, start=1):
        label = _piece_label(idx)
        piece = patch_to_piece_dict(patch, label)
        xy = _piece_dict_sample_outline(piece, n_per_segment=64)
        min_xy = xy.min(dim=0).values
        max_xy = xy.max(dim=0).values
        width = max(float(max_xy[0] - min_xy[0]) * SCENE_UNIT_TO_INCH, SVG_HOLE_RADIUS_IN * 4.0)
        height = max(float(max_xy[1] - min_xy[1]) * SCENE_UNIT_TO_INCH, SVG_HOLE_RADIUS_IN * 4.0)
        max_piece_width = max(max_piece_width, width)
        piece_infos.append({
            "index": idx,
            "label": label,
            "patch": patch,
            "piece": piece,
            "xy": xy,
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
        piece = info["piece"]
        label = info["label"]
        min_xy = info["min_xy"]
        max_xy = info["max_xy"]
        offset_x = float(info["offset_x"])
        offset_y = float(info["offset_y"])
        connections = {
            str(connection["id"]): connection
            for connection in piece.get("stringConnections", [])
        }
        center_x = offset_x + info["width"] * 0.5
        center_y = offset_y + info["height"] * 0.5
        body.append(f'<g id="{escape(label)}">')
        body.append(
            f'<path d="{_piece_dict_svg_path_data(piece, min_xy, float(max_xy[1]), offset_x, offset_y)}" '
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


def _patch_fill_rgba(patch: Patch, alpha: int = 255) -> tuple[int, int, int, int]:
    rgb = patch.albedo.detach().cpu().clamp(0.0, 1.0).numpy().tolist()
    return (
        max(0, min(255, int(round(rgb[0] * 255.0)))),
        max(0, min(255, int(round(rgb[1] * 255.0)))),
        max(0, min(255, int(round(rgb[2] * 255.0)))),
        alpha,
    )


def _orthographic_view_basis(
    azimuth_deg: float | None = None,
    elevation_deg: float | None = None,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np

    if azimuth_deg is None or elevation_deg is None:
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye_dir = np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ], dtype=np.float32)
    forward = -eye_dir / max(float(np.linalg.norm(eye_dir)), 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, forward)
    return right, up, forward


def _flat_patch_mesh_for_png(
    patch: Patch,
    n_per_segment: int,
    thickness: float = Patch.DEFAULT_THICKNESS,
) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    local = patch.sample_spline_local(n_per_segment).detach().cpu().numpy()
    count = len(local)
    half = thickness * 0.5
    front = local.copy()
    back = local.copy()
    front[:, 2] = half
    back[:, 2] = -half
    front_centroid = front.mean(axis=0, keepdims=True)
    back_centroid = back.mean(axis=0, keepdims=True)
    verts = np.concatenate([front_centroid, front, back_centroid, back], axis=0).astype(np.float32)

    back_center = count + 1
    back_start = count + 2
    faces: list[list[int]] = []
    for idx in range(count):
        nxt = (idx + 1) % count
        fi = idx + 1
        fj = nxt + 1
        bi = back_start + idx
        bj = back_start + nxt
        faces.append([0, fj, fi])
        faces.append([back_center, bi, bj])
        faces.append([fi, bi, bj])
        faces.append([fi, bj, fj])
    return verts, np.array(faces, dtype=np.int32)


def _project_patch_for_png(
    patch: Patch,
    *,
    right: "np.ndarray",
    up: "np.ndarray",
    forward: "np.ndarray",
    n_per_segment: int,
) -> dict[str, Any]:
    import numpy as np

    verts, faces = _flat_patch_mesh_for_png(patch, n_per_segment)
    projected = np.stack([
        verts @ right,
        verts @ up,
        verts @ forward,
    ], axis=1)
    min_xy = projected[:, :2].min(axis=0)
    max_xy = projected[:, :2].max(axis=0)
    return {
        "patch": patch,
        "verts": verts,
        "projected": projected,
        "faces": faces,
        "min_xy": min_xy,
        "max_xy": max_xy,
        "width": max(float(max_xy[0] - min_xy[0]), 1e-4),
        "height": max(float(max_xy[1] - min_xy[1]), 1e-4),
    }


def _shade_face_rgba(
    base: tuple[int, int, int, int],
    verts: "np.ndarray",
    face: "np.ndarray",
) -> tuple[int, int, int, int]:
    import numpy as np

    tri = verts[face]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    length = float(np.linalg.norm(normal))
    if length > 1e-8:
        normal = normal / length
    light = np.array([0.5, 1.0, 0.8], dtype=np.float32)
    light = light / np.linalg.norm(light)
    diff = max(float(np.dot(normal, light)), 0.0) * 0.7 + 0.3
    return (
        max(0, min(255, int(round(base[0] * diff)))),
        max(0, min(255, int(round(base[1] * diff)))),
        max(0, min(255, int(round(base[2] * diff)))),
        base[3],
    )


def _rasterize_triangle(
    pixels: "np.ndarray",
    z_buffer: "np.ndarray",
    xy: "np.ndarray",
    z: "np.ndarray",
    color: tuple[int, int, int, int],
) -> None:
    import numpy as np

    height, width = z_buffer.shape
    min_x = max(0, int(math.floor(float(xy[:, 0].min()))))
    max_x = min(width - 1, int(math.ceil(float(xy[:, 0].max()))))
    min_y = max(0, int(math.floor(float(xy[:, 1].min()))))
    max_y = min(height - 1, int(math.ceil(float(xy[:, 1].max()))))
    if min_x > max_x or min_y > max_y:
        return

    x0, y0 = xy[0]
    x1, y1 = xy[1]
    x2, y2 = xy[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) <= 1e-8:
        return

    ys, xs = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
    px = xs + 0.5
    py = ys + 0.5
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    if not bool(inside.any()):
        return

    depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
    target = z_buffer[min_y:max_y + 1, min_x:max_x + 1]
    update = inside & (depth < target)
    if not bool(update.any()):
        return

    target[update] = depth[update]
    pixels[min_y:max_y + 1, min_x:max_x + 1][update] = color


def export_pieces_png(
    patches: list[Patch],
    *,
    output_path: str | Path = EXPORT_PIECES_PNG_PATH,
    pixels_per_scene_unit: float = 240.0,
    margin_scene_units: float = 0.2,
    gap_scene_units: float = 0.25,
    n_per_segment: int = 32,
    supersample: int = 3,
) -> Path:
    """Render all pieces in one straight orthographic row with theta flattened."""
    if not patches:
        raise ValueError("No patches to export.")

    import numpy as np
    from PIL import Image

    right, up, forward = _orthographic_view_basis()
    infos: list[dict[str, Any]] = []
    total_width_units = margin_scene_units * 2.0
    max_height_units = 0.0
    for patch in patches:
        info = _project_patch_for_png(
            patch,
            right=right,
            up=up,
            forward=forward,
            n_per_segment=n_per_segment,
        )
        infos.append(info)
        total_width_units += float(info["width"])
        max_height_units = max(max_height_units, float(info["height"]))

    total_width_units += gap_scene_units * max(0, len(infos) - 1)
    total_height_units = max_height_units + margin_scene_units * 2.0

    ss = max(1, int(supersample))
    scale_px = float(pixels_per_scene_unit) * ss
    width_px = max(1, int(math.ceil(total_width_units * scale_px)))
    height_px = max(1, int(math.ceil(total_height_units * scale_px)))

    pixels = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    z_buffer = np.full((height_px, width_px), np.inf, dtype=np.float32)
    cursor_x = margin_scene_units * scale_px
    top_y = margin_scene_units * scale_px

    for info in infos:
        vertical_offset = top_y + (max_height_units - float(info["height"])) * 0.5 * scale_px
        projected = info["projected"].copy()
        projected[:, 0] = cursor_x + (projected[:, 0] - float(info["min_xy"][0])) * scale_px
        projected[:, 1] = vertical_offset + (float(info["max_xy"][1]) - projected[:, 1]) * scale_px
        base_color = _patch_fill_rgba(info["patch"])
        face_order = sorted(
            info["faces"],
            key=lambda face: float(projected[face, 2].mean()),
            reverse=True,
        )
        for face in face_order:
            color = _shade_face_rgba(base_color, info["verts"], face)
            _rasterize_triangle(
                pixels,
                z_buffer,
                projected[face, :2],
                projected[face, 2],
                color,
            )
        cursor_x += (float(info["width"]) + gap_scene_units) * scale_px

    image = Image.fromarray(pixels, mode="RGBA")
    if ss > 1:
        image = image.resize((math.ceil(width_px / ss), math.ceil(height_px / ss)), Image.Resampling.LANCZOS)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


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
