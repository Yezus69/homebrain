from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from homebrain.data.spatial_dataset import write_json
from homebrain.viz.checkpoint_3d_scene_types_v0 import (
    AccumulatedMapStateV0,
    SceneTimelineFrameV0,
    deterministic_scene_json,
    to_jsonable,
)
from homebrain.viz.checkpoint_3d_viewer_html_v0 import checkpoint_3d_viewer_html


def write_scene_timeline_jsonl(path: str | Path, frames: list[SceneTimelineFrameV0]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(deterministic_scene_json(frame))
            handle.write("\n")
    return target


def write_accumulated_scene_json(
    path: str | Path,
    *,
    final_map: AccumulatedMapStateV0,
    timeline_maps: list[AccumulatedMapStateV0],
    summary: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": "homebrain.checkpoint_3d_accumulated_scene.v0",
        "summary": summary,
        "final_map": final_map.to_dict(),
        "timeline_maps": [state.to_dict() for state in timeline_maps],
    }
    write_json(path, payload, pretty=True)
    return Path(path)


def write_ply_point_cloud(
    path: str | Path,
    scene_or_points: AccumulatedMapStateV0 | dict[str, Any] | list[Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    vertices = _vertices_from_scene(scene_or_points)
    scene_metadata = to_jsonable(scene_or_points)
    if not isinstance(scene_metadata, dict):
        scene_metadata = {}
    if metadata is not None:
        scene_metadata = {**scene_metadata, **metadata}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        "comment homebrain checkpoint 3d visualizer v0",
        *_ply_comments(scene_metadata),
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for x_m, y_m, z_m, red, green, blue in vertices:
        lines.append(f"{x_m:.6f} {y_m:.6f} {z_m:.6f} {red:d} {green:d} {blue:d}")
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return target


def write_ply_lines(path: str | Path, lines_3d: list[list[list[float]]], *, metadata: dict[str, Any] | None = None) -> Path:
    vertices: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    for segment in lines_3d:
        if len(segment) != 2:
            continue
        start = len(vertices)
        vertices.append([float(segment[0][0]), float(segment[0][1]), float(segment[0][2])])
        vertices.append([float(segment[1][0]), float(segment[1][1]), float(segment[1][2])])
        edges.append((start, start + 1))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = [
        "ply",
        "format ascii 1.0",
        "comment homebrain checkpoint 3d visualizer v0 lines",
        *_ply_comments(metadata or {}),
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element edge {len(edges)}",
        "property int vertex1",
        "property int vertex2",
        "end_header",
    ]
    for point in vertices:
        out.append(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
    for start, end in edges:
        out.append(f"{start:d} {end:d}")
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(out) + "\n")
    return target


def write_viewer_html(
    path: str | Path,
    *,
    frames: list[SceneTimelineFrameV0],
    accumulated_scene: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    scene_json = deterministic_scene_json(
        {
            "summary": summary,
            "frames": [frame.to_dict() for frame in frames],
            "accumulated_scene": accumulated_scene,
        }
    ).replace("</", "<\\/")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(checkpoint_3d_viewer_html(scene_json))
    return target


def write_artifact_manifest(path: str | Path, *, root: str | Path, artifacts: list[str], summary: dict[str, Any]) -> Path:
    root_path = Path(root)
    records = []
    for rel in sorted(artifacts):
        artifact_path = root_path / rel
        if artifact_path.exists():
            records.append({"path": rel, "sha256": file_sha256(artifact_path), "bytes": int(artifact_path.stat().st_size)})
    write_json(
        path,
        {
            "schema_version": "homebrain.checkpoint_3d_visualizer_manifest.v0",
            "artifact_count": len(records),
            "artifacts": records,
            "geometry_source": summary.get("geometry_source"),
            "pose_source": summary.get("pose_source"),
            "pose_source_is_oracle": summary.get("pose_source_is_oracle"),
            "dense_3d_claimed": summary.get("dense_3d_claimed"),
            "safety": summary.get("safety"),
            "summary_sha256": hashlib.sha256(deterministic_scene_json(summary).encode("ascii")).hexdigest(),
        },
        pretty=True,
    )
    return Path(path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_lines(points: list[list[float]]) -> list[list[list[float]]]:
    return [[points[index], points[index + 1]] for index in range(max(0, len(points) - 1))]


def camera_frustum_lines_from_frames(frames: list[SceneTimelineFrameV0]) -> list[list[list[float]]]:
    lines: list[list[list[float]]] = []
    for frame in frames:
        if frame.camera_frustum is not None:
            lines.extend(frame.camera_frustum.frustum_lines_map)
    return lines


def _vertices_from_scene(scene_or_points: AccumulatedMapStateV0 | dict[str, Any] | list[Any]) -> list[tuple[float, float, float, int, int, int]]:
    data = to_jsonable(scene_or_points)
    if isinstance(data, list):
        return [_point_vertex(point) for point in data if isinstance(point, list) and len(point) >= 3]
    if not isinstance(data, dict):
        return []
    vertices: list[tuple[float, float, float, int, int, int]] = []
    for point in data.get("point_cloud", []):
        if isinstance(point, list) and len(point) >= 3:
            vertices.append(_point_vertex(point))
    for key, color in (
        ("free_cells", (80, 170, 95)),
        ("obstacle_cells", (190, 65, 55)),
        ("unknown_cells", (145, 145, 145)),
        ("risky_cells", (235, 170, 45)),
        ("hazard_cells", (210, 65, 170)),
        ("stale_cells", (80, 130, 210)),
    ):
        for cell in data.get(key, []):
            if not isinstance(cell, dict):
                continue
            vertices.append((float(cell["x_m"]), float(cell["y_m"]), float(cell.get("z_m", 0.0)), *color))
    return sorted(vertices, key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))


def _point_vertex(point: list[Any]) -> tuple[float, float, float, int, int, int]:
    if len(point) >= 6:
        return (float(point[0]), float(point[1]), float(point[2]), int(point[3]), int(point[4]), int(point[5]))
    return (float(point[0]), float(point[1]), float(point[2]), 105, 115, 125)


def _ply_comments(metadata: dict[str, Any]) -> list[str]:
    comments = []
    for key in ("geometry_source", "pose_source", "pose_source_is_oracle", "dense_3d_claimed"):
        if key in metadata:
            comments.append(f"comment {key}={metadata[key]}")
    safety = metadata.get("safety")
    if isinstance(safety, dict):
        for key in ("replay_only", "not_executed", "control_safe", "raw_pwm_emitted", "hardware_validated"):
            if key in safety:
                comments.append(f"comment safety.{key}={safety[key]}")
    return comments
