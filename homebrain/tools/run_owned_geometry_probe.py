from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.ingest.image_sequence import ImageSequenceIngestSummary, ingest_image_sequence
from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.teachers.artifacts import load_array
from homebrain.teachers.moge_scene_teacher import (
    MOGE_DEFAULT_MODEL_ID,
    MoGeSceneTeacherUnavailableError,
    create_moge_scene_teacher,
)
from homebrain.teachers.qa_scene_teacher import run_scene_teacher_qa
from homebrain.teachers.scene_teacher import (
    SCENE_TEACHER_MANIFEST_FILE,
    SceneTeacherRunConfig,
    SceneTeacherRunSummary,
    load_scene_teacher_manifest,
)
from homebrain.teachers.vggt_scene_teacher import (
    VGGT_DEFAULT_MODEL_ID,
    VGGTSceneTeacherUnavailableError,
    create_vggt_scene_teacher,
)
from homebrain.tools.audit_scene_teacher_signal import audit_scene_teacher_signal

PROBE_SCHEMA_VERSION = "homebrain.owned_geometry_probe_result.v0"
VISUAL_REVIEW_SCHEMA_VERSION = "homebrain.owned_geometry_probe_visual_review.v0"
STATUS_REVIEW_ONLY_NOT_TRAINABLE = "REVIEW_ONLY_NOT_TRAINABLE"
STATUS_BLOCKED_MISSING_TEACHER_SETUP = "BLOCKED_MISSING_TEACHER_SETUP"


def run_owned_geometry_probe(
    *,
    frames_dir: str | Path,
    out_dir: str | Path,
    camera_name: str,
    fps: float,
    teacher_name: str,
    backend_name: str,
    owned_or_license_approved: bool,
    model_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    retry_command: str | None = None,
) -> JsonDict:
    if teacher_name not in {"moge", "vggt"}:
        raise ValueError(f"unsupported teacher {teacher_name!r}; expected 'moge' or 'vggt'")
    if backend_name not in {"real", "fake"}:
        raise ValueError(f"unsupported backend {backend_name!r}; expected 'real' or 'fake'")
    if not owned_or_license_approved:
        raise ValueError("--owned-or-license-approved is required for the owned geometry probe")

    resolved_retry_command = retry_command or _probe_retry_command_from_values(
        frames_dir=frames_dir,
        out_dir=out_dir,
        camera_name=camera_name,
        fps=fps,
        teacher_name=teacher_name,
        backend_name=backend_name,
        model_dir=model_dir,
        checkpoint=checkpoint,
        device=device,
        max_frames=max_frames,
        stride=stride,
    )
    started = time.perf_counter()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_dir = output / "route"
    teacher_artifacts_dir = route_dir / "teacher_artifacts" / f"{teacher_name}_scene_v0_{backend_name}"
    qa_path = teacher_artifacts_dir.parent / f"{teacher_artifacts_dir.name}_qa.json"
    audit_json_path = teacher_artifacts_dir.parent / f"{teacher_artifacts_dir.name}_signal_audit.json"
    audit_md_path = teacher_artifacts_dir.parent / f"{teacher_artifacts_dir.name}_signal_audit.md"
    visual_review_dir = output / "visual_review"
    result_json_path = output / "result.json"
    result_md_path = output / "result.md"

    result: JsonDict = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": None,
        "active_path": "image_sequence -> scene_teacher -> scene_teacher_qa -> signal_audit -> visual_review",
        "inputs": {
            "frames": Path(frames_dir).as_posix(),
            "out": output.as_posix(),
            "camera": camera_name,
            "fps": fps,
            "teacher": teacher_name,
            "backend": backend_name,
            "owned_or_license_approved": owned_or_license_approved,
            "model_dir": Path(model_dir).as_posix() if model_dir is not None else None,
            "checkpoint": Path(checkpoint).as_posix() if checkpoint is not None else None,
            "device": device,
            "max_frames": max_frames,
            "stride": stride,
        },
        "paths": {
            "route_dir": route_dir.as_posix(),
            "teacher_artifacts_dir": teacher_artifacts_dir.as_posix(),
            "scene_teacher_manifest": (teacher_artifacts_dir / SCENE_TEACHER_MANIFEST_FILE).as_posix(),
            "qa_path": qa_path.as_posix(),
            "audit_json_path": audit_json_path.as_posix(),
            "audit_md_path": audit_md_path.as_posix(),
            "visual_review_dir": visual_review_dir.as_posix(),
            "result_json": result_json_path.as_posix(),
            "result_md": result_md_path.as_posix(),
        },
        "steps": {
            "image_sequence_ingest": {"attempted": False, "ok": False},
            "scene_teacher_run": {"attempted": False, "ok": False},
            "scene_teacher_qa": {"attempted": False, "ok": False, "skipped_reason": "teacher_artifacts_missing"},
            "signal_audit": {"attempted": False, "ok": False, "skipped_reason": "qa_missing"},
            "visual_review": {"attempted": False, "ok": False, "skipped_reason": "teacher_artifacts_missing"},
        },
        "teacher_artifacts_exist": False,
        "qa_exists": False,
        "audit_exists": False,
        "visual_review_exists": False,
        "fake_fallback_used": False,
        "missing_setup_fields": [],
        "retry_command": resolved_retry_command,
        "runtime_sec": None,
    }

    try:
        ingest_summary = ingest_image_sequence(
            frames_dir=frames_dir,
            out_dir=route_dir,
            camera_name=camera_name,
            fps=fps,
            max_frames=max_frames,
            stride=stride,
            owned_or_license_approved=owned_or_license_approved,
        )
    except Exception as exc:  # noqa: BLE001 - result.json should preserve the failure.
        result["status"] = "FAILED_IMAGE_SEQUENCE_INGEST"
        result["steps"]["image_sequence_ingest"] = {
            "attempted": True,
            "ok": False,
            "error": str(exc),
        }
        return _finish_probe_result(result, result_json_path, result_md_path, started)

    result["steps"]["image_sequence_ingest"] = _ingest_step(ingest_summary)

    try:
        teacher_summary = _run_scene_teacher(
            teacher_name=teacher_name,
            backend_name=backend_name,
            log_dir=route_dir,
            out_dir=teacher_artifacts_dir,
            model_dir=model_dir,
            checkpoint=checkpoint,
            device=device,
            max_frames=max_frames,
            stride=stride,
        )
    except (MoGeSceneTeacherUnavailableError, VGGTSceneTeacherUnavailableError) as exc:
        result["steps"]["scene_teacher_run"] = {
            "attempted": True,
            "ok": False,
            "error": str(exc),
            "teacher": teacher_name,
            "backend": backend_name,
        }
        if backend_name == "real":
            result["status"] = STATUS_BLOCKED_MISSING_TEACHER_SETUP
            result["missing_setup_fields"] = _missing_teacher_setup_fields(
                teacher_name=teacher_name,
                model_dir=model_dir,
                checkpoint=checkpoint,
            )
            result["teacher_artifacts_exist"] = False
            return _finish_probe_result(result, result_json_path, result_md_path, started)
        result["status"] = "FAILED_FAKE_SCENE_TEACHER"
        return _finish_probe_result(result, result_json_path, result_md_path, started)
    except Exception as exc:  # noqa: BLE001 - unexpected teacher errors should be explicit.
        result["status"] = "FAILED_SCENE_TEACHER"
        result["steps"]["scene_teacher_run"] = {
            "attempted": True,
            "ok": False,
            "error": str(exc),
            "teacher": teacher_name,
            "backend": backend_name,
        }
        return _finish_probe_result(result, result_json_path, result_md_path, started)

    result["steps"]["scene_teacher_run"] = {
        "attempted": True,
        "ok": True,
        "teacher": teacher_summary.teacher_name,
        "backend": teacher_summary.backend,
        "mock": teacher_summary.mock,
        "frame_count": teacher_summary.frame_count,
        "manifest_path": teacher_summary.manifest_path.as_posix(),
    }

    scene_manifest_path = teacher_artifacts_dir / SCENE_TEACHER_MANIFEST_FILE
    teacher_artifacts_exist = scene_manifest_path.exists()
    result["teacher_artifacts_exist"] = teacher_artifacts_exist

    qa_exists = False
    if teacher_artifacts_exist:
        try:
            qa = run_scene_teacher_qa(artifacts_dir=teacher_artifacts_dir, out_path=qa_path)
            qa_exists = qa_path.exists()
            result["steps"]["scene_teacher_qa"] = {
                "attempted": True,
                "ok": True,
                "qa_path": qa_path.as_posix(),
                "frame_count": qa.metrics.get("frame_count"),
                "missing_artifact_count": qa.metrics.get("missing_artifact_count"),
                "artifact_shape_error_count": qa.metrics.get("artifact_shape_error_count"),
                "promotable_to_spatial_pack": qa.metrics.get("promotable_to_spatial_pack"),
            }
        except Exception as exc:  # noqa: BLE001 - preserve QA failure and skip audit.
            result["steps"]["scene_teacher_qa"] = {
                "attempted": True,
                "ok": False,
                "qa_path": qa_path.as_posix(),
                "error": str(exc),
            }
    result["qa_exists"] = qa_exists

    audit_exists = False
    audit_next_allowed_use: str | None = None
    if qa_exists:
        try:
            audit = audit_scene_teacher_signal(
                log_dir=route_dir,
                scene_teacher_dir=teacher_artifacts_dir,
                qa_path=qa_path,
                out_json=audit_json_path,
                out_md=audit_md_path,
                command=_audit_command(
                    log_dir=route_dir,
                    scene_teacher_dir=teacher_artifacts_dir,
                    qa_path=qa_path,
                    out_json=audit_json_path,
                    out_md=audit_md_path,
                ),
            )
            audit_exists = audit_json_path.exists() and audit_md_path.exists()
            audit_next_allowed_use = str(audit.get("next_allowed_use"))
            result["steps"]["signal_audit"] = {
                "attempted": True,
                "ok": True,
                "audit_json_path": audit_json_path.as_posix(),
                "audit_md_path": audit_md_path.as_posix(),
                "next_allowed_use": audit_next_allowed_use,
                "hard_blockers": audit.get("hard_blockers", []),
                "single_frame_geometry_pretrain_candidate": audit.get("single_frame_geometry_pretrain_candidate"),
                "temporal_memory_pretrain_candidate": audit.get("temporal_memory_pretrain_candidate"),
            }
        except Exception as exc:  # noqa: BLE001
            result["steps"]["signal_audit"] = {
                "attempted": True,
                "ok": False,
                "qa_path": qa_path.as_posix(),
                "error": str(exc),
            }
    result["audit_exists"] = audit_exists

    if teacher_artifacts_exist:
        try:
            visual_manifest = _write_visual_review_artifact(
                scene_teacher_dir=teacher_artifacts_dir,
                out_dir=visual_review_dir,
                log_dir=route_dir,
            )
            result["visual_review_exists"] = visual_manifest.exists()
            result["steps"]["visual_review"] = {
                "attempted": True,
                "ok": True,
                "visual_review_manifest": visual_manifest.as_posix(),
            }
        except Exception as exc:  # noqa: BLE001 - final result still answers the probe.
            result["steps"]["visual_review"] = {
                "attempted": True,
                "ok": False,
                "error": str(exc),
            }

    result["status"] = _final_status(backend_name=backend_name, audit_next_allowed_use=audit_next_allowed_use)
    if backend_name == "fake":
        result["status"] = STATUS_REVIEW_ONLY_NOT_TRAINABLE
    return _finish_probe_result(result, result_json_path, result_md_path, started)


def write_visual_review_artifact(*, scene_teacher_dir: str | Path, out_dir: str | Path, max_frames: int = 6) -> Path:
    return _write_visual_review_artifact(
        scene_teacher_dir=scene_teacher_dir,
        out_dir=out_dir,
        log_dir=None,
        max_frames=max_frames,
    )


def _write_visual_review_artifact(
    *,
    scene_teacher_dir: str | Path,
    out_dir: str | Path,
    log_dir: str | Path | None,
    max_frames: int = 6,
) -> Path:
    scene_root = Path(scene_teacher_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_root = Path(log_dir) if log_dir is not None else None
    manifest = load_scene_teacher_manifest(scene_root)
    frame_records = [frame for frame in manifest.get("frames", []) if isinstance(frame, dict)]
    selected = frame_records[:max_frames]
    if not selected:
        raise ValueError("cannot write visual review artifact with zero scene-teacher frames")

    rows: list[np.ndarray] = []
    frame_summaries: list[JsonDict] = []
    for frame in selected:
        artifacts = frame.get("artifacts") if isinstance(frame.get("artifacts"), dict) else {}
        panels = _review_panels(scene_root, artifacts, frame=frame, log_root=log_root)
        rows.append(_join_with_gap(panels, gap=2, axis=1))
        frame_summaries.append(
            {
                "frame_id": frame.get("frame_id"),
                "camera_id": frame.get("camera_id"),
                "source_data_ref": frame.get("source_data_ref"),
                "panels": [
                    "source_rgb",
                    "depth",
                    "confidence",
                    "floor_traversable_mask",
                    "obstacle_risk_mask",
                ],
            }
        )

    contact_sheet = _join_with_gap(rows, gap=2, axis=0)
    contact_sheet_path = output / "scene_teacher_review.ppm"
    _write_ppm(contact_sheet_path, contact_sheet)
    visual_manifest: JsonDict = {
        "schema_version": VISUAL_REVIEW_SCHEMA_VERSION,
        "source_scene_teacher": scene_root.as_posix(),
        "contact_sheet": contact_sheet_path.relative_to(output).as_posix(),
        "frame_count": len(selected),
        "teacher_name": manifest.get("teacher_name"),
        "backend": manifest.get("backend"),
        "mock": bool(manifest.get("mock", False)),
        "synthetic": bool(manifest.get("synthetic", False)),
        "real_perception": bool(manifest.get("real_perception", False)),
        "review_only": True,
        "not_trainable": bool(manifest.get("mock", False) or manifest.get("synthetic", False)),
        "frames": frame_summaries,
    }
    visual_manifest_path = output / "visual_review_manifest.json"
    _write_json(visual_manifest_path, visual_manifest)
    return visual_manifest_path


def _run_scene_teacher(
    *,
    teacher_name: str,
    backend_name: str,
    log_dir: Path,
    out_dir: Path,
    model_dir: str | Path | None,
    checkpoint: str | Path | None,
    device: str | None,
    max_frames: int | None,
    stride: int,
) -> SceneTeacherRunSummary:
    if teacher_name == "moge":
        teacher = create_moge_scene_teacher(
            backend_name=backend_name,
            model_id=MOGE_DEFAULT_MODEL_ID,
            model_dir=model_dir,
            checkpoint=checkpoint,
            device=device,
            max_frames=max_frames,
            stride=stride,
        )
    elif teacher_name == "vggt":
        teacher = create_vggt_scene_teacher(
            backend_name=backend_name,
            model_id=VGGT_DEFAULT_MODEL_ID,
            model_dir=model_dir,
            checkpoint=checkpoint,
            device=device,
            max_frames=max_frames,
            stride=stride,
        )
    else:
        raise ValueError(f"unsupported teacher {teacher_name!r}")
    return teacher.run(SceneTeacherRunConfig(log_dir=log_dir, out_dir=out_dir))


def _ingest_step(summary: ImageSequenceIngestSummary) -> JsonDict:
    return {
        "attempted": True,
        "ok": True,
        "route_dir": summary.out_dir.as_posix(),
        "frame_count": summary.frame_count,
        "image_load_error_count": summary.image_load_error_count,
        "metadata_path": summary.metadata_path.as_posix(),
    }


def _final_status(*, backend_name: str, audit_next_allowed_use: str | None) -> str:
    if audit_next_allowed_use == "temporal_memory_pretrain_candidate":
        return "TEMPORAL_MEMORY_PRETRAIN_CANDIDATE"
    if audit_next_allowed_use == "single_frame_geometry_pretrain_candidate":
        return "SINGLE_FRAME_GEOMETRY_PRETRAIN_CANDIDATE"
    if audit_next_allowed_use == "blocked":
        return "BLOCKED_SIGNAL_AUDIT"
    if backend_name == "real" and audit_next_allowed_use == "review_only":
        return "REAL_REVIEW_ONLY_NOT_TRAINABLE"
    return "BLOCKED_INCOMPLETE_PROBE"


def _missing_teacher_setup_fields(
    *,
    teacher_name: str,
    model_dir: str | Path | None,
    checkpoint: str | Path | None,
) -> list[JsonDict]:
    prefix = teacher_name.upper()
    adapter_env = f"HOMEBRAIN_{prefix}_ADAPTER"
    dir_env = f"HOMEBRAIN_{prefix}_DIR"
    checkpoint_env = f"HOMEBRAIN_{prefix}_CHECKPOINT"
    adapter = os.environ.get(adapter_env)
    model_dir_candidate = _model_dir_candidate(teacher_name, model_dir, dir_env)
    checkpoint_candidate = _checkpoint_candidate(teacher_name, checkpoint, checkpoint_env)
    adapter_valid = adapter is not None and ":" in adapter and all(adapter.split(":", 1))

    missing: list[JsonDict] = []
    if teacher_name == "moge":
        if adapter is not None and not adapter_valid:
            missing.append(
                {
                    "field": adapter_env,
                    "accepted_source": "environment variable with value module:function",
                    "value": adapter,
                    "exists": False,
                    "requirement": "must be module:function when set",
                }
            )
        missing.append(
            {
                "field": "moge_import",
                "accepted_source": "official MoGe package exposing moge.model.v2.MoGeModel",
                "value": "from moge.model.v2 import MoGeModel",
                "exists": None,
                "requirement": "MoGe must be installed or HOMEBRAIN_MOGE_DIR must point at an importable checkout",
            }
        )
        model_id_configured = bool(os.environ.get("HOMEBRAIN_MOGE_MODEL_ID"))
        default_download_allowed = os.environ.get("HOMEBRAIN_MOGE_ALLOW_DOWNLOAD") == "1"
        checkpoint_exists = checkpoint_candidate is not None and checkpoint_candidate.exists()
        if not checkpoint_exists and not model_id_configured and not default_download_allowed:
            missing.append(
                {
                    "field": "model_source",
                    "accepted_source": (
                        f"--checkpoint, {checkpoint_env}, HOMEBRAIN_MOGE_MODEL_ID, "
                        "or HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1"
                    ),
                    "value": checkpoint_candidate.as_posix() if checkpoint_candidate is not None else None,
                    "exists": False,
                    "requirement": "operator-approved checkpoint/model source; default download is disabled unless allowed",
                }
        )
        return missing
    elif adapter is not None and not adapter_valid:
        missing.append(
            {
                "field": adapter_env,
                "accepted_source": "environment variable with value module:function",
                "value": adapter,
                "exists": False,
                "requirement": "must be module:function when set",
            }
        )

    if model_dir_candidate is None or not model_dir_candidate.exists():
        missing.append(
            {
                "field": "model_dir",
                "accepted_source": f"--model-dir, {dir_env}, or external/{teacher_name}",
                "value": model_dir_candidate.as_posix() if model_dir_candidate is not None else None,
                "exists": False,
                "requirement": "existing local teacher checkout",
            }
        )
    if checkpoint_candidate is None or not checkpoint_candidate.exists():
        missing.append(
            {
                "field": "checkpoint",
                "accepted_source": f"--checkpoint, {checkpoint_env}, or external/{teacher_name}/checkpoints",
                "value": checkpoint_candidate.as_posix() if checkpoint_candidate is not None else None,
                "exists": False,
                "requirement": "existing local teacher checkpoint",
            }
        )
    return missing


def _model_dir_candidate(teacher_name: str, model_dir: str | Path | None, env_name: str) -> Path | None:
    if model_dir is not None:
        return Path(model_dir)
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value)
    return Path.cwd() / "external" / teacher_name


def _checkpoint_candidate(teacher_name: str, checkpoint: str | Path | None, env_name: str) -> Path | None:
    if checkpoint is not None:
        return Path(checkpoint)
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value)
    for candidate_name in (f"{teacher_name}.pt", "model.pt"):
        candidate = Path.cwd() / "external" / teacher_name / "checkpoints" / candidate_name
        if candidate.exists():
            return candidate
    return Path.cwd() / "external" / teacher_name / "checkpoints" / f"{teacher_name}.pt"


def _audit_command(
    *,
    log_dir: Path,
    scene_teacher_dir: Path,
    qa_path: Path,
    out_json: Path,
    out_md: Path,
) -> str:
    return subprocess.list2cmdline(
        [
            "python",
            "-m",
            "homebrain.tools.audit_scene_teacher_signal",
            "--log",
            log_dir.as_posix(),
            "--scene-teacher",
            scene_teacher_dir.as_posix(),
            "--qa",
            qa_path.as_posix(),
            "--out-json",
            out_json.as_posix(),
            "--out-md",
            out_md.as_posix(),
        ]
    )


def _retry_command(args: argparse.Namespace) -> str:
    return _probe_retry_command_from_values(
        frames_dir=args.frames,
        out_dir=args.out,
        camera_name=args.camera,
        fps=args.fps,
        teacher_name=args.teacher,
        backend_name=args.backend,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        max_frames=args.max_frames,
        stride=args.stride,
    )


def _probe_retry_command_from_values(
    *,
    frames_dir: str | Path,
    out_dir: str | Path,
    camera_name: str,
    fps: float,
    teacher_name: str,
    backend_name: str,
    model_dir: str | Path | None,
    checkpoint: str | Path | None,
    device: str | None,
    max_frames: int | None,
    stride: int,
) -> str:
    command: list[str] = [
        "python",
        "-m",
        "homebrain.tools.run_owned_geometry_probe",
        "--frames",
        str(frames_dir),
        "--out",
        str(out_dir),
        "--camera",
        str(camera_name),
        "--fps",
        str(fps),
        "--teacher",
        str(teacher_name),
        "--backend",
        str(backend_name),
        "--owned-or-license-approved",
    ]
    for flag, value in (
        ("--model-dir", model_dir),
        ("--checkpoint", checkpoint),
        ("--device", device),
        ("--max-frames", max_frames),
        ("--stride", None if stride == 1 else stride),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    return subprocess.list2cmdline(command)


def _finish_probe_result(result: JsonDict, result_json_path: Path, result_md_path: Path, started: float) -> JsonDict:
    result["runtime_sec"] = float(time.perf_counter() - started)
    _write_json(result_json_path, result)
    _write_result_markdown(result_md_path, result)
    return result


def _write_result_markdown(path: Path, result: JsonDict) -> None:
    steps = result.get("steps", {})
    lines = [
        "# Owned Geometry Probe Result",
        "",
        f"- status: `{result.get('status')}`",
        f"- teacher: `{result['inputs']['teacher']}`",
        f"- backend: `{result['inputs']['backend']}`",
        f"- route: `{result['paths']['route_dir']}`",
        f"- teacher_artifacts_exist: `{str(result.get('teacher_artifacts_exist')).lower()}`",
        f"- qa_exists: `{str(result.get('qa_exists')).lower()}`",
        f"- audit_exists: `{str(result.get('audit_exists')).lower()}`",
        f"- visual_review_exists: `{str(result.get('visual_review_exists')).lower()}`",
        "",
        "## Steps",
    ]
    for name in ("image_sequence_ingest", "scene_teacher_run", "scene_teacher_qa", "signal_audit", "visual_review"):
        step = steps.get(name, {})
        lines.append(
            f"- {name}: attempted=`{str(step.get('attempted')).lower()}` ok=`{str(step.get('ok')).lower()}`"
        )
    if result.get("status") == STATUS_BLOCKED_MISSING_TEACHER_SETUP:
        lines.extend(
            [
                "",
                "## Missing Setup",
                "The real teacher did not start because required local setup was absent. No alternate backend was used.",
            ]
        )
        for item in result.get("missing_setup_fields", []):
            if isinstance(item, dict):
                lines.append(
                    f"- `{item.get('field')}`: {item.get('requirement')} "
                    f"(accepted source: {item.get('accepted_source')}, value: `{item.get('value')}`)"
                )
        if result.get("retry_command"):
            lines.extend(["", "## Retry", f"`{result['retry_command']}`"])
    elif result.get("status") == STATUS_REVIEW_ONLY_NOT_TRAINABLE:
        lines.extend(
            [
                "",
                "Fake backend output is review-only and not trainable.",
            ]
        )
    elif result.get("retry_command"):
        lines.extend(["", "## Retry", f"`{result['retry_command']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_json(path: str | Path, data: JsonDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(data))
        handle.write("\n")


def _review_panels(
    scene_root: Path,
    artifacts: Any,
    *,
    frame: JsonDict | None = None,
    log_root: Path | None = None,
) -> list[np.ndarray]:
    if not isinstance(artifacts, dict):
        raise ValueError("scene-teacher frame is missing artifacts")
    depth = _load_optional_artifact(scene_root, artifacts, "depth")
    confidence = _load_optional_artifact(scene_root, artifacts, "confidence")
    floor = _load_optional_artifact(scene_root, artifacts, "floor_traversable_mask")
    obstacle = _load_optional_artifact(scene_root, artifacts, "obstacle_risk_mask")
    if depth is None:
        point_map = _load_optional_artifact(scene_root, artifacts, "point_map")
        if point_map is not None and point_map.ndim == 3 and point_map.shape[-1] == 3:
            depth = point_map[:, :, 2]
    if depth is None:
        raise ValueError("visual review requires depth or point_map artifacts")
    base = _normalize_gray(np.asarray(depth, dtype=np.float32))
    height, width = _thumbnail_shape(base)
    source_panel = _source_rgb_panel(frame=frame, log_root=log_root, shape=(height, width))
    depth_panel = _gray_rgb(_resize_nearest(base, (height, width)))
    confidence_panel = _tint(
        _resize_nearest(_normalize_gray(confidence if confidence is not None else np.ones_like(base)), (height, width)),
        (40, 180, 80),
    )
    floor_panel = _tint(
        _resize_nearest(_normalize_gray(floor if floor is not None else np.zeros_like(base)), (height, width)),
        (40, 190, 110),
    )
    obstacle_panel = _tint(
        _resize_nearest(_normalize_gray(obstacle if obstacle is not None else np.zeros_like(base)), (height, width)),
        (220, 55, 55),
    )
    return [source_panel, depth_panel, confidence_panel, floor_panel, obstacle_panel]


def _source_rgb_panel(*, frame: JsonDict | None, log_root: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if frame is None or log_root is None:
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    data_ref = frame.get("source_data_ref")
    if not isinstance(data_ref, str):
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    source_path = log_root / data_ref
    try:
        rgb = _load_review_source_rgb(source_path, frame)
    except Exception:  # noqa: BLE001 - visual review should still show teacher panels.
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    return _resize_nearest(np.asarray(rgb, dtype=np.uint8), shape)


def _load_review_source_rgb(path: Path, frame: JsonDict) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".ppm", ".pgm"}:
        return _read_review_netpbm(path)
    if suffix in {".jpg", ".jpeg", ".png"}:
        return _read_review_encoded_image(path)
    width = _positive_int(frame.get("width"))
    height = _positive_int(frame.get("height"))
    frame_format = str(frame.get("format", "")).lower()
    if width is None or height is None:
        raise ValueError("raw source frame width/height are missing")
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if frame_format == "rgb8":
        expected = width * height * 3
        if raw.size != expected:
            raise ValueError(f"raw rgb8 source has {raw.size} bytes, expected {expected}")
        return raw.reshape((height, width, 3)).copy()
    if frame_format == "gray8":
        expected = width * height
        if raw.size != expected:
            raise ValueError(f"raw gray8 source has {raw.size} bytes, expected {expected}")
        gray = raw.reshape((height, width)).copy()
        return np.stack([gray, gray, gray], axis=2)
    raise ValueError(f"unsupported source frame format for review: {frame_format!r}")


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _read_review_netpbm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    index = 0

    def read_token() -> bytes:
        nonlocal index
        while index < len(data):
            char = data[index : index + 1]
            index += 1
            if char == b"#":
                while index < len(data) and data[index : index + 1] not in {b"\n", b"\r"}:
                    index += 1
                continue
            if char.isspace():
                continue
            token = bytearray(char)
            while index < len(data) and not data[index : index + 1].isspace():
                token.extend(data[index : index + 1])
                index += 1
            return bytes(token)
        raise ValueError("unexpected EOF in Netpbm header")

    magic = read_token()
    width = int(read_token())
    height = int(read_token())
    max_value = int(read_token())
    if max_value <= 0 or max_value > 255:
        raise ValueError(f"unsupported Netpbm max value {max_value}")
    while index < len(data) and data[index : index + 1].isspace():
        index += 1
    channels = 3 if magic == b"P6" else 1 if magic == b"P5" else 0
    if channels == 0:
        raise ValueError(f"unsupported Netpbm magic {magic!r}")
    raw = np.frombuffer(data[index : index + width * height * channels], dtype=np.uint8)
    if raw.size != width * height * channels:
        raise ValueError("truncated Netpbm payload")
    if channels == 3:
        return raw.reshape((height, width, 3)).copy()
    gray = raw.reshape((height, width)).copy()
    return np.stack([gray, gray, gray], axis=2)


def _read_review_encoded_image(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore

        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not decode source image {path}") from exc


def _load_optional_artifact(scene_root: Path, artifacts: JsonDict, kind: str) -> np.ndarray | None:
    record = artifacts.get(kind)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    path = scene_root / str(record["path"])
    if not path.exists():
        return None
    return load_array(path)


def _normalize_gray(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError(f"expected 2D review panel, got shape {values.shape}")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    minimum = float(np.percentile(finite, 1))
    maximum = float(np.percentile(finite, 99))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - np.float32(minimum)) / np.float32(maximum - minimum)
    normalized = np.where(np.isfinite(normalized), normalized, np.float32(0.0))
    return np.clip(normalized * np.float32(255.0), 0, 255).astype(np.uint8)


def _thumbnail_shape(panel: np.ndarray, max_side: int = 96, min_side: int = 16) -> tuple[int, int]:
    height, width = panel.shape
    scale = max_side / float(max(height, width, 1))
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    if max(height, width) < min_side:
        scale = min_side / float(max(height, width, 1))
        out_h = max(1, int(round(height * scale)))
        out_w = max(1, int(round(width * scale)))
    return out_h, out_w


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    out_h, out_w = shape
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]


def _gray_rgb(panel: np.ndarray) -> np.ndarray:
    values = np.asarray(panel, dtype=np.uint8)
    return np.stack([values, values, values], axis=2)


def _tint(panel: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(panel, dtype=np.float32) / np.float32(255.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for channel, channel_value in enumerate(color):
        rgb[:, :, channel] = np.clip(values * np.float32(channel_value), 0, 255).astype(np.uint8)
    return rgb


def _join_with_gap(items: list[np.ndarray], *, gap: int, axis: int) -> np.ndarray:
    if not items:
        raise ValueError("cannot join empty image list")
    if axis == 1:
        height = max(item.shape[0] for item in items)
        padded = [_pad_to(item, height, item.shape[1]) for item in items]
        spacer = np.full((height, gap, 3), 255, dtype=np.uint8)
        pieces: list[np.ndarray] = []
        for index, item in enumerate(padded):
            if index:
                pieces.append(spacer)
            pieces.append(item)
        return np.concatenate(pieces, axis=1)
    if axis == 0:
        width = max(item.shape[1] for item in items)
        padded = [_pad_to(item, item.shape[0], width) for item in items]
        spacer = np.full((gap, width, 3), 255, dtype=np.uint8)
        pieces = []
        for index, item in enumerate(padded):
            if index:
                pieces.append(spacer)
            pieces.append(item)
        return np.concatenate(pieces, axis=0)
    raise ValueError(f"unsupported join axis: {axis}")


def _pad_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[: image.shape[0], : image.shape[1], :] = image
    return canvas


def _write_ppm(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"PPM image must be HxWx3, got {image.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _channels = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.asarray(image, dtype=np.uint8).tobytes(order="C"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the single owned-frame production geometry probe."
    )
    parser.add_argument("--frames", required=True, help="Directory containing owned or approved image frames.")
    parser.add_argument("--out", required=True, help="Output probe directory.")
    parser.add_argument("--camera", required=True, help="Camera name for ingested frames.")
    parser.add_argument("--fps", required=True, type=float, help="Source frame rate.")
    parser.add_argument("--teacher", required=True, choices=["moge", "vggt"], help="Geometry teacher.")
    parser.add_argument("--backend", required=True, choices=["real", "fake"], help="Teacher backend.")
    parser.add_argument(
        "--owned-or-license-approved",
        required=True,
        action="store_true",
        help="Required explicit approval for the input frames.",
    )
    parser.add_argument("--model-dir", default=None, help="Optional local teacher checkout path.")
    parser.add_argument("--checkpoint", default=None, help="Optional local checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional teacher device.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap.")
    parser.add_argument("--stride", type=int, default=1, help="Optional input/teacher stride.")
    parser.add_argument(
        "--allow-blocked-exit-zero",
        action="store_true",
        help="Diagnostic only: return exit code 0 for BLOCKED_* probe statuses while still writing result files.",
    )
    args = parser.parse_args(argv)

    result = run_owned_geometry_probe(
        frames_dir=args.frames,
        out_dir=args.out,
        camera_name=args.camera,
        fps=args.fps,
        teacher_name=args.teacher,
        backend_name=args.backend,
        owned_or_license_approved=args.owned_or_license_approved,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        max_frames=args.max_frames,
        stride=args.stride,
        retry_command=_retry_command(args),
    )
    print(json.dumps({"status": result["status"], "result": result["paths"]["result_json"]}, sort_keys=True))
    status = str(result["status"])
    if status.startswith("FAILED_"):
        return 1
    if status.startswith("BLOCKED_") and not args.allow_blocked_exit_zero:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
