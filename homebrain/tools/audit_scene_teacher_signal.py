from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from homebrain.data.spatial_dataset import write_json
from homebrain.ingest.metadata import load_route_metadata
from homebrain.messages.schema import (
    CommandEvent,
    Event,
    FrameEvent,
    ImuEvent,
    JsonDict,
    OdomEvent,
    PoseEvent,
    WheelEvent,
    deterministic_json,
)
from homebrain.replay.segment_log import read_events
from homebrain.teachers.scene_teacher import load_scene_teacher_manifest

SCENE_TEACHER_SIGNAL_AUDIT_SCHEMA_VERSION = "homebrain.scene_teacher_signal_audit.v0"
NEXT_ALLOWED_USES = ("review_only", "geometry_pretrain_candidate", "blocked")
GEOMETRY_PRETRAIN_SCALE_STATUSES = {"metric", "metric_or_route_measured", "measured_metric"}
VALIDITY_MIN = 0.95


def audit_scene_teacher_signal(
    *,
    log_dir: str | Path,
    scene_teacher_dir: str | Path,
    qa_path: str | Path,
    out_json: str | Path,
    out_md: str | Path,
    command: str | None = None,
) -> JsonDict:
    started = time.perf_counter()
    log_root = Path(log_dir)
    scene_root = Path(scene_teacher_dir)
    qa_file = Path(qa_path)
    manifest = load_scene_teacher_manifest(scene_root)
    qa = _read_json(qa_file)
    events = _order_events_for_audit(read_events(log_root))
    metadata = _load_metadata_for_audit(log_root)
    route_truth = _route_metadata_sensor_truth(metadata=metadata, events=events)
    mask_sources = _mask_source_distribution(scene_root=scene_root, manifest=manifest)

    frame_count = int(manifest.get("frame_count", qa.get("frame_count", 0)))
    mock = bool(manifest.get("mock", qa.get("mock", False)))
    synthetic = bool(manifest.get("synthetic", qa.get("synthetic", False)))
    real_perception = bool(manifest.get("real_perception", qa.get("real_perception", False)))
    scale_status = str(manifest.get("scale_status", qa.get("scale_status", "unknown")))
    depth_valid_ratio = _float(qa.get("depth_valid_ratio"))
    confidence_valid_ratio = _float(qa.get("confidence_valid_ratio"))
    pose_valid_ratio = _float(qa.get("pose_valid_ratio"))
    temporal_consistency = _float(qa.get("temporal_geometry_consistency"))
    action_supervision_ok = _action_supervision_ok(manifest=manifest, route_truth=route_truth)
    robot_frame_truth = _robot_frame_truth(manifest=manifest, route_truth=route_truth)
    hard_blockers = _hard_blockers(
        manifest=manifest,
        qa=qa,
        route_truth=route_truth,
        frame_count=frame_count,
        depth_valid_ratio=depth_valid_ratio,
        confidence_valid_ratio=confidence_valid_ratio,
        pose_valid_ratio=pose_valid_ratio,
    )
    promotable_to_spatial_pack = bool(
        not hard_blockers
        and qa.get("promotable_to_spatial_pack") is True
        and real_perception
        and not mock
        and not synthetic
        and scale_status in GEOMETRY_PRETRAIN_SCALE_STATUSES
        and route_truth["owned_or_license_approved"] is True
        and not action_supervision_ok
    )
    next_allowed_use = _next_allowed_use(
        hard_blockers=hard_blockers,
        promotable_to_spatial_pack=promotable_to_spatial_pack,
    )
    recommendations = _recommendations(
        next_allowed_use=next_allowed_use,
        mock=mock,
        synthetic=synthetic,
        real_perception=real_perception,
        scale_status=scale_status,
        hard_blockers=hard_blockers,
    )

    report: JsonDict = {
        "schema_version": SCENE_TEACHER_SIGNAL_AUDIT_SCHEMA_VERSION,
        "goal": "15B real owned-route scene-teacher signal gate",
        "inputs": {
            "log_dir": log_root.as_posix(),
            "scene_teacher_dir": scene_root.as_posix(),
            "qa_path": qa_file.as_posix(),
        },
        "frame_count": frame_count,
        "real_perception": real_perception,
        "mock": mock,
        "synthetic": synthetic,
        "mock_or_synthetic": bool(mock or synthetic),
        "scale_status": scale_status,
        "depth_validity": {
            "depth_valid_ratio": depth_valid_ratio,
            "threshold": VALIDITY_MIN,
            "pass": depth_valid_ratio >= VALIDITY_MIN,
        },
        "confidence_validity": {
            "confidence_valid_ratio": confidence_valid_ratio,
            "threshold": VALIDITY_MIN,
            "pass": confidence_valid_ratio >= VALIDITY_MIN,
        },
        "pose_validity": {
            "pose_valid_ratio": pose_valid_ratio,
            "threshold": VALIDITY_MIN,
            "pass": pose_valid_ratio >= VALIDITY_MIN,
        },
        "track_valid_ratio": _float(qa.get("track_valid_ratio")),
        "temporal_consistency": temporal_consistency,
        "mask_source_distribution": mask_sources,
        "route_metadata_sensor_truth": route_truth,
        "robot_frame_truth": robot_frame_truth,
        "action_supervision_ok": action_supervision_ok,
        "promotable_to_spatial_pack": promotable_to_spatial_pack,
        "next_allowed_use": next_allowed_use,
        "hard_blockers": hard_blockers,
        "qa_quarantine_reasons": sorted(str(item) for item in qa.get("quarantine_reasons", []) if isinstance(item, str)),
        "recommendations": recommendations,
        "safety_flags": {
            "replay_only": manifest.get("replay_only") is True,
            "not_executed": manifest.get("not_executed") is True,
            "control_safe": manifest.get("control_safe") is True,
            "product_training_approved": manifest.get("product_training_approved") is True,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": manifest.get("raw_pwm_emitted") is True,
        },
        "hard_constraints": {
            "trained_model": False,
            "trajectory_scorer_changed": False,
            "policy_eval_loop_added": False,
            "sam2_used": False,
            "ros_nav2_isaac_habitat_sim_added": False,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": manifest.get("raw_pwm_emitted") is True,
        },
        "commands_run": [command] if command else [],
        "runtime_sec": float(time.perf_counter() - started),
    }
    if report["next_allowed_use"] not in NEXT_ALLOWED_USES:
        raise AssertionError(f"invalid next_allowed_use: {report['next_allowed_use']}")
    write_json(out_json, report, pretty=True)
    _write_markdown(Path(out_md), report)
    return report


def _load_metadata_for_audit(log_root: Path) -> JsonDict | None:
    try:
        return load_route_metadata(log_root)
    except Exception as exc:  # noqa: BLE001 - audit reports malformed route metadata.
        return {"_metadata_load_error": str(exc)}


def _order_events_for_audit(events: list[Event]) -> list[Event]:
    return [
        event
        for _index, event in sorted(
            enumerate(events),
            key=lambda indexed: (indexed[1].timestamp_ns, indexed[0]),
        )
    ]


def _route_metadata_sensor_truth(*, metadata: JsonDict | None, events: list[Event]) -> JsonDict:
    event_counts = {
        "frame": sum(isinstance(event, FrameEvent) for event in events),
        "imu": sum(isinstance(event, ImuEvent) for event in events),
        "wheel": sum(isinstance(event, WheelEvent) for event in events),
        "odom": sum(isinstance(event, OdomEvent) for event in events),
        "pose": sum(isinstance(event, PoseEvent) for event in events),
        "command": sum(isinstance(event, CommandEvent) for event in events),
    }
    violations: list[str] = []
    if metadata is None:
        return {
            "metadata_present": False,
            "metadata_load_error": None,
            "source_type": None,
            "owned_or_license_approved_explicit": False,
            "owned_or_license_approved": False,
            "event_counts": event_counts,
            "metadata_sensor_claims": {},
            "robot_frame_truth_allowed": False,
            "robot_frame_truth_claimed_by_route": False,
            "truth_pass": False,
            "violations": ["missing_route_metadata", "owned_or_license_approved_not_explicit"],
        }
    if isinstance(metadata.get("_metadata_load_error"), str):
        return {
            "metadata_present": True,
            "metadata_load_error": metadata["_metadata_load_error"],
            "source_type": None,
            "owned_or_license_approved_explicit": False,
            "owned_or_license_approved": False,
            "event_counts": event_counts,
            "metadata_sensor_claims": {},
            "robot_frame_truth_allowed": False,
            "robot_frame_truth_claimed_by_route": False,
            "truth_pass": False,
            "violations": ["route_metadata_load_failed", "owned_or_license_approved_not_explicit"],
        }

    owned_explicit = isinstance(metadata.get("owned_or_license_approved"), bool)
    owned_approved = metadata.get("owned_or_license_approved") is True
    if not owned_explicit:
        violations.append("owned_or_license_approved_not_explicit")
    if not owned_approved:
        violations.append("owned_or_license_not_approved")

    claims = {
        "has_imu": _bool_or_none(metadata.get("has_imu")),
        "has_wheel_odometry": _bool_or_none(metadata.get("has_wheel_odometry")),
        "has_odometry": _bool_or_none(metadata.get("has_odometry")),
        "has_commands": _bool_or_none(metadata.get("has_commands")),
        "has_camera_to_base_transform": _bool_or_none(metadata.get("has_camera_to_base_transform")),
        "has_robot_base_pose": _bool_or_none(metadata.get("has_robot_base_pose")),
        "has_groundtruth_pose": _bool_or_none(metadata.get("has_groundtruth_pose")),
    }
    _check_sensor_claim(
        violations,
        sensor="imu",
        event_count=event_counts["imu"],
        claim=claims["has_imu"],
    )
    _check_sensor_claim(
        violations,
        sensor="wheel_odometry",
        event_count=event_counts["wheel"],
        claim=claims["has_wheel_odometry"],
    )
    _check_sensor_claim(
        violations,
        sensor="odometry",
        event_count=event_counts["odom"],
        claim=claims["has_odometry"],
    )
    _check_sensor_claim(
        violations,
        sensor="commands",
        event_count=event_counts["command"],
        claim=claims["has_commands"],
    )

    measured_camera_to_base = bool(
        claims["has_camera_to_base_transform"] is True
        and metadata.get("review_assumed_extrinsics") is not True
        and metadata.get("extrinsics_source") not in {"assumed_review_only", "missing", "missing_not_supplied"}
    )
    base_pose_or_odom = bool(
        claims["has_robot_base_pose"] is True
        or claims["has_groundtruth_pose"] is True
        or claims["has_odometry"] is True
        or claims["has_wheel_odometry"] is True
    )
    robot_frame_truth_allowed = bool(measured_camera_to_base and base_pose_or_odom)
    route_robot_truth_claim = bool(
        metadata.get("robot_frame_truth") is True or metadata.get("robot_frame_truth_candidate") is True
    )
    if route_robot_truth_claim and not robot_frame_truth_allowed:
        violations.append("route_claims_robot_frame_truth_without_measured_camera_to_base_and_base_pose_or_odom")

    return {
        "metadata_present": True,
        "metadata_load_error": None,
        "source_type": metadata.get("source_type"),
        "owned_or_license_approved_explicit": owned_explicit,
        "owned_or_license_approved": owned_approved,
        "event_counts": event_counts,
        "metadata_sensor_claims": claims,
        "missing_sensor_notices": metadata.get("missing_sensor_notices", []),
        "measured_camera_to_base": measured_camera_to_base,
        "base_pose_or_odom": base_pose_or_odom,
        "robot_frame_truth_allowed": robot_frame_truth_allowed,
        "robot_frame_truth_claimed_by_route": route_robot_truth_claim,
        "truth_pass": not violations,
        "violations": sorted(set(violations)),
    }


def _check_sensor_claim(violations: list[str], *, sensor: str, event_count: int, claim: bool | None) -> None:
    if event_count > 0 and claim is not True:
        violations.append(f"{sensor}_events_present_but_metadata_does_not_claim_sensor")
    if event_count == 0 and claim is True:
        violations.append(f"metadata_claims_{sensor}_but_no_events_present")


def _robot_frame_truth(*, manifest: JsonDict, route_truth: JsonDict) -> bool:
    scene_claim = manifest.get("robot_frame_truth") is True
    if scene_claim and route_truth.get("robot_frame_truth_allowed") is not True:
        return False
    return bool(scene_claim and route_truth.get("robot_frame_truth_allowed") is True)


def _action_supervision_ok(*, manifest: JsonDict, route_truth: JsonDict) -> bool:
    if manifest.get("action_supervision_ok") is not True:
        return False
    return bool(
        route_truth.get("robot_frame_truth_allowed") is True
        and route_truth.get("metadata_sensor_claims", {}).get("has_commands") is True
    )


def _mask_source_distribution(*, scene_root: Path, manifest: JsonDict) -> JsonDict:
    distributions: dict[str, dict[str, int]] = {
        "confidence_source": {},
        "validity_mask_source": {},
        "floor_traversable_mask_source": {},
        "obstacle_risk_mask_source": {},
        "dynamic_motion_mask_source": {},
    }
    for frame in manifest.get("frames", []):
        if not isinstance(frame, dict):
            continue
        metadata_path = frame.get("metadata_path")
        if not isinstance(metadata_path, str):
            continue
        try:
            metadata = _read_json(scene_root / metadata_path)
        except Exception:  # noqa: BLE001 - missing metadata is already counted by QA.
            continue
        for field, counts in distributions.items():
            source = metadata.get(field)
            if not isinstance(source, str):
                source = "missing"
            counts[source] = counts.get(source, 0) + 1
    return {field: dict(sorted(counts.items())) for field, counts in distributions.items()}


def _hard_blockers(
    *,
    manifest: JsonDict,
    qa: JsonDict,
    route_truth: JsonDict,
    frame_count: int,
    depth_valid_ratio: float,
    confidence_valid_ratio: float,
    pose_valid_ratio: float,
) -> list[str]:
    blockers: list[str] = []
    blockers.extend(str(item) for item in route_truth.get("violations", []) if isinstance(item, str))
    if frame_count <= 0:
        blockers.append("no_scene_teacher_frames")
    if int(qa.get("missing_artifact_count", 0)) > 0:
        blockers.append("missing_scene_teacher_artifacts")
    if int(qa.get("artifact_shape_error_count", 0)) > 0:
        blockers.append("invalid_scene_teacher_artifact_shapes")
    if depth_valid_ratio < VALIDITY_MIN:
        blockers.append("depth_validity_below_gate")
    if confidence_valid_ratio < VALIDITY_MIN:
        blockers.append("confidence_validity_below_gate")
    if pose_valid_ratio < VALIDITY_MIN:
        blockers.append("pose_validity_below_gate")
    if manifest.get("control_safe") is True:
        blockers.append("scene_teacher_control_safe_claim_present")
    if manifest.get("product_training_approved") is True:
        blockers.append("scene_teacher_product_training_claim_present")
    if manifest.get("raw_pwm_emitted") is True:
        blockers.append("raw_pwm_emitted")
    if manifest.get("replay_only") is not True or manifest.get("not_executed") is not True:
        blockers.append("scene_teacher_replay_safety_flags_missing")
    if manifest.get("robot_frame_truth") is True and route_truth.get("robot_frame_truth_allowed") is not True:
        blockers.append("scene_teacher_robot_frame_truth_without_route_robot_frame_truth")
    if manifest.get("action_supervision_ok") is True:
        blockers.append("scene_teacher_action_supervision_claim_present")
    return sorted(set(blockers))


def _next_allowed_use(*, hard_blockers: list[str], promotable_to_spatial_pack: bool) -> str:
    if hard_blockers:
        return "blocked"
    if promotable_to_spatial_pack:
        return "geometry_pretrain_candidate"
    return "review_only"


def _recommendations(
    *,
    next_allowed_use: str,
    mock: bool,
    synthetic: bool,
    real_perception: bool,
    scale_status: str,
    hard_blockers: list[str],
) -> list[str]:
    if next_allowed_use == "blocked":
        return [
            "Repair the listed hard blockers before using this SceneTeacherPack for any training candidate.",
            "Do not invent IMU, odometry, commands, camera-to-base, or robot-frame truth to clear the gate.",
        ]
    if mock or synthetic or not real_perception:
        return [
            "Use this artifact for structural review only; fake or synthetic scene-teacher output is never promotable.",
            "Run the real backend only after local MoGe assets and an explicitly approved owned route are present.",
        ]
    if scale_status not in GEOMETRY_PRETRAIN_SCALE_STATUSES:
        return [
            "Use this artifact for review only until scale is metric or tied to measured route calibration.",
        ]
    return [
        "Review generated geometry before packing; this audit only marks a candidate, not product-training approval.",
    ]


def _read_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _write_markdown(path: Path, report: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    route = report["route_metadata_sensor_truth"]
    lines = [
        "# Scene Teacher Signal Audit",
        "",
        "## Result",
        f"- next_allowed_use: `{report['next_allowed_use']}`",
        f"- promotable_to_spatial_pack: `{str(report['promotable_to_spatial_pack']).lower()}`",
        f"- frame_count: `{report['frame_count']}`",
        f"- real_perception: `{str(report['real_perception']).lower()}`",
        f"- mock_or_synthetic: `{str(report['mock_or_synthetic']).lower()}`",
        f"- scale_status: `{report['scale_status']}`",
        "",
        "## Geometry",
        f"- depth_valid_ratio: `{report['depth_validity']['depth_valid_ratio']}`",
        f"- confidence_valid_ratio: `{report['confidence_validity']['confidence_valid_ratio']}`",
        f"- pose_valid_ratio: `{report['pose_validity']['pose_valid_ratio']}`",
        f"- temporal_consistency: `{report['temporal_consistency']}`",
        "",
        "## Route Truth",
        f"- source_type: `{route.get('source_type')}`",
        f"- owned_or_license_approved: `{str(route.get('owned_or_license_approved')).lower()}`",
        f"- truth_pass: `{str(route.get('truth_pass')).lower()}`",
        f"- robot_frame_truth_allowed: `{str(route.get('robot_frame_truth_allowed')).lower()}`",
        f"- robot_frame_truth: `{str(report['robot_frame_truth']).lower()}`",
        f"- action_supervision_ok: `{str(report['action_supervision_ok']).lower()}`",
        "",
        "## Blockers",
        *([f"- `{item}`" for item in report["hard_blockers"]] if report["hard_blockers"] else ["- `none`"]),
        "",
        "## Recommendations",
        *[f"- {item}" for item in report["recommendations"]],
        "",
        "## Safety",
        "- No model was trained, no trajectory scorer or policy loop was added, and no cmd_vel/raw PWM was emitted.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _iterable_strings(values: Iterable[str]) -> list[str]:
    return [str(value) for value in values]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether a SceneTeacherPack has usable owned-route spatial signal.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--scene-teacher", required=True, help="Input SceneTeacherPack directory.")
    parser.add_argument("--qa", required=True, help="Input SceneTeacherPack QA JSON.")
    parser.add_argument("--out-json", required=True, help="Output signal audit JSON.")
    parser.add_argument("--out-md", required=True, help="Output signal audit Markdown.")
    args = parser.parse_args(argv)
    command = "python -m homebrain.tools.audit_scene_teacher_signal " + " ".join(_iterable_strings(sys.argv[1:]))
    report = audit_scene_teacher_signal(
        log_dir=args.log,
        scene_teacher_dir=args.scene_teacher,
        qa_path=args.qa,
        out_json=args.out_json,
        out_md=args.out_md,
        command=command,
    )
    print(deterministic_json({"next_allowed_use": report["next_allowed_use"], "frame_count": report["frame_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
