from __future__ import annotations

from pathlib import Path
from typing import Any

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, write_json
from homebrain.data.tum_rgbd_route import RouteSequence, discover_tum_rgbd_routes, load_tum_rgbd_sequence
from homebrain.messages.schema import JsonDict

BONN_RGBD_SCHEMA_VERSION = "homebrain.datasets.bonn_rgbd.v0"
BONN_RGBD_ROUTE_ASSOCIATIONS_FILE = "bonn_rgbd_associations.json"
BONN_RGBD_SOURCE_URL = "https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/"
BONN_RGBD_LICENSE_NAME = "Bonn RGB-D Dynamic dataset terms"
BONN_RGBD_LICENSE_REVIEW_STATUS = "pending_human_review"
BONN_RGBD_DEPTH_SCALE = 5000.0
BONN_RGBD_DATASET_NAME = "bonn_rgbd_dynamic"


def bonn_sequence_dir(out_dir: str | Path, sequence: str) -> Path:
    return Path(out_dir) / sequence


def discover_bonn_rgbd_routes(input_root: str | Path) -> list[Path]:
    return discover_tum_rgbd_routes(input_root)


def validate_bonn_sequence_dir(sequence_dir: str | Path) -> None:
    root = Path(sequence_dir)
    required = ("rgb.txt", "depth.txt", "groundtruth.txt")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bonn RGB-D Dynamic sequence is missing required files: {', '.join(missing)}")


def load_bonn_rgbd_sequence(
    sequence_dir: str | Path,
    *,
    route_id: str | None = None,
    split_unit_id: str | None = None,
    max_association_s: float = 0.02,
) -> RouteSequence:
    return load_tum_rgbd_sequence(
        sequence_dir,
        dataset_name=BONN_RGBD_DATASET_NAME,
        route_id=route_id,
        split_unit_id=split_unit_id,
        max_association_s=max_association_s,
        intrinsics={
            "fx": 0.0,
            "fy": 0.0,
            "cx": 0.0,
            "cy": 0.0,
            "depth_scale": BONN_RGBD_DEPTH_SCALE,
            "source": "bonn_rgbd_intrinsics_missing_use_route_metadata_if_available",
        },
    )


def missing_bonn_report(
    out_dir: str | Path,
    *,
    source_root: str | Path,
    reason: str = "missing_bonn_rgbd_root",
    synthetic_or_fixture: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    output = Path(out_dir)
    report: JsonDict = {
        "schema_version": BONN_RGBD_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "dataset_name": BONN_RGBD_DATASET_NAME,
        "source_root": Path(source_root).as_posix(),
        "official_page": BONN_RGBD_SOURCE_URL,
        "license_name": BONN_RGBD_LICENSE_NAME,
        "license_review_status": BONN_RGBD_LICENSE_REVIEW_STATUS,
        "accepted": False,
        "reason": reason,
        "acceptance_reasons": [reason],
        "synthetic_or_fixture": bool(synthetic_or_fixture),
        "real_dataset_required": True,
        "invented_data": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }
    if extra:
        report.update(extra)
    path = output / "report.json"
    write_json(path, report, pretty=True)
    return path
