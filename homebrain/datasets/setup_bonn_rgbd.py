from __future__ import annotations

import argparse
from pathlib import Path

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, write_json
from homebrain.datasets.bonn_rgbd import (
    BONN_RGBD_DATASET_NAME,
    BONN_RGBD_LICENSE_NAME,
    BONN_RGBD_LICENSE_REVIEW_STATUS,
    BONN_RGBD_SCHEMA_VERSION,
    BONN_RGBD_SOURCE_URL,
    discover_bonn_rgbd_routes,
    missing_bonn_report,
    validate_bonn_sequence_dir,
)
from homebrain.messages.schema import JsonDict


def setup_bonn_rgbd(
    *,
    root: str | Path,
    out_dir: str | Path,
    synthetic_or_fixture: bool = False,
) -> tuple[Path, int]:
    source = Path(root)
    output = Path(out_dir)
    if not source.exists():
        return missing_bonn_report(
            output,
            source_root=source,
            reason="missing_bonn_rgbd_root",
            synthetic_or_fixture=synthetic_or_fixture,
        ), 0

    routes = discover_bonn_rgbd_routes(source)
    missing_notices: list[JsonDict] = []
    for route in routes:
        try:
            validate_bonn_sequence_dir(route)
        except FileNotFoundError as exc:
            missing_notices.append({"route": route.name, "status": "invalid", "reason": str(exc)})

    accepted = bool(routes) and not missing_notices and not synthetic_or_fixture
    reason = "ready" if accepted else "fixture_not_accepted" if synthetic_or_fixture else "missing_bonn_rgbd_routes"
    report: JsonDict = {
        "schema_version": BONN_RGBD_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "dataset_name": BONN_RGBD_DATASET_NAME,
        "source_root": source.as_posix(),
        "official_page": BONN_RGBD_SOURCE_URL,
        "license_name": BONN_RGBD_LICENSE_NAME,
        "license_review_status": BONN_RGBD_LICENSE_REVIEW_STATUS,
        "accepted": accepted,
        "reason": reason,
        "acceptance_reasons": [] if accepted else [reason],
        "synthetic_or_fixture": bool(synthetic_or_fixture),
        "route_count": len(routes),
        "routes": [route.name for route in routes],
        "missing_sensor_notices": missing_notices,
        "invented_data": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }
    path = output / "report.json"
    write_json(path, report, pretty=True)
    return path, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a local Bonn RGB-D Dynamic dataset root.")
    parser.add_argument("--root", required=True, help="Local Bonn RGB-D Dynamic dataset root.")
    parser.add_argument("--out", required=True, help="Output report directory.")
    parser.add_argument("--synthetic-or-fixture", action="store_true", help="Mark this source as fixture-only.")
    args = parser.parse_args(argv)
    report, code = setup_bonn_rgbd(
        root=args.root,
        out_dir=args.out,
        synthetic_or_fixture=args.synthetic_or_fixture,
    )
    print(f"wrote Bonn RGB-D setup report to {report.as_posix()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
