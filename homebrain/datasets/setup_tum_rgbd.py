from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from homebrain.datasets.tum_rgbd import (
    TUM_RGBD_DOWNLOADS,
    TUM_RGBD_LICENSE_NAME,
    TUM_RGBD_LICENSE_REVIEW_STATUS,
    TUM_RGBD_SCHEMA_VERSION,
    TUM_RGBD_SOURCE_URL,
    download_and_extract_sequence,
    tum_sequence_dir,
    validate_sequence_dir,
)


def setup_tum_rgbd(
    *,
    out_dir: str | Path,
    sequence: str,
    download: bool,
    max_download_gb: float | None = None,
) -> tuple[Path, int]:
    root = Path(out_dir)
    sequence_path = tum_sequence_dir(root, sequence)
    metadata_path = sequence_path / "dataset_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": TUM_RGBD_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "dataset_name": "TUM RGB-D SLAM Dataset and Benchmark",
        "sequence": sequence,
        "sequence_dir": sequence_path.as_posix(),
        "official_page": TUM_RGBD_SOURCE_URL,
        "download_url": TUM_RGBD_DOWNLOADS.get(sequence),
        "download_requested": download,
        "max_download_gb": max_download_gb,
        "license_name": TUM_RGBD_LICENSE_NAME,
        "license_review_status": TUM_RGBD_LICENSE_REVIEW_STATUS,
        "required_runtime": False,
        "control_safe": False,
        "status": "not_started",
    }
    try:
        if download:
            sequence_path = download_and_extract_sequence(root, sequence, max_download_gb=max_download_gb)
        validate_sequence_dir(sequence_path)
        metadata["sequence_dir"] = sequence_path.as_posix()
        metadata["status"] = "ready"
        metadata["files"] = {
            "rgb": (sequence_path / "rgb.txt").as_posix(),
            "depth": (sequence_path / "depth.txt").as_posix(),
            "groundtruth": (sequence_path / "groundtruth.txt").as_posix(),
            "associations": (sequence_path / "associations.txt").as_posix()
            if (sequence_path / "associations.txt").exists()
            else None,
        }
        return_code = 0
    except Exception as exc:  # noqa: BLE001 - setup must leave evidence.
        metadata["status"] = "blocked"
        metadata["error"] = str(exc)
        return_code = 2
    finally:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, sort_keys=True, indent=2)
            handle.write("\n")
    return metadata_path, return_code


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/validate a public TUM RGB-D sequence.")
    parser.add_argument("--out", required=True, help="Output dataset root, e.g. data/public/tum_rgbd.")
    parser.add_argument("--sequence", required=True, help="TUM sequence name, e.g. freiburg1_xyz.")
    parser.add_argument("--download", action="store_true", help="Download and extract the sequence archive.")
    parser.add_argument("--max-download-gb", type=float, default=None, help="Refuse download above this size.")
    args = parser.parse_args(argv)
    metadata_path, return_code = setup_tum_rgbd(
        out_dir=args.out,
        sequence=args.sequence,
        download=args.download,
        max_download_gb=args.max_download_gb,
    )
    print(f"wrote TUM RGB-D setup metadata to {metadata_path.as_posix()}")
    if return_code != 0:
        print("TUM RGB-D setup blocked; see dataset_metadata.json for details.", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
