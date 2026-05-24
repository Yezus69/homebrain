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
    remote_file_size,
    tum_sequence_dir,
    validate_sequence_dir,
)


def setup_tum_rgbd(
    *,
    out_dir: str | Path,
    sequence: str,
    download: bool,
    max_download_gb: float | None = None,
    allow_large_download: bool = False,
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
        "allow_large_download": bool(allow_large_download),
        "large_download_approval_record": _large_download_approval_record(
            sequence=sequence,
            max_download_gb=max_download_gb,
            allow_large_download=allow_large_download,
        ),
        "license_name": TUM_RGBD_LICENSE_NAME,
        "license_review_status": TUM_RGBD_LICENSE_REVIEW_STATUS,
        "required_runtime": False,
        "control_safe": False,
        "status": "not_started",
    }
    try:
        if download:
            download_url = TUM_RGBD_DOWNLOADS.get(sequence)
            if download_url is not None:
                size = remote_file_size(download_url)
                if size is not None:
                    metadata["selected_archive"] = {
                        "url": download_url,
                        "size_bytes": size,
                        "size_gb": round(size / (1024.0**3), 3),
                    }
            sequence_path = download_and_extract_sequence(
                root,
                sequence,
                max_download_gb=max_download_gb,
                allow_large_download=allow_large_download,
            )
        validate_sequence_dir(sequence_path)
        metadata["sequence_dir"] = sequence_path.as_posix()
        metadata["status"] = "ready"
        metadata["files"] = {
            "rgb": (sequence_path / "rgb.txt").as_posix(),
            "depth": (sequence_path / "depth.txt").as_posix(),
            "groundtruth": (sequence_path / "groundtruth.txt").as_posix(),
            "accelerometer": (sequence_path / "accelerometer.txt").as_posix()
            if (sequence_path / "accelerometer.txt").exists()
            else None,
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


def _large_download_approval_record(
    *,
    sequence: str,
    max_download_gb: float | None,
    allow_large_download: bool,
) -> dict[str, Any] | None:
    if not allow_large_download:
        return None
    return {
        "approved": True,
        "approval_method": "cli_flag_--allow-large-download",
        "approved_at_utc": _utc_now(),
        "sequence": sequence,
        "bounded_by_max_download_gb": max_download_gb,
        "note": (
            "Operator explicitly approved this bounded public dataset download/import. "
            "This does not imply license approval or control safety."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/validate a public TUM RGB-D sequence.")
    parser.add_argument("--out", required=True, help="Output dataset root, e.g. data/public/tum_rgbd.")
    parser.add_argument("--sequence", required=True, help="TUM sequence name, e.g. freiburg1_xyz.")
    parser.add_argument("--download", action="store_true", help="Download and extract the sequence archive.")
    parser.add_argument("--max-download-gb", type=float, default=None, help="Refuse download above this size.")
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Record explicit approval for a bounded multi-GB public dataset download.",
    )
    args = parser.parse_args(argv)
    metadata_path, return_code = setup_tum_rgbd(
        out_dir=args.out,
        sequence=args.sequence,
        download=args.download,
        max_download_gb=args.max_download_gb,
        allow_large_download=args.allow_large_download,
    )
    print(f"wrote TUM RGB-D setup metadata to {metadata_path.as_posix()}")
    if args.allow_large_download:
        print(
            "recorded large-download approval "
            f"for sequence={args.sequence} max_download_gb={args.max_download_gb}"
        )
    if return_code != 0:
        print("TUM RGB-D setup blocked; see dataset_metadata.json for details.", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
