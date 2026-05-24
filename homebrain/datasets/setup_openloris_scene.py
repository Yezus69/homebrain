from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from homebrain.datasets.openloris_scene import (
    DEFAULT_MAX_DOWNLOAD_GB,
    OPENLORIS_DOWNLOAD_PAGE,
    OPENLORIS_HF_PACKAGE_URL,
    OPENLORIS_LICENSE_NAME,
    OPENLORIS_LICENSE_REVIEW_STATUS,
    OPENLORIS_SCHEMA_VERSION,
    OPENLORIS_SOURCE_URL,
    OPENLORIS_TOOLS_COMMIT_INSPECTED,
    OPENLORIS_TOOLS_URL,
    choose_remote_package,
    discover_sequence_root,
    download_remote_package,
    extract_package,
    list_remote_package_files,
    local_disk_free_gb,
    openloris_sequence_dir,
    validate_sequence_dir,
    write_json,
)


def setup_openloris_scene(
    *,
    out_dir: str | Path,
    sequence: str,
    download: bool,
    package_file: str | Path | None = None,
    max_download_gb: float = DEFAULT_MAX_DOWNLOAD_GB,
    allow_large_download: bool = False,
) -> tuple[Path, int]:
    root = Path(out_dir)
    sequence_path = openloris_sequence_dir(root, sequence)
    metadata_path = sequence_path / "dataset_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": OPENLORIS_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "dataset_name": "OpenLORIS-Scene",
        "sequence": sequence,
        "sequence_dir": sequence_path.as_posix(),
        "official_page": OPENLORIS_SOURCE_URL,
        "official_download_page": OPENLORIS_DOWNLOAD_PAGE,
        "huggingface_package_url": OPENLORIS_HF_PACKAGE_URL,
        "tools_url": OPENLORIS_TOOLS_URL,
        "tools_commit_inspected": OPENLORIS_TOOLS_COMMIT_INSPECTED,
        "download_requested": download,
        "package_file": Path(package_file).as_posix() if package_file is not None else None,
        "max_download_gb": float(max_download_gb),
        "allow_large_download": bool(allow_large_download),
        "license_name": OPENLORIS_LICENSE_NAME,
        "license_review_status": OPENLORIS_LICENSE_REVIEW_STATUS,
        "required_runtime": False,
        "control_safe": False,
        "status": "not_started",
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        if package_file is not None:
            sequence_path = extract_package(package_file, root, sequence)
        elif sequence_path.exists():
            sequence_path = discover_sequence_root(sequence_path)
        elif download:
            remote_files = list_remote_package_files()
            metadata["remote_package_files"] = [
                {
                    "path": remote.path,
                    "size_bytes": remote.size_bytes,
                    "size_gb": round(remote.size_gb, 3),
                }
                for remote in remote_files
            ]
            selected = choose_remote_package(sequence, remote_files)
            if selected is None:
                raise RuntimeError("could not list OpenLORIS package files from the official Hugging Face mirror")
            metadata["selected_remote_package"] = {
                "path": selected.path,
                "size_bytes": selected.size_bytes,
                "size_gb": round(selected.size_gb, 3),
            }
            free_gb = local_disk_free_gb(root)
            metadata["local_free_disk_gb"] = round(free_gb, 3)
            if selected.size_gb > max_download_gb and not allow_large_download:
                raise RuntimeError(
                    f"selected OpenLORIS package {selected.path} is {selected.size_gb:.2f} GB, "
                    f"exceeding max_download_gb={max_download_gb:.2f}; rerun with --allow-large-download "
                    "or stage/extract a package manually and call openloris_to_route"
                )
            if selected.size_gb * 1.25 > free_gb:
                raise RuntimeError(
                    f"selected OpenLORIS package {selected.path} needs about {selected.size_gb * 1.25:.2f} GB "
                    f"free for download/extract, but only {free_gb:.2f} GB is available"
                )
            archive_path = download_remote_package(selected, root)
            sequence_path = extract_package(archive_path, root, sequence)
        else:
            raise FileNotFoundError(
                f"OpenLORIS sequence not staged at {sequence_path}; pass --download, --package-file, "
                "or place an extracted package directory there"
            )

        validate_sequence_dir(sequence_path)
        metadata["sequence_dir"] = sequence_path.as_posix()
        metadata["status"] = "ready"
        metadata["files"] = {
            "color": (sequence_path / "color.txt").as_posix(),
            "aligned_depth": (sequence_path / "aligned_depth.txt").as_posix()
            if (sequence_path / "aligned_depth.txt").exists()
            else None,
            "depth": (sequence_path / "depth.txt").as_posix() if (sequence_path / "depth.txt").exists() else None,
            "odom": (sequence_path / "odom.txt").as_posix() if (sequence_path / "odom.txt").exists() else None,
            "groundtruth": (sequence_path / "groundtruth.txt").as_posix()
            if (sequence_path / "groundtruth.txt").exists()
            else None,
            "sensors": (sequence_path / "sensors.yaml").as_posix() if (sequence_path / "sensors.yaml").exists() else None,
        }
        return_code = 0
    except Exception as exc:  # noqa: BLE001 - setup must always leave evidence.
        metadata["status"] = "blocked"
        metadata["error"] = str(exc)
        metadata["blocker_summary"] = str(exc)
        return_code = 2
    finally:
        write_json(metadata_path, metadata, pretty=True)
        write_json(root / "openloris_scene_setup_status.json", metadata, pretty=True)
    return metadata_path, return_code


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/validate a public OpenLORIS-Scene package.")
    parser.add_argument("--out", required=True, help="Output dataset root, e.g. data/public/openloris_scene.")
    parser.add_argument("--sequence", default="cafe1-1", help="OpenLORIS sequence/package stem.")
    parser.add_argument("--download", action="store_true", help="Try to download the selected package.")
    parser.add_argument("--package-file", default=None, help="Already downloaded .tar/.tgz/.zip package to extract.")
    parser.add_argument("--max-download-gb", type=float, default=DEFAULT_MAX_DOWNLOAD_GB)
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Allow multi-GB package downloads when disk space is sufficient.",
    )
    args = parser.parse_args(argv)
    metadata_path, return_code = setup_openloris_scene(
        out_dir=args.out,
        sequence=args.sequence,
        download=args.download,
        package_file=args.package_file,
        max_download_gb=args.max_download_gb,
        allow_large_download=args.allow_large_download,
    )
    print(f"wrote OpenLORIS setup metadata to {metadata_path.as_posix()}")
    if return_code != 0:
        print("OpenLORIS setup blocked; see dataset_metadata.json for details.", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
