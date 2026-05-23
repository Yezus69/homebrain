from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from homebrain.teachers.base import TeacherRunConfig
from homebrain.teachers.da3_teacher import DA3_DEFAULT_MODEL_ID, DA3UnavailableError
from homebrain.teachers.depth_pro_teacher import DepthProUnavailableError
from homebrain.teachers.registry import TEACHER_NAMES, create_teacher


def main(argv: list[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run an offline HomeBrain teacher over a segment log.")
    parser.add_argument("--teacher", required=True, choices=TEACHER_NAMES, help="Teacher implementation to run.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output teacher artifact directory.")
    parser.add_argument(
        "--backend",
        default="real",
        choices=["real", "fake"],
        help="Teacher backend. Use 'fake' only for tests; real never intentionally downloads weights.",
    )
    parser.add_argument("--device", default=None, help="Optional teacher device, e.g. cpu or cuda.")
    parser.add_argument("--model-id", default=DA3_DEFAULT_MODEL_ID, help="DA3 model id for manifests/setup lookup.")
    parser.add_argument("--model-dir", default=None, help="Local DA3 model directory or snapshot path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional DA3 frame cap.")
    parser.add_argument("--window-size", type=int, default=None, help="Optional DA3 batch/window size.")
    parser.add_argument("--stride", type=int, default=1, help="Optional DA3 frame stride.")
    args = parser.parse_args(argv_list)

    if _should_reexec_da3(args):
        return _reexec_with_da3_python(argv_list)

    try:
        teacher = create_teacher(
            args.teacher,
            backend_name=args.backend,
            device=args.device,
            model_id=args.model_id,
            model_dir=args.model_dir,
            max_frames=args.max_frames,
            window_size=args.window_size,
            stride=args.stride,
        )
        summary = teacher.run(TeacherRunConfig(log_dir=args.log, out_dir=args.out))
    except DepthProUnavailableError as exc:
        print(f"Depth Pro teacher unavailable: {exc}", file=sys.stderr)
        return 2
    except DA3UnavailableError as exc:
        print(f"DA3 teacher unavailable: {exc}", file=sys.stderr)
        return 2

    backend_note = f" with {args.backend} backend" if args.teacher in {"depth_pro", "da3"} else ""
    print(f"wrote {summary.frame_count} {args.teacher} teacher frames{backend_note} to {summary.manifest_path.as_posix()}")
    return 0


def _should_reexec_da3(args: argparse.Namespace) -> bool:
    if args.teacher != "da3" or args.backend != "real":
        return False
    if os.environ.get("HOMEBRAIN_DA3_REEXECED") == "1":
        return False
    try:
        da3_api_available = importlib.util.find_spec("depth_anything_3.api") is not None
    except ModuleNotFoundError:
        da3_api_available = False
    if da3_api_available:
        return False
    return _external_da3_python() is not None


def _reexec_with_da3_python(argv_list: list[str]) -> int:
    python_path = _external_da3_python()
    if python_path is None:
        return 2
    repo_root = _repo_root()
    env = os.environ.copy()
    env["HOMEBRAIN_DA3_REEXECED"] = "1"
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not pythonpath else f"{repo_root}{os.pathsep}{pythonpath}"
    completed = subprocess.run(
        [str(python_path), "-m", "homebrain.teachers.run_teacher", *argv_list],
        cwd=repo_root,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _external_da3_python() -> Path | None:
    env_python = os.environ.get("HOMEBRAIN_DA3_PYTHON")
    if env_python and Path(env_python).exists():
        return Path(env_python)

    status_path = _repo_root() / "external" / "da3_setup_status.json"
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            python_value = data.get("python", {}).get("path") if isinstance(data.get("python"), dict) else None
        except json.JSONDecodeError:
            python_value = None
        if isinstance(python_value, str) and Path(python_value).exists():
            return Path(python_value)

    for candidate in (
        _repo_root() / "external" / "venvs" / "da3" / "Scripts" / "python.exe",
        _repo_root() / "external" / "venvs" / "da3" / "bin" / "python",
    ):
        if candidate.exists():
            return candidate
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
