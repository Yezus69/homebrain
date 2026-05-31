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
from homebrain.teachers.dino_teacher import DINO_DEFAULT_MODEL_ID, DINOUnavailableError
from homebrain.teachers.hazard_teacher import HAZARD_DEFAULT_MODEL_ID, HazardTeacherUnavailableError
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
    parser.add_argument("--model-id", default=None, help="Teacher model id for manifests/setup lookup.")
    parser.add_argument("--model-dir", default=None, help="Optional local teacher model directory or snapshot path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional teacher frame cap.")
    parser.add_argument("--window-size", type=int, default=None, help="Optional DA3 batch/window size.")
    parser.add_argument("--stride", type=int, default=1, help="Optional teacher frame stride.")
    parser.add_argument("--image-size", type=int, default=224, help="DINO input image size; must be a multiple of 14.")
    parser.add_argument("--prompts", default=None, help="Hazard teacher prompt/ontology config path.")
    args = parser.parse_args(argv_list)

    if _should_reexec_da3(args):
        return _reexec_with_da3_python(argv_list)
    if _should_reexec_hazard(args):
        return _reexec_with_external_python(argv_list, _external_hazard_python(), env_flag="HOMEBRAIN_HAZARD_REEXECED")

    try:
        teacher = create_teacher(
            args.teacher,
            backend_name=args.backend,
            device=args.device,
            model_id=_default_model_id(args.teacher, args.model_id),
            model_dir=args.model_dir,
            max_frames=args.max_frames,
            window_size=args.window_size,
            stride=args.stride,
            image_size=args.image_size,
            prompts_path=args.prompts,
        )
        summary = teacher.run(TeacherRunConfig(log_dir=args.log, out_dir=args.out))
    except DepthProUnavailableError as exc:
        print(f"Depth Pro teacher unavailable: {exc}", file=sys.stderr)
        return 2
    except DA3UnavailableError as exc:
        print(f"DA3 teacher unavailable: {exc}", file=sys.stderr)
        return 2
    except DINOUnavailableError as exc:
        print(f"DINO teacher unavailable: {exc}", file=sys.stderr)
        return 2
    except HazardTeacherUnavailableError as exc:
        print(f"Hazard teacher unavailable: {exc}", file=sys.stderr)
        return 2

    backend_note = f" with {args.backend} backend" if args.teacher in {"depth_pro", "da3", "dino", "hazard"} else ""
    print(f"wrote {summary.frame_count} {args.teacher} teacher frames{backend_note} to {summary.manifest_path.as_posix()}")
    return 0


def _default_model_id(teacher: str, model_id: str | None) -> str:
    if model_id:
        return model_id
    if teacher == "dino":
        return DINO_DEFAULT_MODEL_ID
    if teacher == "hazard":
        return HAZARD_DEFAULT_MODEL_ID
    return DA3_DEFAULT_MODEL_ID


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


def _should_reexec_hazard(args: argparse.Namespace) -> bool:
    if args.teacher != "hazard" or args.backend != "real":
        return False
    if os.environ.get("HOMEBRAIN_HAZARD_REEXECED") == "1":
        return False
    try:
        groundingdino_available = importlib.util.find_spec("groundingdino.util.inference") is not None
    except ModuleNotFoundError:
        groundingdino_available = False
    if groundingdino_available:
        return False
    return _external_hazard_python() is not None


def _reexec_with_da3_python(argv_list: list[str]) -> int:
    return _reexec_with_external_python(argv_list, _external_da3_python(), env_flag="HOMEBRAIN_DA3_REEXECED")


def _reexec_with_external_python(argv_list: list[str], python_path: Path | None, *, env_flag: str) -> int:
    if python_path is None:
        return 2
    repo_root = _repo_root()
    env = os.environ.copy()
    env[env_flag] = "1"
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


def _external_hazard_python() -> Path | None:
    env_python = os.environ.get("HOMEBRAIN_HAZARD_PYTHON")
    if env_python and Path(env_python).exists() and _hazard_python_has_groundingdino(Path(env_python)):
        return Path(env_python)

    status_path = _repo_root() / "external" / "hazard_teacher_setup_status.json"
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            python_value = data.get("python", {}).get("path") if isinstance(data.get("python"), dict) else None
        except json.JSONDecodeError:
            python_value = None
        if isinstance(python_value, str) and Path(python_value).exists() and _hazard_python_has_groundingdino(Path(python_value)):
            return Path(python_value)

    for candidate in (
        _repo_root() / "external" / "venvs" / "hazard" / "Scripts" / "python.exe",
        _repo_root() / "external" / "venvs" / "hazard" / "bin" / "python",
    ):
        if candidate.exists() and _hazard_python_has_groundingdino(candidate):
            return candidate
    return None


def _hazard_python_has_groundingdino(python_path: Path) -> bool:
    code = (
        "import importlib.util, sys\n"
        "try:\n"
        "    ok = importlib.util.find_spec('groundingdino.util.inference') is not None\n"
        "except ModuleNotFoundError:\n"
        "    ok = False\n"
        "sys.exit(0 if ok else 1)\n"
    )
    try:
        completed = subprocess.run(
            [str(python_path), "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - a bad external interpreter is simply not usable.
        return False
    return completed.returncode == 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
