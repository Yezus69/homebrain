from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from homebrain.teachers.da3_teacher import DA3_DEFAULT_MODEL_ID, DA3_GITHUB_URL

STATUS_SCHEMA_VERSION = "homebrain.da3_setup_status.v0"
STATUS_FILE = "da3_setup_status.json"


def setup_da3_teacher(
    *,
    external_dir: str | Path,
    venv_dir: str | Path,
    model_id: str = DA3_DEFAULT_MODEL_ID,
    download: bool = False,
) -> tuple[Path, int]:
    repo_dir = Path(external_dir)
    venv = Path(venv_dir)
    external_root = repo_dir.parent
    status_path = external_root / STATUS_FILE
    external_root.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "repo": {
            "url": DA3_GITHUB_URL,
            "path": repo_dir.as_posix(),
            "present": repo_dir.exists(),
            "commit": None,
            "status": "not_checked",
        },
        "python": {
            "venv": venv.as_posix(),
            "path": _venv_python(venv).as_posix(),
            "created": False,
            "status": "not_checked",
        },
        "model": {
            "model_id": model_id,
            "local_dir": _model_local_dir(external_root, model_id).as_posix(),
            "download_requested": download,
            "download_status": "not_requested" if not download else "not_started",
        },
        "dependency_status": {
            "commands": [],
            "depth_anything_3_import": "not_checked",
            "torch_import": "not_checked",
        },
        "cuda_status": {
            "checked": False,
            "torch_cuda_available": False,
            "device_count": 0,
            "devices": [],
        },
        "success": False,
    }

    try:
        _ensure_repo(repo_dir, status)
        _ensure_venv(venv, status)
        python = _venv_python(venv)
        _install_dependencies(python, repo_dir, status)
        _check_imports(python, status)
        if download:
            _download_model(python, model_id, _model_local_dir(external_root, model_id), status)
        _check_cuda(python, status)
        status["repo"]["commit"] = _repo_commit(repo_dir)
        status["repo"]["present"] = repo_dir.exists()
        status["python"]["path"] = python.as_posix()
        status["success"] = _setup_success(status)
    except Exception as exc:  # noqa: BLE001 - setup must always write status.
        status["success"] = False
        status["error"] = str(exc)
    finally:
        _write_status(status_path, status)

    return status_path, 0 if status["success"] else 2


def _ensure_repo(repo_dir: Path, status: dict[str, Any]) -> None:
    if repo_dir.exists():
        if not (repo_dir / ".git").exists():
            status["repo"]["status"] = "error_existing_path_not_git_repo"
            raise RuntimeError(f"external-dir exists but is not a git repo: {repo_dir}")
        status["repo"]["status"] = "existing"
        status["repo"]["commit"] = _repo_commit(repo_dir)
        return
    result = _run(["git", "clone", "--recurse-submodules", DA3_GITHUB_URL, str(repo_dir)])
    status["repo"]["clone_command"] = result
    if result["returncode"] != 0:
        status["repo"]["status"] = "clone_failed"
        raise RuntimeError(f"failed to clone DA3 repo into {repo_dir}")
    status["repo"]["status"] = "cloned"
    status["repo"]["commit"] = _repo_commit(repo_dir)


def _ensure_venv(venv: Path, status: dict[str, Any]) -> None:
    python = _venv_python(venv)
    if python.exists():
        status["python"]["status"] = "existing"
        return
    result = _run([sys.executable, "-m", "venv", str(venv)])
    status["python"]["create_command"] = result
    if result["returncode"] != 0 or not python.exists():
        status["python"]["status"] = "create_failed"
        raise RuntimeError(f"failed to create DA3 venv at {venv}")
    status["python"]["created"] = True
    status["python"]["status"] = "created"


def _install_dependencies(python: Path, repo_dir: Path, status: dict[str, Any]) -> None:
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [str(python), "-m", "pip", "install", "torch", "torchvision"],
        [str(python), "-m", "pip", "install", "huggingface_hub"],
        [str(python), "-m", "pip", "install", "-e", str(repo_dir)],
        [str(python), "-m", "pip", "install", "addict"],
    ]
    for command in commands:
        result = _run(command)
        status["dependency_status"]["commands"].append(result)
        if result["returncode"] != 0:
            raise RuntimeError(f"dependency install command failed: {' '.join(command)}")


def _download_model(python: Path, model_id: str, local_dir: Path, status: dict[str, Any]) -> None:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import json\n"
        "from huggingface_hub import snapshot_download\n"
        f"path=snapshot_download(repo_id={model_id!r}, local_dir={str(local_dir)!r})\n"
        "print(json.dumps({'local_dir': path}, sort_keys=True))\n"
    )
    result = _run([str(python), "-c", code], timeout_seconds=3600)
    status["model"]["download_command"] = result
    if result["returncode"] != 0:
        status["model"]["download_status"] = "failed"
        raise RuntimeError(f"failed to download/cache DA3 model {model_id}")
    status["model"]["download_status"] = "downloaded_or_cached"
    status["model"]["local_dir"] = local_dir.as_posix()


def _check_imports(python: Path, status: dict[str, Any]) -> None:
    code = (
        "import json\n"
        "result={}\n"
        "try:\n"
        "    from depth_anything_3.api import DepthAnything3\n"
        "    result['depth_anything_3_import']='ok'\n"
        "    result['depth_anything_3_api']='ok'\n"
        "    result['depth_anything_3_class']=DepthAnything3.__name__\n"
        "except Exception as exc:\n"
        "    result['depth_anything_3_import']='failed'\n"
        "    result['depth_anything_3_error']=str(exc)\n"
        "try:\n"
        "    import torch\n"
        "    result['torch_import']='ok'\n"
        "    result['torch_version']=getattr(torch,'__version__',None)\n"
        "except Exception as exc:\n"
        "    result['torch_import']='failed'\n"
        "    result['torch_error']=str(exc)\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )
    result = _run([str(python), "-c", code])
    parsed = _parse_json_stdout(result)
    if parsed:
        status["dependency_status"].update(parsed)
    if result["returncode"] != 0:
        status["dependency_status"]["import_check_command"] = result
    if status["dependency_status"].get("depth_anything_3_import") != "ok":
        raise RuntimeError("depth_anything_3 import check failed after installation")
    if status["dependency_status"].get("torch_import") != "ok":
        raise RuntimeError("torch import check failed after installation")


def _check_cuda(python: Path, status: dict[str, Any]) -> None:
    code = (
        "import json\n"
        "out={'checked': True, 'torch_cuda_available': False, 'device_count': 0, 'devices': []}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch_cuda_available']=bool(torch.cuda.is_available())\n"
        "    out['device_count']=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0\n"
        "    out['devices']=[torch.cuda.get_device_name(i) for i in range(out['device_count'])]\n"
        "except Exception as exc:\n"
        "    out['error']=str(exc)\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    result = _run([str(python), "-c", code])
    parsed = _parse_json_stdout(result)
    if parsed:
        status["cuda_status"] = parsed
    else:
        status["cuda_status"]["error"] = result.get("stderr_tail") or result.get("stdout_tail")


def _setup_success(status: dict[str, Any]) -> bool:
    dependency = status.get("dependency_status", {})
    model = status.get("model", {})
    if dependency.get("depth_anything_3_import") != "ok":
        return False
    if dependency.get("torch_import") != "ok":
        return False
    if model.get("download_requested") and model.get("download_status") != "downloaded_or_cached":
        return False
    return True


def _run(command: list[str], *, timeout_seconds: int = 1800) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "timeout_seconds": timeout_seconds,
        }


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _parse_json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    stdout = result.get("stdout_tail")
    if not isinstance(stdout, str) or not stdout.strip():
        return {}
    last_line = stdout.strip().splitlines()[-1]
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _repo_commit(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    result = _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], timeout_seconds=30)
    if result["returncode"] != 0:
        return None
    text = str(result.get("stdout_tail", "")).strip()
    return text.splitlines()[-1] if text else None


def _model_local_dir(external_root: Path, model_id: str) -> Path:
    safe_name = model_id.replace("/", "__").replace(":", "_")
    return external_root / "models" / "da3" / safe_name


def _venv_python(venv: Path) -> Path:
    windows_python = venv / "Scripts" / "python.exe"
    if windows_python.exists() or sys.platform.startswith("win"):
        return windows_python
    return venv / "bin" / "python"


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(status, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clone/install/download the Depth Anything 3 offline teacher.")
    parser.add_argument("--external-dir", required=True, help="DA3 git clone directory under ignored external/.")
    parser.add_argument("--venv", required=True, help="Isolated DA3 virtual environment path.")
    parser.add_argument("--model-id", default=DA3_DEFAULT_MODEL_ID, help="Hugging Face DA3 model id.")
    parser.add_argument("--download", action="store_true", help="Download/cache model weights into external/.")
    args = parser.parse_args(argv)

    status_path, returncode = setup_da3_teacher(
        external_dir=args.external_dir,
        venv_dir=args.venv,
        model_id=args.model_id,
        download=args.download,
    )
    print(f"wrote DA3 setup status to {status_path.as_posix()}")
    if returncode != 0:
        print("DA3 setup did not complete successfully; see status JSON for details.", file=sys.stderr)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
