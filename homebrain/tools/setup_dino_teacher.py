from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from homebrain.teachers.dino_teacher import DINO_DEFAULT_MODEL_ID, DINO_GITHUB_URL, DINO_REPO

DINO_SETUP_STATUS_PATH = Path("external") / "dino_setup_status.json"


def setup_dino_teacher(*, model_id: str = DINO_DEFAULT_MODEL_ID, device: str | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model": {
            "model_id": model_id,
            "source": DINO_GITHUB_URL,
            "torch_hub_repo": DINO_REPO,
            "license_review_status": "pending_human_review",
            "runtime_dependency": False,
        },
        "success": False,
    }
    try:
        import torch  # type: ignore[import-not-found]

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.hub.load(DINO_REPO, model_id, trust_repo=True)
        model.eval().to(selected_device)
        status["torch"] = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "device": selected_device,
        }
        status["model"]["loaded"] = True
        status["success"] = True
    except Exception as exc:  # noqa: BLE001 - setup status should preserve exact failure.
        status["error"] = str(exc)
    return status


def write_status(status: dict[str, Any], path: str | Path = DINO_SETUP_STATUS_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(status, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicitly download/cache a DINO-family teacher through torch.hub.")
    parser.add_argument("--model-id", default=DINO_DEFAULT_MODEL_ID, help="Torch hub DINO model id, e.g. dinov2_vits14.")
    parser.add_argument("--device", default=None, help="Optional device for a load smoke test.")
    parser.add_argument("--status", default=DINO_SETUP_STATUS_PATH.as_posix(), help="Output setup status JSON path.")
    args = parser.parse_args(argv)
    status = setup_dino_teacher(model_id=args.model_id, device=args.device)
    status_path = write_status(status, args.status)
    print(f"wrote DINO setup status to {status_path.as_posix()}")
    return 0 if status.get("success") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
