from __future__ import annotations

from homebrain.teachers.base import Teacher
from homebrain.teachers.da3_teacher import create_da3_teacher
from homebrain.teachers.depth_pro_teacher import create_depth_pro_teacher
from homebrain.teachers.dino_teacher import DINO_DEFAULT_MODEL_ID, create_dino_teacher
from homebrain.teachers.mock_teacher import MockTeacher

TEACHER_NAMES: tuple[str, ...] = ("mock", "depth_pro", "da3", "dino")


def create_teacher(
    name: str,
    *,
    backend_name: str = "real",
    device: str | None = None,
    model_id: str | None = None,
    model_dir: str | None = None,
    max_frames: int | None = None,
    window_size: int | None = None,
    stride: int = 1,
    image_size: int = 224,
) -> Teacher:
    if name == "mock":
        return MockTeacher()
    if name == "depth_pro":
        return create_depth_pro_teacher(backend_name=backend_name, device=device)
    if name == "da3":
        return create_da3_teacher(
            backend_name=backend_name,
            device=device,
            model_id=model_id or "depth-anything/DA3-SMALL",
            model_dir=model_dir,
            max_frames=max_frames,
            window_size=window_size,
            stride=stride,
        )
    if name == "dino":
        return create_dino_teacher(
            backend_name=backend_name,
            device=device,
            model_id=model_id or DINO_DEFAULT_MODEL_ID,
            model_dir=model_dir,
            max_frames=max_frames,
            stride=stride,
            image_size=image_size,
        )
    raise ValueError(f"unknown teacher {name!r}; expected one of {', '.join(TEACHER_NAMES)}")
