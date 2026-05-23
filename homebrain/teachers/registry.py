from __future__ import annotations

from homebrain.teachers.base import Teacher
from homebrain.teachers.depth_pro_teacher import create_depth_pro_teacher
from homebrain.teachers.mock_teacher import MockTeacher

TEACHER_NAMES: tuple[str, ...] = ("mock", "depth_pro")


def create_teacher(
    name: str,
    *,
    backend_name: str = "real",
    device: str | None = None,
) -> Teacher:
    if name == "mock":
        return MockTeacher()
    if name == "depth_pro":
        return create_depth_pro_teacher(backend_name=backend_name, device=device)
    raise ValueError(f"unknown teacher {name!r}; expected one of {', '.join(TEACHER_NAMES)}")
