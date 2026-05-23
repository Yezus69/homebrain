from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TeacherRunConfig:
    log_dir: Path
    out_dir: Path


@dataclass(frozen=True)
class TeacherRunSummary:
    teacher_name: str
    teacher_version: str
    mock: bool
    frame_count: int
    manifest_path: Path


class Teacher(ABC):
    name: str
    version: str
    mock: bool

    @abstractmethod
    def run(self, config: TeacherRunConfig) -> TeacherRunSummary:
        """Run the teacher over a segment log and write artifacts."""
