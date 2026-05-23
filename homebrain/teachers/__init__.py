"""Offline teacher artifact tools for HomeBrain."""

from homebrain.teachers.artifacts import (
    DEPTH_PRO_ARTIFACT_KINDS,
    EXPECTED_ARTIFACT_KINDS,
    TEACHER_MANIFEST_FILE,
    TeacherArtifactValidation,
    load_teacher_manifest,
    validate_teacher_artifacts,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig, TeacherRunSummary
from homebrain.teachers.depth_pro_teacher import DepthProTeacher, FakeDepthProBackend, RealDepthProBackend
from homebrain.teachers.mock_teacher import MockTeacher
from homebrain.teachers.registry import TEACHER_NAMES, create_teacher

__all__ = [
    "DEPTH_PRO_ARTIFACT_KINDS",
    "EXPECTED_ARTIFACT_KINDS",
    "TEACHER_MANIFEST_FILE",
    "DepthProTeacher",
    "FakeDepthProBackend",
    "MockTeacher",
    "RealDepthProBackend",
    "TEACHER_NAMES",
    "Teacher",
    "TeacherArtifactValidation",
    "TeacherRunConfig",
    "TeacherRunSummary",
    "create_teacher",
    "load_teacher_manifest",
    "validate_teacher_artifacts",
    "write_teacher_manifest",
]
