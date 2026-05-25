"""Offline teacher artifact tools for HomeBrain."""

from homebrain.teachers.artifacts import (
    DA3_ARTIFACT_KINDS,
    DEPTH_PRO_ARTIFACT_KINDS,
    DINO_ARTIFACT_KINDS,
    EXPECTED_ARTIFACT_KINDS,
    TEACHER_MANIFEST_FILE,
    TeacherArtifactValidation,
    load_teacher_manifest,
    validate_teacher_artifacts,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig, TeacherRunSummary
_LAZY_EXPORTS = {
    "SCENE_TEACHER_FRAME_ARTIFACT_KINDS": (
        "homebrain.teachers.scene_teacher",
        "SCENE_TEACHER_FRAME_ARTIFACT_KINDS",
    ),
    "SCENE_TEACHER_MANIFEST_FILE": ("homebrain.teachers.scene_teacher", "SCENE_TEACHER_MANIFEST_FILE"),
    "SCENE_TEACHER_PACK_SCHEMA_VERSION": (
        "homebrain.teachers.scene_teacher",
        "SCENE_TEACHER_PACK_SCHEMA_VERSION",
    ),
    "SCENE_TEACHER_WINDOW_ARTIFACT_KINDS": (
        "homebrain.teachers.scene_teacher",
        "SCENE_TEACHER_WINDOW_ARTIFACT_KINDS",
    ),
    "DA3Teacher": ("homebrain.teachers.da3_teacher", "DA3Teacher"),
    "DepthProTeacher": ("homebrain.teachers.depth_pro_teacher", "DepthProTeacher"),
    "DINOTeacher": ("homebrain.teachers.dino_teacher", "DINOTeacher"),
    "FakeDA3Backend": ("homebrain.teachers.da3_teacher", "FakeDA3Backend"),
    "FakeDepthProBackend": ("homebrain.teachers.depth_pro_teacher", "FakeDepthProBackend"),
    "FakeDINOBackend": ("homebrain.teachers.dino_teacher", "FakeDINOBackend"),
    "MockTeacher": ("homebrain.teachers.mock_teacher", "MockTeacher"),
    "RealDA3Backend": ("homebrain.teachers.da3_teacher", "RealDA3Backend"),
    "RealDepthProBackend": ("homebrain.teachers.depth_pro_teacher", "RealDepthProBackend"),
    "RealDINOBackend": ("homebrain.teachers.dino_teacher", "RealDINOBackend"),
    "FakeVGGTSceneBackend": ("homebrain.teachers.vggt_scene_teacher", "FakeVGGTSceneBackend"),
    "FakeMoGeSceneBackend": ("homebrain.teachers.moge_scene_teacher", "FakeMoGeSceneBackend"),
    "RealVGGTSceneBackend": ("homebrain.teachers.vggt_scene_teacher", "RealVGGTSceneBackend"),
    "RealMoGeSceneBackend": ("homebrain.teachers.moge_scene_teacher", "RealMoGeSceneBackend"),
    "VGGTSceneTeacher": ("homebrain.teachers.vggt_scene_teacher", "VGGTSceneTeacher"),
    "MoGeSceneTeacher": ("homebrain.teachers.moge_scene_teacher", "MoGeSceneTeacher"),
    "TEACHER_NAMES": ("homebrain.teachers.registry", "TEACHER_NAMES"),
    "create_teacher": ("homebrain.teachers.registry", "create_teacher"),
    "create_vggt_scene_teacher": ("homebrain.teachers.vggt_scene_teacher", "create_vggt_scene_teacher"),
    "create_moge_scene_teacher": ("homebrain.teachers.moge_scene_teacher", "create_moge_scene_teacher"),
    "load_scene_teacher_manifest": ("homebrain.teachers.scene_teacher", "load_scene_teacher_manifest"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "DEPTH_PRO_ARTIFACT_KINDS",
    "DA3_ARTIFACT_KINDS",
    "DA3Teacher",
    "DINO_ARTIFACT_KINDS",
    "DINOTeacher",
    "EXPECTED_ARTIFACT_KINDS",
    "TEACHER_MANIFEST_FILE",
    "DepthProTeacher",
    "FakeDA3Backend",
    "FakeDepthProBackend",
    "FakeDINOBackend",
    "MockTeacher",
    "RealDA3Backend",
    "RealDepthProBackend",
    "RealDINOBackend",
    "FakeVGGTSceneBackend",
    "FakeMoGeSceneBackend",
    "RealVGGTSceneBackend",
    "RealMoGeSceneBackend",
    "VGGTSceneTeacher",
    "MoGeSceneTeacher",
    "SCENE_TEACHER_FRAME_ARTIFACT_KINDS",
    "SCENE_TEACHER_MANIFEST_FILE",
    "SCENE_TEACHER_PACK_SCHEMA_VERSION",
    "SCENE_TEACHER_WINDOW_ARTIFACT_KINDS",
    "TEACHER_NAMES",
    "Teacher",
    "TeacherArtifactValidation",
    "TeacherRunConfig",
    "TeacherRunSummary",
    "create_teacher",
    "create_moge_scene_teacher",
    "create_vggt_scene_teacher",
    "load_scene_teacher_manifest",
    "load_teacher_manifest",
    "validate_teacher_artifacts",
    "write_teacher_manifest",
]
