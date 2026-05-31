import json
from pathlib import Path
import argparse

import numpy as np

from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.teachers.artifacts import load_array, load_teacher_manifest, validate_teacher_artifacts
from homebrain.teachers.hazard_backends import _rasterize_grounding_dino_boxes
from homebrain.teachers.hazard_config import load_hazard_prompt_config
from homebrain.teachers.hazard_teacher import run_hazard_teacher
from homebrain.teachers import run_teacher


def test_fake_hazard_teacher_is_quarantined_and_deterministic(tmp_path: Path) -> None:
    route = tmp_path / "route"
    out = tmp_path / "hazard"
    generate_dummy_log(route)

    summary = run_hazard_teacher(route, out, backend_name="fake", prompts_path="configs/hazard/prompts.yaml")
    manifest = load_teacher_manifest(out)

    assert summary.teacher_name == "hazard"
    assert manifest["teacher_name"] == "hazard"
    assert manifest["backend"] == "fake"
    assert manifest["mock"] is True
    assert manifest["synthetic"] is True
    assert manifest["real_perception"] is False
    assert manifest["weak_label"] is True
    assert manifest["control_safe"] is False
    assert manifest["trainable_for"] == "hazard_pretrain_only"
    assert manifest["hazard_positive_frame_count"] > 0

    validation = validate_teacher_artifacts(out)
    assert validation.artifact_load_success is True
    assert validation.teacher_mock_used is True
    assert validation.artifact_determinism_pass is True

    first = manifest["frames"][0]
    masks = load_array(out / first["artifacts"]["hazard_masks"]["path"])
    confidence = load_array(out / first["artifacts"]["hazard_confidence"]["path"])
    boxes = json.loads((out / first["artifacts"]["hazard_boxes"]["path"]).read_text(encoding="utf-8"))
    assert masks.shape[1:] == (first["height"], first["width"])
    assert masks.dtype == np.float32
    assert confidence.shape == masks.shape
    assert boxes["hazard_boxes"][0]["score"] > 0.0


def test_real_hazard_teacher_can_reexec_external_python(monkeypatch) -> None:
    args = argparse.Namespace(teacher="hazard", backend="real")
    monkeypatch.delenv("HOMEBRAIN_HAZARD_REEXECED", raising=False)
    monkeypatch.setattr(run_teacher.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(run_teacher, "_external_hazard_python", lambda: Path("/tmp/hazard-python"))

    assert run_teacher._should_reexec_hazard(args) is True


def test_real_hazard_box_above_class_threshold_becomes_positive_mask() -> None:
    prompt_config = load_hazard_prompt_config("configs/hazard/prompts.yaml")
    masks, confidence, records = _rasterize_grounding_dino_boxes(
        boxes=np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
        logits=np.asarray([0.36], dtype=np.float32),
        phrases=["cable"],
        image_shape=(8, 8),
        class_names=prompt_config.class_names,
        prompt_config=prompt_config,
    )

    assert records[0].class_name == "cable"
    assert float(np.max(masks)) >= 0.5
    assert np.isclose(float(np.max(confidence)), 0.36, atol=1.0e-5)


def test_hazard_external_python_skips_env_without_groundingdino(monkeypatch, tmp_path: Path) -> None:
    bad_python = tmp_path / "bad-python"
    good_python = tmp_path / "good-python"
    bad_python.write_text("", encoding="utf-8")
    good_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("HOMEBRAIN_HAZARD_PYTHON", bad_python.as_posix())
    monkeypatch.setattr(
        run_teacher,
        "_hazard_python_has_groundingdino",
        lambda path: Path(path) == good_python,
    )
    monkeypatch.setattr(run_teacher, "_repo_root", lambda: tmp_path)
    status = tmp_path / "external" / "hazard_teacher_setup_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"python": {"path": good_python.as_posix()}}) + "\n",
        encoding="utf-8",
    )

    assert run_teacher._external_hazard_python() == good_python
