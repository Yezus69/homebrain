import json

from homebrain.eval.run_eval import evaluate_log, write_eval_metrics
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.teachers.mock_teacher import run_mock_teacher


def test_eval_metrics_for_dummy_log(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    out = tmp_path / "dummy_eval.json"
    generate_dummy_log(source)

    metrics = evaluate_log(source, eval_runtime_sec=0.0)
    write_eval_metrics(metrics, out)
    written = json.loads(out.read_text(encoding="utf-8"))

    assert written == metrics
    assert metrics["event_count"] == 24
    assert metrics["frame_count"] == 6
    assert metrics["dropped_frame_count"] == 0
    assert metrics["event_ordering_error_count"] == 0
    assert metrics["replay_determinism_pass"] is True
    assert metrics["brain_output_count"] == 6
    assert metrics["eval_runtime_sec"] == 0.0


def test_eval_metrics_validate_teacher_artifacts(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    artifacts = tmp_path / "teacher_artifacts"
    out = tmp_path / "dummy_eval_with_teacher.json"
    generate_dummy_log(source)
    run_mock_teacher(source, artifacts)

    metrics = evaluate_log(source, teacher_artifacts=artifacts, eval_runtime_sec=0.0)
    write_eval_metrics(metrics, out)
    written = json.loads(out.read_text(encoding="utf-8"))

    assert written == metrics
    assert metrics["frame_count"] == 6
    assert metrics["teacher_artifact_count"] == 30
    assert metrics["teacher_mock_used"] is True
    assert metrics["artifact_load_success"] is True
    assert metrics["frames_with_teacher_artifacts"] == 6
    assert metrics["missing_artifact_count"] == 0
    assert metrics["artifact_shape_error_count"] == 0
    assert metrics["artifact_determinism_pass"] is True
