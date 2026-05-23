import json

from homebrain.eval.run_eval import evaluate_log, write_eval_metrics
from homebrain.replay.generate_dummy_log import generate_dummy_log


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
