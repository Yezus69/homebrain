Complete Goal29: Action-Conditioned Neural Risk World Model without stopping until HomeBrain has a tested replay-only world-model path that improves the robot brain’s ability to perceive, remember, predict, and choose safer useful local motion in dynamic indoor scenes.

Main objective:

Build the next hard vertical slice of the robot brain:

online SceneState + BEV memory
+ bounded candidate trajectories
+ action-conditioned neural future BEV/risk prediction
+ learned candidate outcome scoring
+ replay-only safety envelope
+ route/fixture eval artifacts

This is not a scaffolding task. This is not a docs task. This is not a simulator task. This is not a hardware-control task.

The core question the code must answer is:

“Given what the robot currently sees and remembers, and given each bounded candidate trajectory, which candidate is likely to become unsafe, uncertain, blocked, or useful over the next short horizon?”

Read policy:

1. Read AGENTS.md first.
2. Read CURRENT_STATUS.md and EVALS.md.
3. Read ARCHITECTURE.md only if needed for module boundaries.
4. Inspect the relevant code and tests before reading any other Markdown.
5. Do not read archive docs, old plans, stale goals, or deleted-doc replacements.

Hard constraints:

- No real robot exists.
- No real robot hardware control.
- No raw PWM.
- No fake hardware APIs.
- No high-fidelity sim assumption.
- No YouTube-to-fake-physics simulator.
- No online RL.
- No giant new framework.
- No unconnected scaffolding.
- No runtime dependency on heavy teacher models.
- No future-frame, future-label, route-ground-truth, oracle-BEV, or teacher-artifact leakage at runtime.
- Keep all runtime decisions replay-only.

Required safety invariants:

replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false
hardware_validated=false

Architecture target:

Implement an action-conditioned neural future-risk/world-model path that uses existing HomeBrain concepts where possible:

- current/local BEV,
- online SceneState / memory BEV,
- uncertainty,
- sensor masks,
- pose deltas / odom where available,
- previous action or candidate trajectory encoding where available,
- bounded local candidate trajectories,
- future BEV/risk labels for training/eval only,
- candidate outcome labels for training/eval only.

Prefer extending the existing FutureBEV / rollout / candidate scoring path over creating a parallel stack.

If existing names are different, follow the repo’s current naming style. Do not add a second architecture unless needed.

Model requirement:

Implement a small trainable neural model suitable for two RTX 4090-class GPUs.

Preferred design:

- BEV-history encoder using small ConvGRU, temporal ConvNet, or compact transformer.
- Scene memory input.
- Candidate trajectory conditioning.
- Future BEV/risk prediction over short horizons.
- Candidate outcome heads.

The model should predict, where labels exist:

- future free probability,
- future occupied probability,
- future unknown probability,
- future risk probability,
- uncertainty,
- candidate collision risk,
- candidate future collision risk,
- candidate unknown exposure,
- candidate progress,
- candidate coverage/new-area gain,
- candidate stop/recovery reason or logits if appropriate.

Do not build a giant VLA or end-to-end motor policy.

Data/label requirement:

Use existing route packs, BEV packs, FutureBEV packs, teacher artifacts, and public-data import paths where possible.

If real route data is unavailable in the test environment, create deterministic tiny fixtures that still test the hard logic.

At minimum, add fixtures for:

1. moving obstacle crossing the robot path,
2. static obstacle in known free space,
3. unknown corridor,
4. high uncertainty near the footprint,
5. candidate that looks safe now but becomes unsafe in the future.

The moving-obstacle fixture is mandatory.

Pack requirement:

Add or extend a pack format/dataset path that can provide:

- past BEV or SceneState history,
- current BEV,
- memory BEV,
- uncertainty map,
- sensor mask,
- pose deltas,
- candidate trajectories / candidate footprints,
- future BEV targets,
- future valid horizon masks,
- candidate outcome labels.

The pack must be deterministic and must reject or warn on bad/missing data instead of silently inventing truth.

Runtime requirement:

Wire the model into runtime decision behind an explicit config/checkpoint flag.

Runtime decision must:

- use only online-available inputs,
- use SceneState/memory if available,
- score bounded candidate trajectories,
- always include a stop candidate,
- produce replay-only candidate selection artifacts,
- preserve conservative safety flags,
- not execute cmd_vel,
- not require teacher models at runtime,
- not require future labels at runtime.

Safety envelope requirement:

Add or extend replay-only safety envelope outputs for:

- stale sensor stop,
- high uncertainty stop,
- high predicted risk stop,
- invalid candidate rejection,
- bounded recovery proposal if existing architecture supports it,
- explicit stop/recovery reason.

Do not create real hardware transport.

Evaluation requirement:

Use EVALS.md gates.

This task must improve or directly exercise these gates:

- Gate D: Online SceneState/memory quality,
- Gate E: Future BEV/risk prediction,
- Gate F: Learned candidate scoring,
- Gate H: Hardware-readiness replay safety.

Gate G route-heldout robustness should be used if existing route packs or reports are available.

Required baselines:

Compare against all relevant baselines that can run in the repo:

- previous accepted behavior / Goal28 path if available,
- current-BEV-copy-forward,
- stop-only,
- transparent scorer,
- unknown-is-dangerous conservative scorer.

A learned model is not progress unless it beats at least one meaningful baseline on a relevant fixture or route without violating safety invariants.

Testing requirement:

Add focused tests proving:

1. pack shapes are correct,
2. pack generation is deterministic,
3. model forward pass works,
4. no future-frame/future-label/teacher/oracle/ground-truth leakage occurs at runtime,
5. moving obstacle fixture changes future risk,
6. candidate scoring changes when future risk crosses a candidate footprint,
7. unsafe future candidate is rejected or ranked worse,
8. stop candidate remains available,
9. high uncertainty can trigger a replay-only stop reason,
10. safety invariants remain conservative,
11. report JSON is written,
12. no raw PWM or hardware transport is introduced.

If feasible, add a tiny overfit/loss-decrease test on the deterministic fixture. Keep it small enough for CI.

Metrics/report requirement:

Produce a machine-readable report artifact, preferably JSON, with at least:

{
  "accepted": false,
  "goal": "Goal29",
  "gates_improved": [],
  "routes": [],
  "fixtures": [],
  "baselines": [],
  "metrics": {},
  "artifacts": [],
  "tests": [],
  "hard_failures": [],
  "caveats": [],
  "safety": {
    "replay_only": true,
    "not_executed": true,
    "control_safe": false,
    "raw_pwm_emitted": false,
    "hardware_validated": false
  }
}

Set accepted=true only if the implementation passes tests, produces metrics, improves at least one meaningful Gate E/F metric over a baseline, and preserves all safety invariants.

Suggested useful metrics:

- future_free_iou_or_proxy,
- future_occupied_iou_or_proxy,
- future_unknown_error_or_proxy,
- future_risk_auc_or_proxy,
- improvement_vs_copy_forward,
- candidate_risk_ranking_accuracy,
- unsafe_candidate_rejection_rate,
- stop_selected_fraction,
- dominant_action_fraction,
- selected_candidate_entropy,
- unknown_exposure_mean,
- command_envelope_violation_count,
- high_uncertainty_stop_count,
- high_risk_stop_count,
- raw_pwm_emitted,
- control_safe,
- hardware_validated.

Code hygiene requirement:

Do not let the repo grow sideways.

Before adding a new module, ask:

- What existing train/eval/runtime path will call it?
- What test fails without it?
- What metric or artifact proves it ran?
- What old code can be deleted or simplified?

Prefer modifying existing paths over adding parallel ones.

Avoid files over 400 lines unless strongly justified.
Avoid functions over 80 lines unless strongly justified.
Avoid broad rewrites.
Avoid new config layers unless required by the vertical slice.
Do not update lots of Markdown.

Allowed Markdown updates:

- Update CURRENT_STATUS.md after the run with the latest truthful state.
- Update EVALS.md only if a real eval gate changed.
- Do not add milestone history elsewhere.

Deliverables:

1. Implementation of the action-conditioned neural future-risk/world-model path.
2. Pack/dataset support or deterministic fixture support for BEV history + candidates + future labels.
3. Runtime integration behind explicit config/checkpoint flag.
4. Replay-only safety envelope outputs.
5. Tests for shapes, leakage, dynamic obstacle behavior, candidate scoring, safety invariants, and report writing.
6. Baseline comparison.
7. Machine-readable report artifact.
8. Updated CURRENT_STATUS.md only after the result is known.

Acceptance bar:

The task is complete only when:

- focused tests pass,
- git diff --check passes,
- full pytest passes if feasible,
- a report artifact exists,
- at least one meaningful Gate E or Gate F metric improves over a baseline on a deterministic fixture or existing route,
- runtime remains replay-only,
- raw_pwm_emitted=false,
- control_safe=false,
- hardware_validated=false,
- no dead scaffolding remains.

If full route-heldout data is unavailable, still complete the deterministic fixtures and report that route-heldout evidence is unavailable. Do not fake route-heldout success.

If the learned model does not beat a baseline, keep the tests and report the failure honestly, but do not mark Goal29 accepted.