# HomeBrain Codex Contract

HomeBrain is a research-to-product software stack for a future low-speed indoor vacuum/mop robot.

The hard problem is not scaffolding. The hard problem is a mostly-neural robot brain that can perceive, remember, predict, and choose safe useful motion in messy dynamic homes using camera/IR/IMU/wheel data.

## Current north star

Build a replay-first neural scene world model and candidate-action brain:

real robot/public route data
-> deterministic log/replay
-> open-weight teacher artifacts offline
-> direct runtime student
-> online SceneState
-> local + scene BEV memory
-> short-horizon future BEV/risk prediction
-> bounded candidate trajectory scoring
-> replay-only cmd_vel proposal
-> hard metrics and visual proof

Runtime decisions remain replay-only until real hardware safety exists.

## Read policy

Codex should read only this file by default.

For implementation tasks, inspect the relevant code and tests before reading more Markdown.

Read `CURRENT_STATUS.md` only for broad robot-brain, runtime, milestone, or blocker questions.

Read `EVALS.md` only when changing metrics, training, policy, runtime decision behavior, or acceptance gates.

Read `ARCHITECTURE.md` only when changing the high-level robot-brain design or moving module boundaries.

Do not read old plans, deleted-doc replacements, archive docs, stale goals, stale handoff files, or historical milestone notes unless explicitly asked.

When uncertain, inspect code and tests before reading more Markdown.

## Current accepted baseline

The latest accepted local baseline is Goal28 caveat repair, not Goal26 or Goal27.

Goal28 is still not product deployment. It is replay-only public-data evidence with online SceneState, scene-memory-conditioned FutureBEV selection, learned visual/memory pose correction, non-degenerate behavior-cloning labels, and three heldout OpenLORIS scenes.

Remaining blockers:
- no real robot hardware loop,
- no synchronized owned RGB/IR/IMU/wheel/command logs,
- no controller/watchdog/recovery/docking/safety gate,
- weak traversability/free-space labels,
- weak dynamic-risk supervision,
- narrow public-route coverage,
- no policy good enough for real hardware.

## What useful work means

A change is useful only if it improves at least one of:

1. real-data ingestion/calibration/provenance,
2. open-weight teacher labels/features,
3. direct runtime student quality,
4. BEV/free/occupied/unknown/risk label quality,
5. online SceneState/spatial memory,
6. action-conditioned future BEV/risk prediction,
7. learned candidate trajectory scoring,
8. replay metrics/evals/visual proof,
9. hardware-readiness safety envelope in replay only,
10. code deletion/simplification without behavior loss.

If a change does not improve one of those, do not do it.

## What not to build

Do not add:
- generic robotics framework code,
- placeholder daemons,
- fake hardware APIs,
- unconnected CLIs,
- untrained model classes,
- synthetic-only milestones,
- dashboards before the underlying BEV/risk outputs work,
- RL loops without action-conditioned data,
- YouTube-to-fake-physics simulators,
- hand-authored diversity tricks counted as learned policy success,
- raw PWM control.

## Architecture rule

Use modern ML, but do not confuse “mostly neural” with “no geometry.”

Allowed and expected:
- calibration,
- timestamps,
- camera-to-base transforms,
- odometry/IMU/wheel inputs,
- BEV grids,
- candidate trajectory footprints,
- hard safety envelopes,
- deterministic replay,
- explicit provenance.

The neural core should be perception, memory, future prediction, uncertainty, and candidate outcome scoring.

## Open-weight model rule

Use large open-weight models as offline teachers, not runtime dependencies.

Good teacher roles:
- depth/geometry,
- dense visual features,
- segmentation/masks,
- dynamic-object hints,
- traversability/risk weak labels,
- navigation priors for pseudo-labels or ablations.

Train a smaller direct runtime student. Runtime should not require DINO/MoGe/Depth/SAM-style heavy teachers every control tick.

## RL/ML rule

Do not start with online RL.

Current priority is model-based/offline learning:
- learn current BEV/risk,
- learn scene memory,
- learn short-horizon future BEV/risk under candidate motion,
- learn candidate outcome scores,
- evaluate route-heldout.

RL can come later as offline objective/value learning over the learned world model, but only after labels, dynamics, and evals are credible.

## Required validation

Every nontrivial change must include:
- focused tests,
- deterministic artifacts or metrics,
- comparison against a baseline when behavior changes,
- no runtime leakage from future frames, future labels, route ground truth, oracle BEV, or teacher outputs,
- replay safety flags unchanged:
  replay_only=true
  not_executed=true
  control_safe=false
  raw_pwm_emitted=false

For model/policy work, include at least one of:
- tiny overfit test,
- route-heldout metric,
- synthetic fixture for a specific failure mode,
- visual proof artifact,
- baseline comparison.

## Code hygiene

Prefer deleting or merging old paths over adding parallel versions.

No new module unless:
- an existing train/eval/runtime path uses it,
- a test fails without it,
- an artifact or metric proves it ran.

Avoid files over 400 lines unless there is a strong reason.
Avoid broad rewrites.
Avoid adding config layers before behavior works.
Every large addition should delete, simplify, or obsolete something.

## Code growth budget

Every PR must explain net new code.

Prefer:
- modifying existing paths over adding new ones,
- one model path over many parallel model paths,
- one eval command over many wrappers,
- deleting obsolete goal-specific code after the stronger path exists.

Soft limits:
- avoid new files over 400 lines,
- avoid functions over 80 lines,
- avoid new packages unless the runtime/train/eval path uses them,
- if a change adds more than 800 net lines, it should delete or consolidate something too.

A new module is rejected unless:
- an existing command calls it,
- a test covers it,
- a metric/artifact proves it ran,
- it improves a listed gate.

## Next hard direction

Goal29 should not be another milestone wrapper.

Goal29 should improve real hardware readiness by preserving Goal28’s online SceneState path while attacking the real remaining gaps:
- broader route-heldout robustness,
- dynamic-risk supervision,
- stronger traversability/free-space labels,
- replay-compatible watchdog/recovery/safety envelope,
- direct RGB-D/RGB-IR student quality,
- action-conditioned future risk prediction,
- non-collapsed learned candidate scoring across more scenes.

Do not claim robot-brain success unless the eval says it.