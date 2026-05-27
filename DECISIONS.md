## D012 - World model first, RL later

Decision: HomeBrain will prioritize learned scene memory, future BEV/risk prediction, and candidate outcome scoring before online RL.

Reason: There is no real robot, no large owned robot dataset, and no reliable interactive home simulator. RL without credible action-conditioned dynamics will optimize fake objectives.

Status: active.

## D013 - Mostly neural does not mean geometry-free

Decision: Calibration, odometry, frame transforms, BEV grids, footprint projection, and safety envelopes are allowed and expected. The neural part should learn perception, memory, uncertainty, future risk, and candidate outcome scoring.

Reason: A real robot needs measurable state and safety boundaries. Rejecting geometry because it looks classical would make deployment less realistic, not more modern.

Status: active.

## D014 - Open-weight teachers, deployable student

Decision: Large open-weight models are used offline for labels, features, audits, and distillation. The runtime path should trend toward a smaller direct student.

Reason: A consumer robot cannot depend on many huge teacher models every control tick.

Status: active.

## D015 - Code growth must buy measured behavior

Decision: New code must be connected to train/eval/runtime and must produce a test, metric, artifact, or deletion. Large parallel paths are not allowed without retiring older paths.

Reason: Prevent slop code and scaffolding accumulation.

Status: active.