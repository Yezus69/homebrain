# HomeBrain

HomeBrain is a production-directed software stack for an indoor floor-cleaning robot brain.

The target robot has:
- RGB camera(s)
- IR camera later
- IMU
- wheel encoders
- differential-drive motors
- no lidar
- no GPS
- no ultrasonic dependency

The current phase is software-only. Hardware integration comes later.

The first useful system is not a perfect robot. The first useful system is a deterministic log/replay/eval spine that can ingest indoor visual data, run teacher models, train a small spatial-memory student, and produce measurable outputs.

Start here:
1. Read `PROJECT_BRIEF.md`.
2. Read `AGENTS.md`.
3. Read `CODEX_GOALS.md`.
4. Run one Codex goal at a time.
5. After every goal, inspect or share `CURRENT_STATUS.md`, not the whole codebase.
