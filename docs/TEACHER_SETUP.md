# Teacher Setup

HomeBrain teachers are offline training-data tools. They are not production runtime dependencies, and their outputs are not control-safe robot commands.

## Depth Pro

Depth Pro support is optional. The HomeBrain test suite does not install Depth Pro and never downloads model weights.

Official sources:
- Code: <https://github.com/apple/ml-depth-pro>
- License: <https://github.com/apple/ml-depth-pro/blob/main/LICENSE>
- Paper page: <https://machinelearning.apple.com/research/depth-pro>

License status in HomeBrain: `pending_human_review`. Do not treat Depth Pro code, weights, or outputs as product-runtime approved until `LICENSE_AUDIT.md` is updated by a human reviewer.

### Install

Use an isolated environment. From outside this repository:

```bash
git clone https://github.com/apple/ml-depth-pro
cd ml-depth-pro
python -m venv .venv
source .venv/bin/activate
pip install -e .
source get_pretrained_models.sh
```

On Windows, run the shell script from Git Bash or WSL, or follow the official repository's current checkpoint download instructions.

### Run Real Depth Pro

After importing a route:

```bash
python -m homebrain.teachers.run_teacher --teacher depth_pro --backend real --log runs/room_walk_route --out runs/room_walk_route/teacher_artifacts/depth_pro
python -m homebrain.teachers.visualize_artifacts --artifacts runs/room_walk_route/teacher_artifacts/depth_pro --out runs/room_walk_depth_pro_viz
python -m homebrain.eval.run_eval --log runs/room_walk_route --teacher-artifacts runs/room_walk_route/teacher_artifacts/depth_pro --out runs/room_walk_depth_pro_eval.json
```

If Depth Pro or local checkpoints are unavailable, the CLI fails clearly. HomeBrain does not download weights automatically.

### Test Backend

The fake backend exists only for tests and local pipeline checks:

```bash
python -m homebrain.teachers.run_teacher --teacher depth_pro --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/depth_pro_fake
```

Fake backend artifacts are explicitly marked `mock: true`, `synthetic: true`, and `real_perception: false`.

### Depth Pro Artifacts

Per frame:
- `depth_m.npy`: metric-depth estimate in meters for real Depth Pro runs.
- `depth_confidence.npy`: model confidence if available, otherwise a documented HomeBrain heuristic.
- `focallength_px.npy`: focal length in pixels from Depth Pro, frame intrinsics, or a metadata-labeled fallback.
- `bev_preview.npy`: rough visualization-only preview, not a BEV navigation label.
- `metadata.json`: provenance, dependency status, confidence source, and safety notes.

Route level:
- `teacher_manifest.json` records `teacher_name=depth_pro`, backend status, license review status, dependency status, and whether the run used real perception or a fake test backend.
