# Teacher Setup

HomeBrain teachers are offline training-data tools. They are not production runtime dependencies, and their outputs are not control-safe robot commands.

## Depth Anything 3

Depth Anything 3 support is optional and isolated under `external/`. The normal test suite uses only the fake DA3 backend and must not download weights.

Official sources:
- Code: <https://github.com/ByteDance-Seed/depth-anything-3>
- Model: <https://huggingface.co/depth-anything/DA3-SMALL>

POC status in HomeBrain: offline teacher use is allowed when provenance is
recorded. Do not treat DA3 code, weights, or outputs as product-runtime or
control-safety approved.

### Setup

This explicit setup command is allowed to clone, install dependencies, and download/cache weights:

```powershell
python -m homebrain.tools.setup_da3_teacher --external-dir external/depth-anything-3 --venv external/venvs/da3 --model-id depth-anything/DA3-SMALL --download
```

It writes `external/da3_setup_status.json` with the cloned repo commit, venv Python path, model cache path, dependency status, and CUDA status. `external/` and model weights are gitignored.

The DA3 upstream package currently imports `addict` from the API path, so HomeBrain setup installs that small missing dependency and validates `from depth_anything_3.api import DepthAnything3`, not only the top-level namespace.

### Run Fake DA3

The fake backend exists only for tests and pipeline checks:

```bash
python -m homebrain.teachers.run_teacher --teacher da3 --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/da3_fake
python -m homebrain.geometry.qa_self_calibration --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/da3_fake --out runs/dummy_da3_self_calib_qa.json
```

Fake DA3 artifacts are explicitly marked `mock=true`, `synthetic=true`, and `real_perception=false`. QA must quarantine them and must not promote them to weak BEV.

### Run Real DA3

After setup, the normal Python CLI can auto-reexec through the external DA3 venv when the current environment lacks the DA3 API. Running the venv Python directly is also valid:

```powershell
python -m homebrain.teachers.run_teacher --teacher da3 --backend real --device cpu --model-id depth-anything/DA3-SMALL --max-frames 60 --window-size 10 --stride 1 --log runs/room_walk_001_route_short60 --out runs/room_walk_001_route_short60/teacher_artifacts/da3

.\external\venvs\da3\Scripts\python.exe -m homebrain.teachers.run_teacher --teacher da3 --backend real --device cpu --model-id depth-anything/DA3-SMALL --max-frames 60 --window-size 10 --stride 1 --log runs\room_walk_001_route_short60 --out runs\room_walk_001_route_short60\teacher_artifacts\da3
```

Real DA3 per-frame artifacts:
- `depth.npy`: relative teacher depth, not metric robot truth.
- `confidence.npy`: teacher confidence if available.
- `intrinsics.npy`: teacher-estimated camera intrinsics.
- `extrinsics.npy`: teacher-estimated camera pose/extrinsics.
- `metadata.json`: provenance, dependency status, calibration metadata, and safety notes.

Route manifest fields include provenance/license status, `calibration_class=teacher_estimated`, `scale_source=teacher_relative_not_metric`, `not_robot_frame_truth=true`, and `control_safety=not_control_safe_training_teacher_only`.

### Self-Calibration QA and Weak BEV

Run QA before any DA3-derived weak BEV:

```bash
python -m homebrain.geometry.qa_self_calibration --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --out runs/room_walk_001_da3_self_calib_qa_short60.json
```

Weak BEV promotion is gated:

```bash
python -m homebrain.geometry.da3_to_weak_bev --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --qa runs/room_walk_001_da3_self_calib_qa_short60.json --out runs/room_walk_001_route_short60/geometry/da3_weak_bev
```

If QA fails, the command refuses to write BEV unless `--force-review` is supplied. Forced outputs are still `weak_label=true`, `control_safe=false`, `calibration_class=teacher_estimated`, `not_robot_frame_truth=true`, and `trainable_for=geometry_pretrain_only`.

## Depth Pro

Depth Pro support is optional. The HomeBrain test suite does not install Depth Pro and never downloads model weights.

Official sources:
- Code: <https://github.com/apple/ml-depth-pro>
- License: <https://github.com/apple/ml-depth-pro/blob/main/LICENSE>
- Paper page: <https://machinelearning.apple.com/research/depth-pro>

POC status in HomeBrain: offline teacher use is allowed when provenance is
recorded. Do not treat Depth Pro code, weights, or outputs as product-runtime or
control-safety approved.

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

On Windows without Conda, a local venv works. From this repository root:

```powershell
cd external\ml-depth-pro
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e .

New-Item -ItemType Directory -Force checkpoints
curl.exe -L --fail --continue-at - --output checkpoints\depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
```

If using Hugging Face instead of Apple's direct URL, recent `huggingface_hub` versions use the `hf` CLI:

```powershell
.\.venv\Scripts\hf.exe download --local-dir checkpoints apple/DepthPro
```

HomeBrain looks for the checkpoint in the editable `ml-depth-pro` clone. If the weights live elsewhere, set `DEPTH_PRO_CHECKPOINT` to the full `depth_pro.pt` path before running the real backend.

### Run Real Depth Pro

After importing a route:

```powershell
.\external\ml-depth-pro\.venv\Scripts\python.exe -m homebrain.teachers.run_teacher --teacher depth_pro --backend real --device cuda --log runs\room_walk_route --out runs\room_walk_route\teacher_artifacts\depth_pro
.\external\ml-depth-pro\.venv\Scripts\python.exe -m homebrain.teachers.visualize_artifacts --artifacts runs\room_walk_route\teacher_artifacts\depth_pro --out runs\room_walk_depth_pro_viz
.\external\ml-depth-pro\.venv\Scripts\python.exe -m homebrain.eval.run_eval --log runs\room_walk_route --teacher-artifacts runs\room_walk_route\teacher_artifacts\depth_pro --out runs\room_walk_depth_pro_eval.json
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
