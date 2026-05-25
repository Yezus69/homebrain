# LICENSE_AUDIT.md

Codex must update this when adding external models, datasets, or libraries.

Do not assume a dependency is production-safe. Verify from official repos/model cards/licenses.

## Status meanings

- `UNVERIFIED`: not checked yet
- `RESEARCH_ONLY`: allowed for experiments, not product runtime/training
- `TRAINING_ONLY`: can be used to produce pseudo-labels/training data if license allows
- `RUNTIME_OK`: can be used in deployed product, pending legal review
- `AVOID`: do not use

## Candidate models and datasets

| Name | Intended use | Current status | Production runtime? | Notes |
|---|---|---:|---:|---|
| DINO-family visual model | dense features / teacher | UNVERIFIED / pending_human_review | No | Goal 7 used DINOv2-small `dinov2_vits14` through torch.hub as an offline frozen feature teacher; verify exact repo + weight license before product/training approval. |
| Depth Anything family | geometry/depth teacher | UNVERIFIED / pending_human_review | No | DA3-SMALL setup added for offline teacher use; verify exact code/model card/license before product use. |
| Apple Depth Pro | optional metric depth teacher | UNVERIFIED / pending_human_review | No | Official repo: https://github.com/apple/ml-depth-pro. License terms require human review before product use. |
| MoGe / point-map model | geometry teacher | MIT observed / pending_human_review | No | Goal 16C installed official MoGe from `https://github.com/microsoft/MoGe` at commit `07444410f1e33f402353b99d6ccd26bd31e469e8` and downloaded `Ruicheng/moge-2-vits-normal` revision `679230677b4d282c6f304189a93e98e14f085902` to the local Hugging Face cache for offline teacher use only. Not product-training-approved or runtime-safe. |
| VGGT-family | offline multi-view geometry teacher | UNVERIFIED / pending_human_review | No | Goal 15A added an optional local VGGT-style scene-teacher adapter plus fake backend tests. No real code/checkpoint was downloaded or approved. Verify exact repo/checkpoint license before real use. |
| SAM-family video segmentation | mask/dynamic-object teacher | UNVERIFIED | No | Verify exact version. |
| GNM / ViNT / NoMaD | navigation prior / baseline | UNVERIFIED | No | Verify code/checkpoint/dataset license. |
| AI2-THOR / ProcTHOR | synthetic control data | UNVERIFIED | No | Use only after license check. |
| iGibson | synthetic control data | UNVERIFIED | No | Use only after license check. |
| ARKitScenes | public indoor geometry data | UNVERIFIED | No | Verify dataset terms. |
| ScanNet / ScanNet++ | public indoor geometry data | UNVERIFIED | No | Verify dataset terms. |
| TUM RGB-D SLAM Dataset | public RGB-D / pose geometry anchor | UNVERIFIED / pending_human_review | No | Official page lists CC BY 4.0; Goal 10B downloaded/imported `freiburg2_pioneer_slam` but kept it out of robot-frame action labels because camera-to-base/base-pose semantics are missing. |
| OpenLORIS-Scene | public robot-mounted indoor RGB-D/IMU/odom/pose bridge | PoC allowed / product pending_human_review | No | Official dataset page lists CC BY-ND 4.0; local HomeBrain PoC training/eval is allowed, but product training/runtime approval and derived dataset redistribution remain blocked. |
| Ego4D / EPIC-KITCHENS | visual clutter/dynamics data | UNVERIFIED | No | Verify dataset terms. |

## Rule

If a goal adds a dependency, it must add:
- source URL or local provenance
- license name
- allowed use
- risk
- whether it is required at runtime
- whether a mock/fallback exists

## Goal 1 dependency update

Goal 1 added the teacher artifact interface and deterministic mock teacher only.

- No real teacher model, dataset, checkpoint, or external model weight was added.
- No torch, ROS, Nav2, Habitat, Isaac, or simulator dependency was added.
- The mock teacher is local HomeBrain code and emits `mock: true`, `synthetic: true`, and `real_perception: false`.
- The only numerical file dependency used by the implementation is `numpy`, which was already allowed by `AGENTS.md` for initial development.
- Candidate real teacher licenses above remain `UNVERIFIED` and must be audited before any real wrapper/checkpoint is treated as usable.

## Goal 3 dependency update

Goal 3 added an optional Depth Pro teacher wrapper.

- Source URL: https://github.com/apple/ml-depth-pro
- License URL: https://github.com/apple/ml-depth-pro/blob/main/LICENSE
- Model/checkpoint source: official Depth Pro checkpoint setup from the Apple repository.
- License name/status in HomeBrain: `pending_human_review`; do not mark as runtime-safe or product-safe yet.
- Intended use: offline training-time geometry teacher producing pseudo-label artifacts.
- Required at runtime: no.
- Required for normal tests: no.
- Mock/fallback status: a fake Depth Pro backend exists only when explicitly selected with `--backend fake`; those artifacts are marked `mock: true`, `synthetic: true`, and `real_perception: false`.
- Risk: real Depth Pro outputs are monocular estimates and are not control-safe. Use them only as teacher artifacts until calibrated evaluation and legal review are complete.

## Goal 6A dependency update

Goal 6A added explicit setup and a real/fake wrapper for Depth Anything 3.

- Source URL: https://github.com/ByteDance-Seed/depth-anything-3
- Model/checkpoint source: https://huggingface.co/depth-anything/DA3-SMALL
- License name/status in HomeBrain: `pending_human_review`; do not mark as runtime-safe, product-safe, or training-approved beyond local review artifacts yet.
- Intended use: offline geometry/self-calibration teacher producing pseudo-label artifacts.
- Required at runtime: no.
- Required for normal tests: no.
- Mock/fallback status: a fake DA3 backend exists only when explicitly selected with `--backend fake`; those artifacts are marked `mock: true`, `synthetic: true`, and `real_perception: false`.
- Risk: DA3-SMALL outputs are relative teacher geometry and camera estimates, not robot-frame metric truth or control-safe labels. Weak BEV promotion is gated by self-calibration QA.

## Goal 6B dependency update

Goal 6B added a public RGB-D/pose truth-anchor importer for TUM RGB-D.

- Source URL: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- Sequence/download used: `freiburg1_xyz`, `rgbd_dataset_freiburg1_xyz.tgz`
- License name/status in HomeBrain: `CC BY 4.0` observed on the official dataset page, with `license_review_status=pending_human_review`.
- Intended use: offline public RGB-D/pose geometry anchor for depth/pose/BEV structural checks and geometry pretraining review.
- Required at runtime: no.
- Required for normal tests: no; tests use a tiny synthetic TUM-style fixture generated locally.
- Mock/fallback status: no fake public dataset is substituted for missing real data. If download/setup fails, setup metadata and `BLOCKERS.md` must record the exact failure.
- Risk: TUM camera trajectories are dataset camera poses and are not robot-frame metric traversability truth; generated BEV anchors remain `control_safe=false` and `not_robot_frame_truth=true`.

## Goal 7 dependency update

Goal 7 added DINOv2-small as an offline frozen visual feature teacher and PyTorch student training/inference for SpatialMemoryNet v0.

- Source URL: https://github.com/facebookresearch/dinov2
- Model/checkpoint source: torch hub `facebookresearch/dinov2`, model id `dinov2_vits14`, checkpoint file `dinov2_vits14_pretrain.pth`.
- Setup status: `external/dino_setup_status.json` reports `success=true`, `torch=2.1.0+cu121`, CUDA available, and model loaded on `cuda`.
- License name/status in HomeBrain: `pending_human_review`; do not mark as runtime-safe, product-safe, or training-approved beyond local research artifacts yet.
- Intended use: offline frozen dense visual feature teacher for representation pretraining.
- Required at runtime: no. Student checkpoint inference in `modeld` uses saved DINO feature artifacts for replay, not the DINO model.
- Required for normal tests: no. Tests use `--backend fake` artifacts marked `mock=true`, `synthetic=true`, and `real_perception=false`.
- Mock/fallback status: fake DINO backend exists only when explicitly selected with `--backend fake`.
- Risk: DINO features are visual representation artifacts, not geometry/traversability truth or control-safety evidence; all Goal 7 outputs remain `representation_pretraining_only=true` and `control_safe=false`.

## Goal 7B dependency update

Goal 7B did not add new external dependencies, models, datasets, or product-runtime requirements.

- DINOv2-small was reused as an offline frozen feature teacher and run on the TUM `freiburg1_xyz` route; status remains `pending_human_review`.
- TUM RGB-D was reused for public RGB-D/depth/pose geometry anchors; status remains `CC BY 4.0` observed with `license_review_status=pending_human_review`.
- DA3 weak geometry labels were reused only as `weak_visual_geometry`; status remains `pending_human_review`.
- TUM pose deltas are stored as `camera_relative_dataset_pose`, explicitly not robot odometry or robot-frame truth.
- All new checkpoints, metrics, replay outputs, and contact sheets are `representation_pretraining_only=true` and `control_safe=false`.

## Goal 10A dependency update

Goal 10A added dataset bridge code for public robot-mounted data and attempted bounded public-data setup. It did not import a real OpenLORIS/TUM Pioneer sequence because both selected archives exceeded the configured caps.

- OpenLORIS-Scene source URL: https://lifelong-robotic-vision.github.io/dataset/scene.html
- OpenLORIS official download docs: https://github.com/lifelong-robotic-vision/OpenLORIS-Scene/blob/master/download.md
- OpenLORIS Hugging Face package mirror inspected: https://huggingface.co/datasets/shixuesong/openloris-scene/tree/main/package
- OpenLORIS tools inspected: https://github.com/lifelong-robotic-vision/openloris-scene-tools at commit `ce6a4839f618bf036d3f3dbae14561bfc7413641`.
- OpenLORIS license name/status in HomeBrain: `CC BY-ND 4.0` observed on the official dataset page, with `license_review_status=pending_human_review`.
- Intended use: offline public robot-mounted RGB-D/IMU/odom/pose bridge for robot-frame BEV and action-sanity review.
- Required at runtime: no.
- Required for normal tests: no; tests use a tiny synthetic OpenLORIS-style fixture generated locally and explicitly remain replay-only.
- Mock/fallback status: no fake public dataset is substituted for missing real data. Synthetic fixtures test adapter semantics only and must not be reported as real dataset performance.
- Risk: CC BY-ND terms need human/legal review before training/product use, and public robot-mounted geometry is not action supervision unless calibrated robot-frame transforms and action sanity pass.

TUM RGB-D was also extended to try the robot-mounted `freiburg2_pioneer_slam` fallback.

- TUM source URL: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- Fallback sequence/download attempted: `freiburg2_pioneer_slam`, `rgbd_dataset_freiburg2_pioneer_slam.tgz`.
- License name/status in HomeBrain: `CC BY 4.0` observed previously, with `license_review_status=pending_human_review`.
- Risk: TUM Pioneer is robot-mounted, but HomeBrain still must not mark it action-valid without robot/base pose and camera-to-base semantics passing the robot-frame action contract.

## Goal 10B dependency update

Goal 10B approved bounded public dataset download/import and did not add product-runtime dependencies, training frameworks, simulator/middleware, or hardware control.

- TUM RGB-D source URL: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- TUM sequence/download used: `freiburg2_pioneer_slam`, `rgbd_dataset_freiburg2_pioneer_slam.tgz`, archive size recorded as `1.515` GB.
- TUM license name/status in HomeBrain: `CC BY 4.0`, with `license_review_status=pending_human_review`.
- TUM intended use: offline public robot-mounted RGB-D/camera-pose import review. It is not action supervision because HomeBrain has no non-assumed camera-to-base transform or robot/base pose semantics for this route.
- TUM runtime requirement: no.
- TUM mock/fallback status: no fake TUM data or transforms were substituted.
- TUM risk: route groundtruth remains preserved as dataset camera pose, not robot base pose; no TUM Pioneer robot-frame BEV/action labels were generated.

- OpenLORIS-Scene source URL: https://lifelong-robotic-vision.github.io/dataset/scene.html
- OpenLORIS official download docs: https://github.com/lifelong-robotic-vision/OpenLORIS-Scene/blob/master/download.md
- OpenLORIS package mirror used: https://huggingface.co/datasets/shixuesong/openloris-scene/tree/main/package
- OpenLORIS tools/static transform source inspected: https://github.com/lifelong-robotic-vision/openloris-scene-tools at commit `ce6a4839f618bf036d3f3dbae14561bfc7413641`.
- OpenLORIS sequence/package used: `cafe1-1_2`, selected package `package/cafe1-1_2-package.tar`, package size recorded as `6.954` GB.
- OpenLORIS license name/status in HomeBrain: `CC BY-ND 4.0`, with `license_review_status=pending_human_review`.
- OpenLORIS intended use: offline public robot-mounted RGB-D/IMU/odom/pose replay-only robot-frame BEV and action sanity review.
- OpenLORIS runtime requirement: no.
- OpenLORIS mock/fallback status: no fake public data, camera-to-base transform, base pose, odom, IMU, or command stream was substituted. Setup used an existing local 7-Zip executable only to extract nested `.7z` package contents.
- OpenLORIS risk: `CC BY-ND 4.0` needs human/legal review before any training/product use; action sanity passing is structural replay evidence only and remains `control_safe=false`.

## Goal 11A dependency update

Goal 11A did not add external datasets, model families, simulator/middleware, hardware-control libraries, or product-runtime dependencies.

- DINOv2-small was reused as an offline frozen feature teacher and run on OpenLORIS `cafe1-1_2`; status remains `pending_human_review`.
- PyTorch was reused for local SpatialMemoryNet and TrajectoryScorerNet training/inference; no new training framework was added.
- OpenLORIS-Scene `cafe1-1_2` was reused for replay-only robot-frame BEV/action supervision review; license remains `CC BY-ND 4.0` with `license_review_status=pending_human_review`.
- ActionLabelPack v3 deterministic expert labels were used only for local replay scorer training/eval; `product_training_approved=false`.
- The learned trajectory scorer checkpoint is local HomeBrain code, not an external model/checkpoint dependency.
- Required at runtime: no. DINO, SpatialMemoryNet, and TrajectoryScorerNet are used only in replay/modeld experiments in this goal.
- Mock/fallback status: tests use fake DINO features and synthetic/controlled packs; fake features remain marked `mock=true`, `synthetic=true`, and `real_perception=false`.
- Risk: the OpenLORIS action labels collapse to a single motion candidate (`straight_short`) in the learned-scorer subset, and OpenLORIS license terms still need human/legal review before any product training use.

## Goal 11B dependency update

Goal 11B did not add new external model families, datasets, simulator/middleware, hardware-control libraries, or product-runtime dependencies.

- DINOv2-small was reused as an offline frozen feature teacher on OpenLORIS `cafe1-1_2`, `office1-1_7`, and `corridor1-1`; status remains `pending_human_review`.
- PyTorch was reused for local SpatialMemoryNet and TrajectoryScorerNet training/eval; no new training framework was added.
- OpenLORIS-Scene was reused for replay-only public robot-mounted RGB-D/IMU/odom/pose benchmark routes. Evaluated packages: `package/cafe1-1_2-package.tar` (`6.954` GB), `package/office1-1_7-package.tar` (`9.889` GB), and `package/corridor1-1.7z` (`12.902` GB). `home1-1_5` was selected but kept out because robot-frame BEV refused missing camera intrinsics.
- OpenLORIS license name/status in HomeBrain remains `CC BY-ND 4.0` with `license_review_status=pending_human_review`.
- Intended use: local replay/eval benchmark for robot-frame BEV, spatial memory, future-motion labels, and candidate trajectory scoring. It is not product-approved training data.
- Required at runtime: no. DINO, SpatialMemoryNet, and TrajectoryScorerNet are used only for offline/replay Goal 11B experiments.
- Mock/fallback status: no fake public routes, intrinsics, transforms, odometry, or poses were substituted. Geometry-only DA3/TUM/phone frames were explicitly excluded from ActionLabelPack v4 action supervision.
- Risk: CC BY-ND terms still need human/legal review before any training/product use; true leave-one-route-out folds still diagnose held-out action-distribution concentration on corridor and office; all artifacts remain `replay_only=true`, `not_executed=true`, `control_safe=false`, and `product_training_approved=false`.

## Goal 12C dependency update

Goal 12C did not add new external datasets, model families, simulator/middleware, hardware-control libraries, or product-runtime dependencies.

- OpenLORIS-Scene was reclassified for HomeBrain policy as local PoC training/eval allowed, not product-approved training data.
- Required OpenLORIS flags: `poc_training_eval_allowed=true`, `product_training_approved=false`, `runtime_dependency=false`, `derived_dataset_redistribution_allowed=false`, `attribution_required=true`, and `license_name=CC BY-ND 4.0`.
- Intended use: local replay/eval proof-of-concept for robot-frame BEV, SpatialMemoryV1 memory validation, future-hidden-cell evaluation, deterministic occlusion stress, and true leave-one-route-out diagnostics.
- Required at runtime: no.
- Redistribution: OpenLORIS-derived generated datasets/artifacts/checkpoints/reports are not approved for redistribution and should remain under ignored generated-data paths such as `runs/` or local `data/public/`.
- Risk: this policy does not approve OpenLORIS for product training, runtime dependency, control safety, or distribution of derived datasets. Any product use still requires human/legal review.

## Goal 15A dependency update

Goal 15A added a SceneTeacherPack v0 interface and an optional VGGT-style backend
adapter. It did not add product-runtime dependencies, download real model weights,
or install external repositories.

- VGGT-family source/provenance: operator-supplied local `external/vggt`
  checkout or `HOMEBRAIN_VGGT_ADAPTER=module:function`; no official repo or
  checkpoint is selected by HomeBrain in this goal.
- License name/status in HomeBrain: `pending_human_review`; do not mark as
  runtime-safe, product-safe, or product-training-approved.
- Intended use: offline scene/geometry teacher for depth, point maps, camera
  parameters, point tracks, confidence/validity masks, and placeholder
  traversable/risk/dynamic masks.
- Required at runtime: no.
- Required for normal tests: no.
- Mock/fallback status: fake VGGT-style backend exists only when explicitly
  selected with `--backend fake`; artifacts are marked `mock=true`,
  `synthetic=true`, `real_perception=false`, `replay_only=true`,
  `not_executed=true`, `control_safe=false`, and
  `product_training_approved=false`.
- Real backend setup status: optional real check reported local VGGT unavailable
  because no local checkout/checkpoint/adapter was configured. HomeBrain does
  not clone repositories or auto-download weights during scene-teacher runs.
- Risk: SceneTeacherPack and derived scene-teacher BEVs are weak review geometry,
  not robot-frame action truth or control-safety evidence. Exact real
model/checkpoint licenses must be audited before any training/product use.

## Goal 15B dependency update

Goal 15B added a MoGe SceneTeacherPack v0 wrapper and an owned-route signal
audit. It did not add product-runtime dependencies, download real model weights,
install external repositories, train any model, or run policy/control loops.

- MoGe source/provenance: operator-supplied local `external/moge` checkout,
  `HOMEBRAIN_MOGE_DIR`, local checkpoint, or
  `HOMEBRAIN_MOGE_ADAPTER=module:function`; no official repo or checkpoint is
  selected by HomeBrain in this goal.
- License name/status in HomeBrain: `pending_human_review`; do not mark as
  runtime-safe, product-safe, or product-training-approved.
- Intended use: offline scene/geometry teacher producing depth, point maps,
  intrinsics, confidence/validity masks, and optional mask metadata for owned or
  license-approved indoor routes.
- Required at runtime: no.
- Required for normal tests: no.
- Mock/fallback status: fake MoGe backend exists only when explicitly selected
  with `--backend fake`; artifacts are marked `mock=true`, `synthetic=true`,
  `real_perception=false`, `replay_only=true`, `not_executed=true`,
  `control_safe=false`, and `product_training_approved=false`.
- Real backend setup status: optional real checks reported local MoGe and VGGT
  unavailable because no local checkout/checkpoint/adapter was configured.
  HomeBrain does not clone repositories or auto-download weights during
  scene-teacher runs.
- Risk: SceneTeacherPack signal audits are gates for review or geometry-pretrain
  candidates only. Exact real model/checkpoint licenses and owned-route
  provenance must be audited before any product training or runtime use.

## Goal 15C dependency update

Goal 15C did not add external dependencies, install MoGe, download checkpoints,
train models, run policies, or add product-runtime requirements.

- Owned/local frames used: `data/inbox/room_walk_001/frames`, imported only with
  explicit `--owned-or-license-approved` into
  `runs/goal15c_room_walk_001_route`.
- MoGe source/provenance remains operator-supplied local `external/moge`,
  `HOMEBRAIN_MOGE_DIR`, local checkpoint, or
  `HOMEBRAIN_MOGE_ADAPTER=module:function`; no official repo or checkpoint is
  selected by HomeBrain.
- License name/status in HomeBrain remains `pending_human_review`; no real MoGe
  code or weights were present to audit, and nothing is marked runtime-safe,
  product-safe, or product-training-approved.
- Required at runtime: no.
- Required for normal tests: no.
- Real backend setup status: blocked. The real MoGe run on the owned route
  reported missing local `external/moge` or `HOMEBRAIN_MOGE_DIR` /
  `HOMEBRAIN_MOGE_ADAPTER`; `external/models` contains only `da3`.
- Risk: there is still no real MoGe signal to evaluate. The audit gate now
  distinguishes `single_frame_geometry_pretrain_candidate` from
  `temporal_memory_pretrain_candidate`; temporal promotion requires real teacher
  extrinsics or route pose/odometry evidence. Any future real MoGe license and
  checkpoint provenance still require human review before product training or
  runtime use.

## Goal 16B dependency update

Goal 16B added a built-in adapter for the official MoGe v2 Python interface. It
did not install MoGe, download model weights, train models, run policies, or add
product-runtime requirements.

- MoGe code interface: `from moge.model.v2 import MoGeModel`.
- Default adapter: `homebrain.teachers.moge_official_adapter:run_scene_teacher`.
- Default model id when explicit download is allowed:
  `Ruicheng/moge-2-vits-normal`.
- Operator-selected model id source: `HOMEBRAIN_MOGE_MODEL_ID`.
- Local checkpoint source: `--checkpoint` or `HOMEBRAIN_MOGE_CHECKPOINT` when
  the path exists.
- Download policy: no default model download unless
  `HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1`; current Goal 16B probe did not set it.
- Current provenance: no importable MoGe package was present, no checkpoint was
  loaded, and no model id/checkpoint license could be audited from an actual
  artifact.
- License name/status in HomeBrain: `pending_human_review`; no product runtime,
  product-safety, or product-training approval.
- Intended use: offline scene/geometry teacher for owned or license-approved
  route logs only.
- Required at runtime: no.
- Required for normal tests: no; tests monkeypatch a fake `moge.model.v2`
  module and never download weights.
- Mock/fallback status: fake MoGe remains explicit review-only and never
  trainable. `backend=real` has no fake fallback.
- Risk: exact MoGe code/checkpoint/model-card license and provenance still need
  human review before any product training or runtime use.

## Goal 16C dependency update

Goal 16C installed and ran real official MoGe for the first owned-frame
SceneTeacherPack. It did not edit HomeBrain source, train models, run policies,
or add product-runtime requirements.

- MoGe code source URL: https://github.com/microsoft/MoGe.
- Local code path: ignored checkout `external/moge`.
- Code commit used: `07444410f1e33f402353b99d6ccd26bd31e469e8`.
- Python package install: `python -m pip install -e external\moge`, resulting in
  editable `moge==2.0.0` at
  `C:\Users\Asav\source\repos\homebrain\external\moge`.
- Observed code license: `MIT` from `pyproject.toml` / `pip show moge`; the
  repository `LICENSE` file also includes Apache-2.0 text, so human review is
  still required before product training/runtime approval.
- MoGe model source: Hugging Face `Ruicheng/moge-2-vits-normal`.
- Model source URL: https://huggingface.co/Ruicheng/moge-2-vits-normal.
- Model revision used: `679230677b4d282c6f304189a93e98e14f085902`.
- Observed model-card metadata: Hugging Face API returned `license: mit` and tag
  `license:mit`.
- Local model cache path:
  `C:\Users\Asav\.cache\huggingface\hub\models--Ruicheng--moge-2-vits-normal\snapshots\679230677b4d282c6f304189a93e98e14f085902\model.pt`.
- Local model file size observed: `140550416` bytes. The model was not committed
  to HomeBrain and remains outside tracked source.
- Model-source configuration used for the probe:
  `HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1`; no `HOMEBRAIN_MOGE_MODEL_ID`,
  `HOMEBRAIN_MOGE_CHECKPOINT`, or `--checkpoint` was used.
- Transitive packages installed from MoGe requirements include `utils3d` from
  `https://github.com/EasternJournalist/utils3d.git` at
  `3fab839f0be9931dac7c8488eb0e1600c236e183`, `pipeline` from
  `https://github.com/EasternJournalist/pipeline.git` at
  `866f059d2a05cde05e4a52211ec5051fd5f276d6`, and PyPI packages such as
  `trimesh`, `gradio`, `fastapi`, `pydantic`, `moderngl`, and `orjson`.
- Intended use: offline scene/geometry teacher on owned or
  license-approved route logs.
- Required at runtime: no.
- Required for normal tests: no.
- Mock/fallback status: `backend=real` used no fake fallback; fake MoGe remains
  explicit review-only when selected by tests.
- Goal 16C artifact status: real non-mock SceneTeacherPack exists at
  `runs/goal16c_moge_real_probe/route/teacher_artifacts/moge_scene_v0_real` and
  signal audit reports `next_allowed_use=single_frame_geometry_pretrain_candidate`.
- Safety/license status: `control_safe=false`, `product_training_approved=false`,
  `robot_frame_truth=false`, `action_supervision_ok=false`, and
  `license_review_status=pending_human_review`.
- Risk: output has no teacher temporal extrinsics, no route pose/odom evidence,
  no measured camera-to-base transform, no point tracks, uniform confidence, and
  placeholder floor/obstacle masks. It is usable only as review-gated
  single-frame geometry pretraining evidence until deeper audit and human/legal
  approval.

## Goal 17A dependency update

Goal 17A did not add external dependencies, install new packages, download new
datasets, train models, run policies, or add product-runtime requirements.

- MoGe code/checkpoint provenance reused the Goal 16C setup:
  `external/moge` at commit `07444410f1e33f402353b99d6ccd26bd31e469e8` and
  Hugging Face model `Ruicheng/moge-2-vits-normal` revision
  `679230677b4d282c6f304189a93e98e14f085902`.
- Model-source configuration reused the explicitly allowed Goal 16C setting
  `HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1`. No new model id or checkpoint source was
  introduced.
- Public data reused existing local OpenLORIS-Scene routes and
  SpatialTrainPacks under `runs/goal11b_nightly/`; no redownload was performed.
- OpenLORIS license/status remains `CC BY-ND 4.0` observed, local
  research/replay only, product-training approval pending, runtime dependency
  false, and derived dataset redistribution not approved.
- Goal 17A generated real MoGe SceneTeacherPacks, QA reports, comparison reports,
  and contact sheets under `runs/goal17a_moge_openloris_teacher_quality/`.
- Safety/license status remains `control_safe=false`,
  `product_training_approved=false`, `moge_robot_frame_truth=false`,
  `action_supervision_ok=false`, and `license_review_status=pending_human_review`.
- Risk: the comparison supports only a local/replay single-frame geometry
  candidate gate. MoGe is not robot-frame truth, action supervision, temporal
  memory evidence, control-safety evidence, or product-training-approved data.
