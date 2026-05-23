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
| MoGe / point-map model | geometry teacher | UNVERIFIED | No | Verify official license. |
| VGGT-family | offline multi-view geometry teacher | UNVERIFIED | No | Some checkpoints may differ in license. Verify exact checkpoint. |
| SAM-family video segmentation | mask/dynamic-object teacher | UNVERIFIED | No | Verify exact version. |
| GNM / ViNT / NoMaD | navigation prior / baseline | UNVERIFIED | No | Verify code/checkpoint/dataset license. |
| AI2-THOR / ProcTHOR | synthetic control data | UNVERIFIED | No | Use only after license check. |
| iGibson | synthetic control data | UNVERIFIED | No | Use only after license check. |
| ARKitScenes | public indoor geometry data | UNVERIFIED | No | Verify dataset terms. |
| ScanNet / ScanNet++ | public indoor geometry data | UNVERIFIED | No | Verify dataset terms. |
| TUM RGB-D SLAM Dataset | public RGB-D / pose geometry anchor | UNVERIFIED / pending_human_review | No | Official page lists CC BY 4.0; keep pending human review before product/training approval. |
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
