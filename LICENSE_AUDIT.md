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
| DINO-family visual model | dense features / teacher | UNVERIFIED | No | Verify exact repo + weight license. |
| Depth Anything family | geometry/depth teacher | UNVERIFIED | No | Verify exact version and checkpoint license. |
| Apple Depth Pro | optional metric depth teacher | UNVERIFIED / pending_human_review | No | Official repo: https://github.com/apple/ml-depth-pro. License terms require human review before product use. |
| MoGe / point-map model | geometry teacher | UNVERIFIED | No | Verify official license. |
| VGGT-family | offline multi-view geometry teacher | UNVERIFIED | No | Some checkpoints may differ in license. Verify exact checkpoint. |
| SAM-family video segmentation | mask/dynamic-object teacher | UNVERIFIED | No | Verify exact version. |
| GNM / ViNT / NoMaD | navigation prior / baseline | UNVERIFIED | No | Verify code/checkpoint/dataset license. |
| AI2-THOR / ProcTHOR | synthetic control data | UNVERIFIED | No | Use only after license check. |
| iGibson | synthetic control data | UNVERIFIED | No | Use only after license check. |
| ARKitScenes | public indoor geometry data | UNVERIFIED | No | Verify dataset terms. |
| ScanNet / ScanNet++ | public indoor geometry data | UNVERIFIED | No | Verify dataset terms. |
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
