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
| Apple Depth Pro | metric depth teacher | UNVERIFIED | No | Verify official license. |
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
