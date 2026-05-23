# Expert Action Data Plan

HomeBrain does not have expert robot cleaning demonstrations yet. Future action supervision should be assembled from multiple sources, each with explicit provenance and license review. Nothing in this plan is commercially cleared until `LICENSE_AUDIT.md` or a later legal review says so.

## Candidate Supervision Sources

| source_family | candidate_source | supervision_type | license_review_status | intended_use | notes |
| --- | --- | --- | --- | --- | --- |
| public RGB-D/pose datasets | ScanNet, ScanNet++, NYU Depth V2, TUM RGB-D, 7-Scenes, Matterport-family indoor data | depth, pose, geometry, local BEV pseudo-labels | pending_human_review | pretrain or evaluate geometry and pose heads | Use only after license and redistribution terms are reviewed; many datasets are research-only or require registration. |
| dynamic indoor datasets | JRDB-style robot logs, Ego4D indoor clips where applicable, human-object interaction datasets such as BEHAVE/InterCap-style data | dynamic object masks, temporary obstacle/risk labels, human motion context | pending_human_review | teach dynamic-risk handling and do-not-permanently-map behavior | These are candidate sources for visual dynamics, not cleaning action ground truth. |
| weak teacher labels | Depth Pro, Depth Anything-family, DINO-family features, SAM-family masks, optical-flow or tracking teachers | offline pseudo-labels for depth, semantics, motion, uncertainty, and review queues | pending_human_review | produce weak labels and features for student training after QA | Teacher outputs must remain marked as weak labels; no runtime safety claim follows from them. |
| navigation foundation priors | GNM, ViNT, NoMaD-style navigation models | pseudo-action priors, waypoint preferences, trajectory ranking hints | pending_human_review | provide candidate trajectory priors for offline comparison and distillation | No integration is approved here; future work must label outputs as pseudo-action priors, not expert control. |
| algorithmic coverage planners | deterministic grid coverage, frontier selection, boustrophedon-like sweeps, obstacle-aware candidate scoring on controlled maps | coverage action labels, candidate trajectory scores, oracle-like planner baselines | internal_algorithm_review_needed | generate controlled supervision where geometry and objective are known | These labels can be useful for coverage behavior, but only in the simplified grid worlds they are generated from. |
| future real robot logs | HomeBrain hardware logs with RGB/IR/IMU/wheel encoders, commands, interventions, stuck events, docking attempts | real action logs, recovery labels, near-collision labels, coverage outcomes, operator corrections | not_available_until_hardware_and_consent_policy | main production-directed dataset once hardware exists | Must include consent, retention, privacy, and deletion policy before collection beyond local experiments. |

## Required Dataset Fields

Every future action-data entry should carry:

```text
source_family
source_name
supervision_type
license_review_status
provenance
weak_label
control_safe
runtime_dependency
review_required
```

## Integration Order

1. Keep packaging and QA deterministic for existing BEV labels.
2. Add human review notes and calibration status to weak-label packages.
3. Add algorithmic coverage-planner labels on controlled grids before learned action heads.
4. Add public RGB-D/pose and dynamic visual datasets only after license review.
5. Add GNM/ViNT/NoMaD pseudo-action priors only as offline teachers, never as unreviewed control.
6. Replace pseudo-action labels with real robot logs once hardware and consent policy exist.

