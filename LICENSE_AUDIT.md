# LICENSE_AUDIT.md - POC Data and Model Policy

HomeBrain is currently a local proof of concept. We can use public datasets and
open-weight/open-source models for local training, eval, and teacher labeling
when provenance is recorded. This file is not a legal approval record for a
shipping robot.

## Status Meanings

- `POC_OK`: allowed for local proof-of-concept training/eval.
- `TEACHER_ONLY`: allowed as an offline teacher, not runtime control.
- `PRODUCT_REVIEW_LATER`: do not claim commercial/product approval yet.
- `AVOID`: do not use.

POC use still requires:

- source URL or local provenance;
- model/dataset id and revision when known;
- generated artifacts kept under ignored local paths when redistribution is
  unclear;
- mock/synthetic artifacts marked explicitly;
- `control_safe=false` until real hardware safety gates exist.

## Current Sources

| Name | POC status | Product/runtime status | Notes |
| --- | --- | --- | --- |
| OpenLORIS-Scene | `POC_OK` | `PRODUCT_REVIEW_LATER` | Public robot-mounted RGB-D/IMU/odom/pose bridge. Good for local replay/eval, robot-frame BEV experiments, future-motion labels, and policy POC. Do not redistribute derived datasets unless separately cleared. |
| TUM RGB-D | `POC_OK` | `PRODUCT_REVIEW_LATER` | Good for RGB-D/pose geometry anchors. Not automatically robot action truth. |
| DINO-family / DINOv2 | `TEACHER_ONLY` | `PRODUCT_REVIEW_LATER` | Offline frozen feature teacher for student training. Not a required runtime dependency. |
| MoGe | `TEACHER_ONLY` | `PRODUCT_REVIEW_LATER` | Offline metric geometry/point-map teacher. Current local setup observed official MoGe and `Ruicheng/moge-2-vits-normal`; use for POC geometry, not control safety. |
| Depth Anything family | `TEACHER_ONLY` | `PRODUCT_REVIEW_LATER` | Offline depth/geometry teacher. |
| Depth Pro | `TEACHER_ONLY` | `PRODUCT_REVIEW_LATER` | Offline metric-depth teacher. |
| VGGT-family | `TEACHER_ONLY` | `PRODUCT_REVIEW_LATER` | Optional local multi-view geometry teacher. |
| SAM-family | `TEACHER_ONLY` | `PRODUCT_REVIEW_LATER` | Candidate mask/dynamic-object teacher. |
| GNM / ViNT / NoMaD-style navigation priors | `TEACHER_ONLY` | `PRODUCT_REVIEW_LATER` | Candidate offline pseudo-action priors only. |
| AI2-THOR / ProcTHOR / iGibson | `POC_OK` after setup review | `PRODUCT_REVIEW_LATER` | Simulation may provide stress/action labels, not core truth. |

## Practical Rule

For this repo phase, do not block useful local experiments because a dataset is
not product-approved. Block only when provenance is missing, a source forbids
the intended local use, artifacts are being redistributed without clearance, or
the code tries to claim runtime/control safety.
