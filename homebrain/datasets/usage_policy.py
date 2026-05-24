from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from homebrain.datasets.openloris_scene import OPENLORIS_LICENSE_NAME


DATASET_USAGE_POLICY_SCHEMA_VERSION = "homebrain.dataset_usage_policy.v0"


@dataclass(frozen=True)
class DatasetUsagePolicy:
    dataset_name: str
    license_name: str
    license_review_status: str
    poc_training_eval_allowed: bool
    product_training_approved: bool
    runtime_dependency: bool
    derived_dataset_redistribution_allowed: bool
    attribution_required: bool
    generated_artifacts_gitignored: bool
    notes: str
    schema_version: str = DATASET_USAGE_POLICY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OPENLORIS_USAGE_POLICY = DatasetUsagePolicy(
    dataset_name="OpenLORIS-Scene",
    license_name=OPENLORIS_LICENSE_NAME,
    license_review_status="poc_allowed_product_pending_human_review",
    poc_training_eval_allowed=True,
    product_training_approved=False,
    runtime_dependency=False,
    derived_dataset_redistribution_allowed=False,
    attribution_required=True,
    generated_artifacts_gitignored=True,
    notes=(
        "Allowed for local HomeBrain proof-of-concept training and evaluation only. "
        "Do not treat OpenLORIS-derived data, checkpoints, reports, or labels as product-approved, "
        "runtime-required, control-safe, or redistributable derived datasets."
    ),
)


def openloris_usage_policy() -> dict[str, Any]:
    return OPENLORIS_USAGE_POLICY.to_dict()


def require_poc_allowed(dataset_name: str) -> dict[str, Any]:
    policy = dataset_usage_policy(dataset_name)
    if not policy["poc_training_eval_allowed"]:
        raise ValueError(f"{dataset_name} is not allowed for local PoC training/eval")
    return policy


def require_product_training_approved(dataset_name: str) -> dict[str, Any]:
    policy = dataset_usage_policy(dataset_name)
    if not policy["product_training_approved"]:
        raise ValueError(f"{dataset_name} is not approved for product training")
    return policy


def require_runtime_dependency_allowed(dataset_name: str) -> dict[str, Any]:
    policy = dataset_usage_policy(dataset_name)
    if not policy["runtime_dependency"]:
        raise ValueError(f"{dataset_name} is not approved as a runtime dependency")
    return policy


def dataset_usage_policy(dataset_name: str) -> dict[str, Any]:
    normalized = dataset_name.strip().lower().replace("_", "-")
    if normalized in {"openloris", "openloris-scene", "openloris-scene-dataset"}:
        return openloris_usage_policy()
    raise KeyError(f"unknown dataset usage policy: {dataset_name!r}")
