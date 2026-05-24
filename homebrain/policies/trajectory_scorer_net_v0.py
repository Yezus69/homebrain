from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from homebrain.data.spatial_dataset import load_example_npz, read_json, write_json
from homebrain.messages.schema import BrainOutputEvent, JsonDict, deterministic_json
from homebrain.policies.candidate_trajectories import CandidateTrajectory, candidates_hash, generate_default_candidates
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev
from homebrain.replay.segment_log import read_events

TRAJECTORY_SCORER_V0_SOURCE = "trajectory_scorer_net_v0"
TRAJECTORY_SCORER_V0_CHECKPOINT_VERSION = "homebrain.trajectory_scorer_v0.checkpoint.v0"
TRAJECTORY_SCORER_TRAIN_SCHEMA_VERSION = "homebrain.trajectory_scorer_v0_train_metrics.v0"
TRAJECTORY_SCORER_EVAL_SCHEMA_VERSION = "homebrain.trajectory_scorer_v0_eval_metrics.v0"
COLLISION_THRESHOLD = 0.35
UNKNOWN_BLOCK_THRESHOLD = 0.95

BEV_INPUT_CHANNELS: tuple[str, ...] = (
    "free",
    "occupied",
    "unknown",
    "traversable",
    "risky",
    "confidence",
    "uncertainty",
)

CANDIDATE_FEATURE_NAMES: tuple[str, ...] = (
    "linear_velocity_norm",
    "angular_velocity_norm",
    "duration_norm",
    "is_stop",
    "is_rotation_only",
    "footprint_fraction",
    "occupied_max",
    "occupied_mean",
    "unknown_max",
    "unknown_mean",
    "uncertainty_mean",
    "free_mean",
    "traversable_mean",
    "coverage_gain_fraction",
    "covered_fraction",
    "smoothness_norm",
    "terminal_forward_norm",
    "terminal_left_norm",
    "terminal_yaw_norm",
)

GEOMETRY_ONLY_SOURCE_FAMILIES: frozenset[str] = frozenset(
    {
        "phone_or_teacher_estimated_geometry",
        "public_rgbd_camera_pose_geometry",
    }
)


@dataclass(frozen=True)
class TrajectoryScorerNetConfig:
    bev_channels: int = len(BEV_INPUT_CHANNELS)
    candidate_feature_dim: int = len(CANDIDATE_FEATURE_NAMES)
    hidden_dim: int = 64

    def to_dict(self) -> JsonDict:
        return {
            "bev_channels": int(self.bev_channels),
            "candidate_feature_dim": int(self.candidate_feature_dim),
            "hidden_dim": int(self.hidden_dim),
            "bev_input_channels": list(BEV_INPUT_CHANNELS),
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TrajectoryScorerNetConfig":
        return cls(
            bev_channels=int(data.get("bev_channels", len(BEV_INPUT_CHANNELS))),
            candidate_feature_dim=int(data.get("candidate_feature_dim", len(CANDIDATE_FEATURE_NAMES))),
            hidden_dim=int(data.get("hidden_dim", 64)),
        )


class TrajectoryScorerNetV0(nn.Module):
    """Small replay-only candidate ranker.

    The model scores fixed candidate trajectories from explicit BEV/candidate
    features. Higher logits mean "more expert-like"; they are not motor
    commands and must stay behind replay/eval gates.
    """

    def __init__(self, config: TrajectoryScorerNetConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(config.bev_channels, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(config.candidate_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 32, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        bev: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, candidate_count, _feature_dim = candidate_features.shape
        bev_context = self.bev_encoder(bev).view(batch_size, 32)
        candidate_context = self.candidate_encoder(candidate_features)
        repeated_context = bev_context[:, None, :].expand(batch_size, candidate_count, 32)
        logits = self.head(torch.cat([candidate_context, repeated_context], dim=2)).squeeze(2)
        if candidate_mask is not None:
            logits = logits.masked_fill(candidate_mask <= 0.0, -1.0e9)
        return logits


@dataclass(frozen=True)
class ActionScorerSample:
    record: JsonDict
    source_example_path: Path
    action_example_path: Path
    bev: LocalBev
    bev_tensor: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    total_expert_score: np.ndarray
    target_logits: np.ndarray
    selected_index: int
    source_weight: float


class ActionLabelScorerDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        action_pack: str | Path,
        *,
        split: str | None = "train",
        source_names: set[str] | None = None,
        source_families: set[str] | None = None,
        bev_source: str = "oracle",
        modeld_dir: str | Path | None = None,
        val_modulus: int = 5,
        max_examples: int | None = None,
        allow_geometry_only: bool = False,
    ) -> None:
        if split not in {None, "train", "val"}:
            raise ValueError("split must be 'train', 'val', or None")
        if bev_source not in {"oracle", "model"}:
            raise ValueError("bev_source must be 'oracle' or 'model'")
        if val_modulus < 2:
            raise ValueError("val_modulus must be at least 2")
        if bev_source == "model" and modeld_dir is None:
            raise ValueError("modeld_dir is required for model-BEV action scorer data")

        self.root = Path(action_pack)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("package_type") != "ActionLabelPack":
            raise ValueError(f"expected ActionLabelPack manifest: {self.root}")
        if self.manifest.get("control_safe") is not False:
            raise ValueError("ActionLabelPack must remain control_safe=false")
        self.bev_source = bev_source
        self.split = split
        self.val_modulus = val_modulus
        self.source_roots = _resolve_source_roots(self.root, self.manifest)
        self.candidate_ids = _string_list(self.manifest.get("candidate_ids"), "candidate_ids")
        self.grid_shape = _grid_shape(self.manifest)
        self.meters_per_cell = _positive_float(self.manifest.get("meters_per_cell"), 0.05)
        self.robot_radius_m = _positive_float(self.manifest.get("robot_radius_m"), 0.18, allow_zero=True)
        self.candidates = generate_default_candidates(
            grid_shape=self.grid_shape,
            meters_per_cell=self.meters_per_cell,
            robot_radius_m=self.robot_radius_m,
        )
        if [candidate.id for candidate in self.candidates] != self.candidate_ids:
            raise ValueError("ActionLabelPack candidate ids do not match default candidate generator")
        manifest_candidate_hash = self.manifest.get("candidate_hash")
        if isinstance(manifest_candidate_hash, str) and manifest_candidate_hash != candidates_hash(self.candidates):
            raise ValueError("ActionLabelPack candidate hash does not match generated candidates")

        examples = self.manifest.get("examples")
        if not isinstance(examples, list):
            raise ValueError("ActionLabelPack manifest examples must be a list")
        model_bev_map = _load_model_bev_map(modeld_dir) if bev_source == "model" else {}
        filtered_records = _filter_records(
            examples,
            split=split,
            val_modulus=val_modulus,
            source_names=source_names,
            source_families=source_families,
            allow_geometry_only=allow_geometry_only,
        )
        if max_examples is not None:
            filtered_records = filtered_records[:max_examples]
        if not filtered_records:
            raise ValueError("no action-label examples matched the scorer dataset filters")

        self.samples = self._load_samples(filtered_records, model_bev_map=model_bev_map)
        self.source_distribution = _distribution(str(sample.record.get("source_name", "unknown")) for sample in self.samples)
        self.source_family_distribution = _distribution(
            str(sample.record.get("source_family", "unknown")) for sample in self.samples
        )
        self.selected_distribution = _distribution(self.candidate_ids[sample.selected_index] for sample in self.samples)
        self.feature_alignment = {
            "bev_source": bev_source,
            "example_count": len(self.samples),
            "model_bev_frame_count": len(model_bev_map),
            "missing_model_bev_count": 0,
            "source_dirs": [root.as_posix() for root in self.source_roots],
        }
        self.excluded_geometry_only_sources = _excluded_geometry_only_sources(self.manifest)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "bev": torch.from_numpy(sample.bev_tensor.copy()),
            "candidate_features": torch.from_numpy(sample.candidate_features.copy()),
            "candidate_mask": torch.from_numpy(sample.candidate_mask.copy()),
            "target_logits": torch.from_numpy(sample.target_logits.copy()),
            "total_expert_score": torch.from_numpy(sample.total_expert_score.copy()),
            "selected_index": torch.tensor(sample.selected_index, dtype=torch.long),
            "source_weight": torch.tensor(sample.source_weight, dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }

    def _load_samples(
        self,
        records: list[tuple[int, JsonDict]],
        *,
        model_bev_map: dict[tuple[str, int], LocalBev],
    ) -> list[ActionScorerSample]:
        samples: list[ActionScorerSample] = []
        coverage_by_key: dict[tuple[str, str, str], np.ndarray] = {}
        for _global_index, record in records:
            action_example_path = self.root / str(record["example_path"])
            action_example = load_example_npz(action_example_path)
            _validate_action_example_flags(action_example, action_example_path)
            source_example_path = self._source_example_path(str(record["source_ref"]))
            if self.bev_source == "model":
                key = (str(record["sequence_id"]), int(record["frame_id"]))
                bev = model_bev_map.get(key)
                if bev is None:
                    raise KeyError(
                        "missing model BEV for action-label frame "
                        f"{record.get('sequence_id')}:{record.get('frame_id')}; "
                        "filter to sources covered by the modeld run"
                    )
            else:
                bev = _load_oracle_bev(source_example_path)
            coverage_key = (
                str(record.get("source_name", "unknown")),
                str(record.get("sequence_id", "unknown")),
                str(record.get("camera_id", "unknown")),
            )
            covered = coverage_by_key.setdefault(coverage_key, np.zeros(bev.shape, dtype=np.bool_))
            candidate_features = candidate_feature_matrix(
                bev=bev,
                candidates=self.candidates,
                coverage_mask=covered,
            )
            _update_coverage_mask(covered, bev)
            total_scores = np.asarray(action_example["total_expert_score"], dtype=np.float32)
            selected_by_expert = np.asarray(action_example["selected_by_expert"], dtype=np.bool_)
            if total_scores.shape != (len(self.candidates),):
                raise ValueError(f"total_expert_score has wrong shape: {action_example_path}")
            if selected_by_expert.shape != (len(self.candidates),) or int(np.count_nonzero(selected_by_expert)) != 1:
                raise ValueError(f"selected_by_expert must select exactly one candidate: {action_example_path}")
            selected_index = int(np.flatnonzero(selected_by_expert)[0])
            samples.append(
                ActionScorerSample(
                    record=record,
                    source_example_path=source_example_path,
                    action_example_path=action_example_path,
                    bev=bev,
                    bev_tensor=bev_tensor(bev),
                    candidate_features=candidate_features,
                    candidate_mask=np.ones((len(self.candidates),), dtype=np.float32),
                    total_expert_score=total_scores,
                    target_logits=_target_logits_from_expert_scores(total_scores),
                    selected_index=selected_index,
                    source_weight=_record_source_weight(record),
                )
            )
        return samples

    def _source_example_path(self, source_ref: str) -> Path:
        matches = [root / source_ref for root in self.source_roots if (root / source_ref).exists()]
        if not matches:
            raise FileNotFoundError(f"could not resolve ActionLabelPack source_ref {source_ref!r}")
        return matches[0]


def bev_tensor(bev: LocalBev) -> np.ndarray:
    bev.validate()
    free = _prob(bev.free)
    occupied = _prob(bev.occupied)
    unknown = _prob(bev.unknown)
    traversable = _prob(bev.traversable) if bev.traversable is not None else free
    risky = _prob(bev.risky) if bev.risky is not None else occupied
    if bev.confidence is not None:
        confidence = _prob(bev.confidence)
    elif bev.uncertainty is not None:
        confidence = np.float32(1.0) - _prob(bev.uncertainty)
    else:
        confidence = np.float32(1.0) - unknown
    if bev.uncertainty is not None:
        uncertainty = _prob(bev.uncertainty)
    else:
        uncertainty = np.float32(1.0) - confidence
    return np.stack(
        [free, occupied, unknown, traversable, risky, confidence, uncertainty],
        axis=0,
    ).astype(np.float32)


def candidate_feature_matrix(
    *,
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
    coverage_mask: np.ndarray | None,
) -> np.ndarray:
    bev.validate()
    free = _prob(bev.free)
    occupied = np.maximum(_prob(bev.occupied), _prob(bev.risky) if bev.risky is not None else 0.0)
    unknown = _prob(bev.unknown)
    traversable = _prob(bev.traversable) if bev.traversable is not None else free
    if bev.uncertainty is not None:
        uncertainty = _prob(bev.uncertainty)
    elif bev.confidence is not None:
        uncertainty = np.float32(1.0) - _prob(bev.confidence)
    else:
        uncertainty = unknown
    if coverage_mask is None:
        coverage = np.zeros(bev.shape, dtype=np.bool_)
    else:
        coverage = np.asarray(coverage_mask, dtype=np.bool_)
        if coverage.shape != bev.shape:
            raise ValueError(f"coverage mask shape {coverage.shape} does not match BEV {bev.shape}")

    rows, cols = bev.shape
    grid_area = float(max(rows * cols, 1))
    values: list[list[float]] = []
    for candidate in candidates:
        cells = candidate.footprint_cells
        occupied_values = _cell_values(occupied, cells, default=1.0)
        unknown_values = _cell_values(unknown, cells, default=1.0)
        uncertainty_values = _cell_values(uncertainty, cells, default=1.0)
        free_values = _cell_values(free, cells, default=0.0)
        traversable_values = _cell_values(traversable, cells, default=0.0)
        covered_values = _cell_values(coverage.astype(np.float32), cells, default=0.0)
        coverage_gain = 0.0
        for row, col in cells:
            if 0 <= row < rows and 0 <= col < cols and free[row, col] >= 0.5 and not bool(coverage[row, col]):
                coverage_gain += 1.0
        linear = float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0))
        angular = float(candidate.cmd_vel_proxy.get("angular_velocity_radps", 0.0))
        terminal = candidate.poses[-1] if candidate.poses else None
        terminal_forward = 0.0 if terminal is None else terminal.x_m / max(rows, 1)
        terminal_left = 0.0 if terminal is None else terminal.y_m / max(cols, 1)
        terminal_yaw = 0.0 if terminal is None else terminal.yaw_rad / math.pi
        is_rotation_only = abs(linear) < 1.0e-6 and abs(angular) > 1.0e-6
        smoothness = abs(angular) * candidate.duration_s + (0.05 if is_rotation_only else 0.0)
        values.append(
            [
                linear / 0.30,
                angular / 1.25,
                candidate.duration_s / 2.5,
                1.0 if candidate.id == "stop" else 0.0,
                1.0 if is_rotation_only else 0.0,
                len(cells) / grid_area,
                float(np.max(occupied_values)),
                float(np.mean(occupied_values)),
                float(np.max(unknown_values)),
                float(np.mean(unknown_values)),
                float(np.mean(uncertainty_values)),
                float(np.mean(free_values)),
                float(np.mean(traversable_values)),
                coverage_gain / grid_area,
                float(np.mean(covered_values)),
                smoothness / 1.5,
                terminal_forward,
                terminal_left,
                terminal_yaw,
            ]
        )
    return np.asarray(values, dtype=np.float32)


def train_trajectory_scorer_v0(
    *,
    action_pack: str | Path,
    out_dir: str | Path,
    source_names: set[str] | None = None,
    source_families: set[str] | None = None,
    max_steps: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1.0e-3,
    device_name: str | None = None,
    max_examples: int | None = None,
) -> JsonDict:
    torch.manual_seed(11)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = ActionLabelScorerDataset(
        action_pack,
        split="train",
        source_names=source_names,
        source_families=source_families,
        max_examples=max_examples,
    )
    val_dataset = ActionLabelScorerDataset(
        action_pack,
        split="val",
        source_names=source_names,
        source_families=source_families,
        max_examples=max_examples,
    )
    config = TrajectoryScorerNetConfig(
        bev_channels=int(train_dataset.samples[0].bev_tensor.shape[0]),
        candidate_feature_dim=int(train_dataset.samples[0].candidate_features.shape[1]),
    )
    model = TrajectoryScorerNetV0(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=_torch_generator(),
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    initial_train = evaluate_trajectory_scorer_loader(model, train_eval_loader, device=device)
    initial_val = evaluate_trajectory_scorer_loader(model, val_loader, device=device)
    iterator = cycle(train_loader)
    model.train()
    for _step in range(max_steps):
        batch = _batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["bev"], batch["candidate_features"], batch["candidate_mask"])
        losses = trajectory_scorer_loss(logits, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    final_train = evaluate_trajectory_scorer_loader(model, train_eval_loader, device=device)
    final_val = evaluate_trajectory_scorer_loader(model, val_loader, device=device)
    train_loss_start = float(initial_train["loss"])
    train_loss_end = float(final_train["loss"])
    val_loss_start = float(initial_val["loss"])
    val_loss = float(final_val["loss"])
    loss_reduction_ratio = (train_loss_start - train_loss_end) / train_loss_start if train_loss_start > 0.0 else 0.0
    metrics: JsonDict = {
        "schema_version": TRAJECTORY_SCORER_TRAIN_SCHEMA_VERSION,
        "action_pack": Path(action_pack).as_posix(),
        "out_dir": Path(out_dir).as_posix(),
        "device": str(device),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "example_count": len(train_dataset),
        "val_example_count": len(val_dataset),
        "source_filters": {
            "source_names": sorted(source_names) if source_names else [],
            "source_families": sorted(source_families) if source_families else [],
        },
        "source_distribution": train_dataset.source_distribution,
        "val_source_distribution": val_dataset.source_distribution,
        "source_family_distribution": train_dataset.source_family_distribution,
        "excluded_geometry_only_sources_from_action_training": train_dataset.excluded_geometry_only_sources,
        "candidate_ids": list(train_dataset.candidate_ids),
        "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
        "bev_input_channels": list(BEV_INPUT_CHANNELS),
        "candidate_hash": candidates_hash(train_dataset.candidates),
        "train_loss_start": train_loss_start,
        "train_loss_end": train_loss_end,
        "loss_reduction_ratio": float(loss_reduction_ratio),
        "val_loss_start": val_loss_start,
        "val_loss": val_loss,
        "train_top1_action_agreement": float(final_train["top1_action_agreement"]),
        "val_top1_action_agreement": float(final_val["top1_action_agreement"]),
        "val_rank_correlation_or_proxy": float(final_val["rank_correlation_or_proxy"]),
        "random_top1_action_agreement": float(1.0 / max(len(train_dataset.candidate_ids), 1)),
        "beats_random": bool(final_val["top1_action_agreement"] > 1.0 / max(len(train_dataset.candidate_ids), 1)),
        "representation_pretraining_only": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train_metrics.json", metrics, pretty=True)
    write_json(
        output / "config.json",
        {
            "schema_version": "homebrain.trajectory_scorer_v0_train_config.v0",
            "model_config": config.to_dict(),
            "action_pack": Path(action_pack).as_posix(),
            "source_filters": metrics["source_filters"],
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        },
        pretty=True,
    )
    save_trajectory_scorer_checkpoint(
        output / "checkpoint.pt",
        model.cpu(),
        metadata={
            "action_pack": Path(action_pack).as_posix(),
            "source_filters": metrics["source_filters"],
            "candidate_ids": list(train_dataset.candidate_ids),
            "candidate_hash": candidates_hash(train_dataset.candidates),
            "meters_per_cell": train_dataset.meters_per_cell,
            "robot_radius_m": train_dataset.robot_radius_m,
            "grid_shape": list(train_dataset.grid_shape),
            "excluded_geometry_only_sources_from_action_training": train_dataset.excluded_geometry_only_sources,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        },
        metrics=metrics,
    )
    return metrics


def eval_trajectory_scorer_v0(
    *,
    checkpoint: str | Path,
    action_pack: str | Path,
    out_path: str | Path,
    source_names: set[str] | None = None,
    source_families: set[str] | None = None,
    bev_source: str = "oracle",
    modeld_dir: str | Path | None = None,
    split: str | None = "val",
    device_name: str | None = None,
    max_examples: int | None = None,
    viz_dir: str | Path | None = None,
    max_viz_frames: int = 24,
) -> JsonDict:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_trajectory_scorer_checkpoint(checkpoint, map_location=device)
    model.to(device)
    dataset = ActionLabelScorerDataset(
        action_pack,
        split=split,
        source_names=source_names,
        source_families=source_families,
        bev_source=bev_source,
        modeld_dir=modeld_dir,
        max_examples=max_examples,
    )
    predictions, inference_fps = _predict_dataset(model, dataset, device=device)
    metrics = _prediction_metrics(
        predictions=predictions,
        dataset=dataset,
        checkpoint=Path(checkpoint),
        action_pack=Path(action_pack),
        split=split,
        bev_source=bev_source,
        modeld_dir=Path(modeld_dir) if modeld_dir is not None else None,
        inference_fps=inference_fps,
        train_metrics=payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
    )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path.with_name(f"{output_path.stem}_predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in predictions:
            handle.write(deterministic_json(item))
            handle.write("\n")
    metrics["predictions_jsonl"] = prediction_path.as_posix()
    if viz_dir is not None:
        viz_paths = write_trajectory_scorer_contact_sheets(
            dataset=dataset,
            predictions=predictions,
            out_dir=viz_dir,
            max_frames=max_viz_frames,
        )
        metrics["contact_sheets"] = viz_paths
    write_json(output_path, metrics, pretty=True)
    return metrics


def evaluate_trajectory_scorer_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> JsonDict:
    model.eval()
    losses: list[float] = []
    top1: list[float] = []
    rank_corrs: list[float] = []
    started = time.perf_counter()
    frame_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            logits = model(batch["bev"], batch["candidate_features"], batch["candidate_mask"])
            loss_parts = trajectory_scorer_loss(logits, batch)
            losses.append(float(loss_parts["loss"].detach().cpu()))
            selected = torch.argmax(logits, dim=1)
            expert = batch["selected_index"]
            top1.extend((selected == expert).detach().cpu().numpy().astype(np.float32).tolist())
            expert_utility = (-batch["total_expert_score"]).detach().cpu().numpy()
            pred_logits = logits.detach().cpu().numpy()
            for row in range(pred_logits.shape[0]):
                rank_corrs.append(_rank_correlation(pred_logits[row], expert_utility[row]))
            frame_count += int(logits.shape[0])
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    return {
        "loss": _mean(losses),
        "top1_action_agreement": _mean(top1),
        "rank_correlation_or_proxy": _mean(rank_corrs),
        "inference_fps": frame_count / elapsed,
    }


def trajectory_scorer_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    selected_loss = F.cross_entropy(logits, batch["selected_index"])
    mask = batch["candidate_mask"]
    raw_score_loss = ((logits - batch["target_logits"]) ** 2 * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    source_weight = batch["source_weight"].view(-1)
    score_loss = (raw_score_loss * source_weight).sum() / source_weight.sum().clamp_min(1.0)
    total = selected_loss + torch.tensor(0.10, dtype=selected_loss.dtype, device=selected_loss.device) * score_loss
    return {"loss": total, "selected_loss": selected_loss, "score_loss": score_loss}


def score_local_bev_with_model(
    *,
    model: TrajectoryScorerNetV0,
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
    coverage_memory: CoverageMemory | None,
    device: torch.device,
) -> tuple[np.ndarray, str, np.ndarray]:
    coverage_mask = coverage_memory.covered if coverage_memory is not None else None
    features = candidate_feature_matrix(bev=bev, candidates=candidates, coverage_mask=coverage_mask)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(bev_tensor(bev)[None, ...]).to(device),
            torch.from_numpy(features[None, ...]).to(device),
            torch.ones((1, len(candidates)), dtype=torch.float32, device=device),
        )[0]
    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    selected_index = int(np.argmax(logits_np))
    return logits_np, candidates[selected_index].id, features


def candidate_score_records(
    *,
    candidates: list[CandidateTrajectory],
    logits: np.ndarray,
    candidate_features: np.ndarray,
    selected_candidate_id: str,
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for index, candidate in enumerate(candidates):
        feature_values = {
            name: round(float(candidate_features[index, feature_index]), 6)
            for feature_index, name in enumerate(CANDIDATE_FEATURE_NAMES)
        }
        records.append(
            {
                **candidate.to_dict(),
                "trajectory_score": {
                    "schema_version": "homebrain.learned_trajectory_score.v0",
                    "scorer": TRAJECTORY_SCORER_V0_SOURCE,
                    "candidate_id": candidate.id,
                    "learned_logit": round(float(logits[index]), 6),
                    "higher_is_better": True,
                    "selected_by_learned_scorer": candidate.id == selected_candidate_id,
                    "candidate_features": feature_values,
                    "replay_only": True,
                    "not_executed": True,
                    "control_safe": False,
                    "product_training_approved": False,
                },
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
        )
    return records


def save_trajectory_scorer_checkpoint(
    path: str | Path,
    model: TrajectoryScorerNetV0,
    *,
    metadata: JsonDict,
    metrics: JsonDict,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": TRAJECTORY_SCORER_V0_CHECKPOINT_VERSION,
            "model_name": "TrajectoryScorerNetV0",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "metrics": metrics,
        },
        target,
    )


def load_trajectory_scorer_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TrajectoryScorerNetV0, JsonDict]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != TRAJECTORY_SCORER_V0_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported trajectory scorer checkpoint {payload.get('checkpoint_version')!r}; "
            f"expected {TRAJECTORY_SCORER_V0_CHECKPOINT_VERSION!r}"
        )
    config = TrajectoryScorerNetConfig.from_dict(payload["model_config"])
    model = TrajectoryScorerNetV0(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def write_trajectory_scorer_contact_sheets(
    *,
    dataset: ActionLabelScorerDataset,
    predictions: list[JsonDict],
    out_dir: str | Path,
    max_frames: int,
) -> JsonDict:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_by_index = {int(item["sample_index"]): item for item in predictions}
    all_tiles: list[np.ndarray] = []
    disagreement_tiles: list[np.ndarray] = []
    for index, sample in enumerate(dataset.samples):
        prediction = prediction_by_index[index]
        learned = str(prediction["learned_selected_candidate_id"])
        expert = str(prediction["expert_selected_candidate_id"])
        tile = _overlay_tile(sample.bev, dataset.candidates, expert_selected=expert, learned_selected=learned)
        if len(all_tiles) < max_frames:
            all_tiles.append(tile)
        if learned != expert and len(disagreement_tiles) < max_frames:
            disagreement_tiles.append(tile)
    all_path = output / "bev_candidates_expert_learned_contact_sheet.ppm"
    disagreement_path = output / "disagreement_cases_contact_sheet.ppm"
    _write_contact_sheet(all_path, all_tiles)
    _write_contact_sheet(disagreement_path, disagreement_tiles)
    return {
        "bev_candidates_expert_learned": all_path.as_posix(),
        "disagreement_cases": disagreement_path.as_posix(),
    }


def _predict_dataset(
    model: TrajectoryScorerNetV0,
    dataset: ActionLabelScorerDataset,
    *,
    device: torch.device,
) -> tuple[list[JsonDict], float]:
    model.eval()
    predictions: list[JsonDict] = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, sample in enumerate(dataset.samples):
            logits = model(
                torch.from_numpy(sample.bev_tensor[None, ...]).to(device),
                torch.from_numpy(sample.candidate_features[None, ...]).to(device),
                torch.from_numpy(sample.candidate_mask[None, ...]).to(device),
            )[0].detach().cpu().numpy()
            learned_index = int(np.argmax(logits))
            expert_index = int(sample.selected_index)
            learned_selected = dataset.candidate_ids[learned_index]
            expert_selected = dataset.candidate_ids[expert_index]
            predictions.append(
                {
                    "schema_version": "homebrain.trajectory_scorer_v0_prediction.v0",
                    "sample_index": index,
                    "sequence_id": sample.record.get("sequence_id"),
                    "camera_id": sample.record.get("camera_id"),
                    "frame_id": sample.record.get("frame_id"),
                    "timestamp_ns": sample.record.get("timestamp_ns"),
                    "source_name": sample.record.get("source_name"),
                    "source_family": sample.record.get("source_family"),
                    "source_ref": sample.record.get("source_ref"),
                    "bev_source": dataset.bev_source,
                    "learned_selected_candidate_id": learned_selected,
                    "expert_selected_candidate_id": expert_selected,
                    "top1_agrees_with_expert": learned_selected == expert_selected,
                    "candidate_scores": [
                        {
                            "candidate_id": candidate_id,
                            "learned_logit": round(float(logits[candidate_index]), 6),
                            "expert_total_score": round(float(sample.total_expert_score[candidate_index]), 6),
                            "expert_utility": round(float(-sample.total_expert_score[candidate_index]), 6),
                            "selected_by_expert": candidate_index == expert_index,
                            "selected_by_learned_scorer": candidate_index == learned_index,
                            "collision_proxy": round(
                                float(_action_label_array(sample.action_example_path, "collision_proxy")[candidate_index]),
                                6,
                            ),
                            "unknown_penalty": round(
                                float(_action_label_array(sample.action_example_path, "unknown_penalty")[candidate_index]),
                                6,
                            ),
                            "coverage_gain": round(
                                float(_action_label_array(sample.action_example_path, "coverage_gain")[candidate_index]),
                                6,
                            ),
                        }
                        for candidate_index, candidate_id in enumerate(dataset.candidate_ids)
                    ],
                    "rank_correlation_or_proxy": _rank_correlation(logits, -sample.total_expert_score),
                    "replay_only": True,
                    "not_executed": True,
                    "control_safe": False,
                    "product_training_approved": False,
                }
            )
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    return predictions, len(dataset) / elapsed


def _prediction_metrics(
    *,
    predictions: list[JsonDict],
    dataset: ActionLabelScorerDataset,
    checkpoint: Path,
    action_pack: Path,
    split: str | None,
    bev_source: str,
    modeld_dir: Path | None,
    inference_fps: float,
    train_metrics: JsonDict,
) -> JsonDict:
    learned_ids = [str(item["learned_selected_candidate_id"]) for item in predictions]
    expert_ids = [str(item["expert_selected_candidate_id"]) for item in predictions]
    agreements = [1.0 if learned == expert else 0.0 for learned, expert in zip(learned_ids, expert_ids)]
    selected_distribution = _distribution(learned_ids)
    expert_distribution = _distribution(expert_ids)
    dominant_action_fraction = max(selected_distribution.values(), default=0) / max(len(predictions), 1)
    selected_stop_count = sum(1 for value in learned_ids if value == "stop")
    selected_collision: list[float] = []
    selected_unknown: list[float] = []
    selected_coverage: list[float] = []
    rank_corrs: list[float] = []
    per_source_agreements: dict[str, list[float]] = {}
    for item in predictions:
        source = str(item.get("source_name", "unknown"))
        per_source_agreements.setdefault(source, []).append(1.0 if item["top1_agrees_with_expert"] else 0.0)
        selected_id = str(item["learned_selected_candidate_id"])
        score_records = item["candidate_scores"]
        selected_score = next(score for score in score_records if score["candidate_id"] == selected_id)
        selected_collision.append(float(selected_score["collision_proxy"]))
        selected_unknown.append(float(selected_score["unknown_penalty"]))
        selected_coverage.append(float(selected_score["coverage_gain"]))
        rank_corrs.append(float(item["rank_correlation_or_proxy"]))
    unsafe = [
        value >= COLLISION_THRESHOLD or unknown >= UNKNOWN_BLOCK_THRESHOLD
        for value, unknown in zip(selected_collision, selected_unknown)
    ]
    random_agreement = 1.0 / max(len(dataset.candidate_ids), 1)
    collapse = dominant_action_fraction >= 0.90
    return {
        "schema_version": TRAJECTORY_SCORER_EVAL_SCHEMA_VERSION,
        "checkpoint": checkpoint.as_posix(),
        "action_pack": action_pack.as_posix(),
        "split": split,
        "bev_source": bev_source,
        "modeld_dir": modeld_dir.as_posix() if modeld_dir is not None else None,
        "example_count": len(predictions),
        "candidate_count": len(dataset.candidate_ids),
        "candidate_ids": list(dataset.candidate_ids),
        "source_distribution": dataset.source_distribution,
        "source_family_distribution": dataset.source_family_distribution,
        "excluded_geometry_only_sources_from_action_training": dataset.excluded_geometry_only_sources,
        "top1_action_agreement": _mean(agreements),
        "random_top1_action_agreement": float(random_agreement),
        "beats_random": bool(_mean(agreements) > random_agreement),
        "selected_motion_fraction": float((len(predictions) - selected_stop_count) / max(len(predictions), 1)),
        "selected_stop_fraction": float(selected_stop_count / max(len(predictions), 1)),
        "action_entropy": _entropy(selected_distribution),
        "dominant_action_fraction": float(dominant_action_fraction),
        "rank_correlation_or_proxy": _mean(rank_corrs),
        "collision_proxy_rate": float(sum(1 for value in selected_collision if value >= COLLISION_THRESHOLD) / max(len(selected_collision), 1)),
        "coverage_gain_mean": _mean(selected_coverage),
        "unsafe_selected_rate": float(sum(1 for value in unsafe if value) / max(len(unsafe), 1)),
        "inference_fps": float(inference_fps),
        "distribution_collapse_flag": bool(collapse),
        "selected_distribution": selected_distribution,
        "heuristic_expert_distribution": expert_distribution,
        "heuristic_expert_action_entropy": _entropy(expert_distribution),
        "heuristic_expert_stop_fraction": float(sum(1 for value in expert_ids if value == "stop") / max(len(expert_ids), 1)),
        "per_source_top1_action_agreement": {
            source: _mean(values) for source, values in sorted(per_source_agreements.items())
        },
        "train_metrics_summary": {
            "train_loss_end": train_metrics.get("train_loss_end"),
            "val_loss": train_metrics.get("val_loss"),
            "val_top1_action_agreement": train_metrics.get("val_top1_action_agreement"),
            "product_training_approved": train_metrics.get("product_training_approved", False),
        },
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
    }


def _load_model_bev_map(modeld_dir: str | Path | None) -> dict[tuple[str, int], LocalBev]:
    if modeld_dir is None:
        return {}
    root = Path(modeld_dir)
    records: dict[tuple[str, int], LocalBev] = {}
    for event in read_events(root):
        if not isinstance(event, BrainOutputEvent) or not isinstance(event.local_bev_ref, str):
            continue
        frame_id = -1
        if isinstance(event.debug, dict) and isinstance(event.debug.get("input_frame_id"), int):
            frame_id = int(event.debug["input_frame_id"])
        if frame_id < 0:
            continue
        artifact_path = root / event.local_bev_ref
        with np.load(artifact_path, allow_pickle=False) as data:
            uncertainty = np.asarray(data["uncertainty_grid"], dtype=np.float32)
            records[(event.sequence_id, frame_id)] = LocalBev(
                free=np.asarray(data["bev_free_prob"], dtype=np.float32),
                occupied=np.asarray(data["bev_occupied_prob"], dtype=np.float32),
                unknown=np.asarray(data["bev_unknown_prob"], dtype=np.float32),
                traversable=np.asarray(data["bev_traversable_prob"], dtype=np.float32),
                risky=np.asarray(data["bev_risky_prob"], dtype=np.float32),
                confidence=(np.float32(1.0) - uncertainty).astype(np.float32),
                uncertainty=uncertainty,
                source="model",
            )
    return records


def _load_oracle_bev(path: Path) -> LocalBev:
    example = load_example_npz(path)
    free = np.asarray(example["bev_free"], dtype=np.float32)
    obstacle = np.asarray(example["bev_obstacle"], dtype=np.float32)
    unknown = np.asarray(example["bev_unknown"], dtype=np.float32)
    confidence = np.asarray(example["bev_confidence"], dtype=np.float32)
    traversable = np.asarray(example["bev_traversable"], dtype=np.float32) if "bev_traversable" in example else free.copy()
    risky = np.asarray(example["bev_risky"], dtype=np.float32) if "bev_risky" in example else obstacle.copy()
    return LocalBev(
        free=free,
        occupied=obstacle,
        unknown=unknown,
        traversable=traversable,
        risky=risky,
        confidence=confidence,
        uncertainty=(np.float32(1.0) - _prob(confidence)).astype(np.float32),
        source="labels",
    )


def _filter_records(
    examples: list[Any],
    *,
    split: str | None,
    val_modulus: int,
    source_names: set[str] | None,
    source_families: set[str] | None,
    allow_geometry_only: bool,
) -> list[tuple[int, JsonDict]]:
    records: list[tuple[int, JsonDict]] = []
    for index, record in enumerate(examples):
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            continue
        if record.get("action_supervision_ok") is not True:
            continue
        if record.get("control_safe") is not False:
            raise ValueError(f"action example must remain control_safe=false: {record.get('example_path')}")
        source_name = str(record.get("source_name", ""))
        source_family = str(record.get("source_family", ""))
        if source_names is not None and source_name not in source_names:
            continue
        if source_families is not None and source_family not in source_families:
            continue
        if not allow_geometry_only and source_family in GEOMETRY_ONLY_SOURCE_FAMILIES:
            raise ValueError(f"geometry-only source entered action scorer dataset: {source_name}")
        is_val = (index % val_modulus) == 0
        if split == "train" and is_val:
            continue
        if split == "val" and not is_val:
            continue
        records.append((index, dict(record)))
    return records


def _resolve_source_roots(action_pack_root: Path, manifest: JsonDict) -> list[Path]:
    raw = manifest.get("source_dirs")
    if not isinstance(raw, list):
        raise ValueError("ActionLabelPack manifest must include source_dirs")
    roots: list[Path] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        candidate = Path(item)
        if candidate.exists():
            roots.append(candidate)
        else:
            roots.append(action_pack_root.parent / candidate)
    return roots


def _target_logits_from_expert_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    utility = -scores
    std = float(np.std(utility))
    if std < 1.0e-6:
        return np.zeros_like(utility, dtype=np.float32)
    return ((utility - np.float32(np.mean(utility))) / np.float32(std)).astype(np.float32)


def _validate_action_example_flags(example: dict[str, np.ndarray], path: Path) -> None:
    for field, expected in (("replay_only", True), ("not_executed", True), ("control_safe", False)):
        if field not in example:
            raise ValueError(f"action example missing {field}: {path}")
        value = bool(np.asarray(example[field]).item())
        if value is not expected:
            raise ValueError(f"action example {field}={value}, expected {expected}: {path}")


def _record_source_weight(record: JsonDict) -> float:
    value = record.get("source_weight", 1.0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 1.0


def _excluded_geometry_only_sources(manifest: JsonDict) -> JsonDict:
    action_filter = manifest.get("action_sanity_filter")
    if not isinstance(action_filter, dict):
        return {}
    excluded = action_filter.get("excluded_by_source")
    if not isinstance(excluded, dict):
        return {}
    return {str(key): int(value) for key, value in sorted(excluded.items())}


def _update_coverage_mask(mask: np.ndarray, bev: LocalBev) -> None:
    free = _prob(bev.free)
    traversable = _prob(bev.traversable) if bev.traversable is not None else free
    mask |= (free >= 0.5) | (traversable >= 0.5)


def _action_label_array(path: Path, field: str) -> np.ndarray:
    example = load_example_npz(path)
    return np.asarray(example[field], dtype=np.float32)


def _grid_shape(manifest: JsonDict) -> tuple[int, int]:
    raw = manifest.get("grid_shape")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("ActionLabelPack manifest must include grid_shape")
    return (int(raw[0]), int(raw[1]))


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return list(value)


def _positive_float(value: Any, default: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number > 0.0 or (allow_zero and number >= 0.0):
            return number
    return default


def _cell_values(array: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    if not cells:
        return np.asarray([default], dtype=np.float32)
    values: list[float] = []
    rows, cols = array.shape
    for row, col in cells:
        if 0 <= row < rows and 0 <= col < cols:
            values.append(float(array[row, col]))
        else:
            values.append(default)
    if not values:
        values.append(default)
    return np.asarray(values, dtype=np.float32)


def _prob(array: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _torch_generator() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(11)
    return generator


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _distribution(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0.0:
            entropy -= probability * math.log2(probability)
    return float(entropy)


def _rank_correlation(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_rank = _ranks(np.asarray(predicted, dtype=np.float64))
    target_rank = _ranks(np.asarray(target, dtype=np.float64))
    pred_std = float(np.std(pred_rank))
    target_std = float(np.std(target_rank))
    if pred_std < 1.0e-9 or target_std < 1.0e-9:
        return 0.0
    corr = float(np.corrcoef(pred_rank, target_rank)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def _overlay_tile(
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
    *,
    expert_selected: str,
    learned_selected: str,
) -> np.ndarray:
    free = _prob(bev.free)
    occupied = _prob(bev.occupied)
    unknown = _prob(bev.unknown)
    uncertainty = _prob(bev.uncertainty) if bev.uncertainty is not None else unknown
    rgb = np.stack([occupied, free, np.maximum(unknown * 0.55, uncertainty * 0.65)], axis=2)
    tile = (rgb * np.float32(160.0)).round().astype(np.uint8)
    scale = 6 if min(bev.shape) >= 16 else 16
    tile = np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)
    for candidate in candidates:
        color = np.asarray([110, 110, 110], dtype=np.uint8)
        if candidate.id == expert_selected:
            color = np.asarray([35, 230, 80], dtype=np.uint8)
        if candidate.id == learned_selected:
            color = np.asarray([255, 225, 40], dtype=np.uint8) if learned_selected == expert_selected else np.asarray([255, 60, 60], dtype=np.uint8)
        for row, col in candidate.footprint_cells:
            _paint_cell(tile, row, col, scale, color)
    return tile


def _paint_cell(tile: np.ndarray, row: int, col: int, scale: int, color: np.ndarray) -> None:
    row_start = row * scale
    col_start = col * scale
    if row_start < 0 or col_start < 0 or row_start >= tile.shape[0] or col_start >= tile.shape[1]:
        return
    row_end = min(row_start + scale, tile.shape[0])
    col_end = min(col_start + scale, tile.shape[1])
    patch = tile[row_start:row_end, col_start:col_end]
    tile[row_start:row_end, col_start:col_end] = ((patch.astype(np.uint16) + color.astype(np.uint16)) // 2).astype(np.uint8)


def _write_contact_sheet(path: Path, tiles: list[np.ndarray]) -> None:
    if not tiles:
        path.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        return
    tile_h, tile_w, _channels = tiles[0].shape
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    path.write_bytes(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii") + sheet.tobytes())


def scorer_checkpoint_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _parse_repeated(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {value for value in values if value}


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train replay-only TrajectoryScorerNet v0 from ActionLabelPack labels.")
    parser.add_argument("--action-pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--source-family", action="append", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)
    metrics = train_trajectory_scorer_v0(
        action_pack=args.action_pack,
        out_dir=args.out,
        source_names=_parse_repeated(args.source_name),
        source_families=_parse_repeated(args.source_family),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
        max_examples=args.max_examples,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate replay-only TrajectoryScorerNet v0 on oracle or model BEV.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--source-family", action="append", default=None)
    parser.add_argument("--bev-source", choices=("oracle", "model"), default="oracle")
    parser.add_argument("--modeld", default=None)
    parser.add_argument("--split", default="val", help="train, val, all, none, or empty string for all examples")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--viz-out", default=None)
    parser.add_argument("--max-viz-frames", type=int, default=24)
    args = parser.parse_args(argv)
    metrics = eval_trajectory_scorer_v0(
        checkpoint=args.checkpoint,
        action_pack=args.action_pack,
        out_path=args.out,
        source_names=_parse_repeated(args.source_name),
        source_families=_parse_repeated(args.source_family),
        bev_source=args.bev_source,
        modeld_dir=args.modeld,
        split=None if args.split in {"", "all", "none"} else args.split,
        device_name=args.device,
        max_examples=args.max_examples,
        viz_dir=args.viz_out,
        max_viz_frames=args.max_viz_frames,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0
