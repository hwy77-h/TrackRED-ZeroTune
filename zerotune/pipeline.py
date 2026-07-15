from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .drift import GateConfig, TrackingState, classify_mask
from .io_utils import write_prediction
from .mask_ops import clean_mask, iou, largest_component_ratio, mask_center
from .motion import local_tvl1_warp
from .sam2_branch import run_sam2_causal


def _quality_score(stats: dict, seed_area: int, intentional_absence: bool = False) -> tuple[float, str]:
    if intentional_absence:
        return 1.0, "intentional_absence"

    area = float(stats.get("area", 0) or 0)
    area_ratio = float(stats.get("area_ratio", 0) or 0)
    prev_iou = float(stats.get("prev_iou", 0) or 0)
    center_jump = float(stats.get("center_jump", float("inf")) or float("inf"))
    component_ratio = float(stats.get("largest_component_ratio", 0) or 0)
    flags = []

    if area <= 0:
        return 0.0, "empty"

    if area_ratio <= 0:
        area_score = 0.0
    else:
        area_score = max(0.0, 1.0 - abs(float(np.log(area_ratio))) / float(np.log(2.5)))
    iou_score = min(1.0, max(0.0, prev_iou / 0.90))
    center_limit = max(20.0, 0.75 * float(np.sqrt(max(1, seed_area))))
    center_score = 0.0 if not np.isfinite(center_jump) else max(0.0, 1.0 - center_jump / center_limit)
    component_score = min(1.0, max(0.0, component_ratio))

    if not 0.55 <= area_ratio <= 1.80:
        flags.append("area")
    if prev_iou < 0.35:
        flags.append("iou")
    if not np.isfinite(center_jump) or center_jump > center_limit:
        flags.append("center")
    if component_ratio < 0.95:
        flags.append("component")

    score = 0.30 * area_score + 0.30 * iou_score + 0.25 * center_score + 0.15 * component_score
    return float(max(0.0, min(1.0, score))), "|".join(flags) if flags else "ok"


@dataclass
class PipelineConfig:
    sam2_root: Path
    checkpoint: Path
    config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    work_dir: Path = Path("work")
    output_dir: Path | None = None
    device: str = "cpu"
    image_size: int = 512
    bbox_padding: int = 0
    vos_optimized: bool = False
    preprocessing_mode: str = "trackrad_percentile"
    preprocessing_percentile_low: float = 0.5
    preprocessing_percentile_high: float = 99.5
    motion_method: str = "tvl1"
    save_intermediates: bool = False
    min_component_area: int = 25
    roi_padding: int = 32
    gate_mode: str = "coldstart_adaptive"
    coldstart_frames: int = 5
    hybrid_frames: int = 10
    rolling_window: int = 8
    max_continuous_red: int = 5
    lost_mode: str = "hold"
    trusted_iou_min: float = 0.90
    trusted_area_min: float = 0.85
    trusted_area_max: float = 1.15
    trusted_component_ratio_min: float = 0.95
    collapse_area_min: float = 0.80
    collapse_iou_max: float = 0.94
    red_sam2_reacquire: bool = True
    red_reacquire_prev_iou_min: float = 0.65
    red_reacquire_prev_area_min: float = 0.50
    red_reacquire_prev_area_max: float = 1.80
    red_reacquire_seed_area_min: float = 0.30
    red_reacquire_seed_area_max: float = 3.00
    red_reacquire_center_max: float = 12.0
    red_reacquire_component_min: float = 0.95
    thorax_empty_on_sam2_empty: bool = True
    empty_consecutive_frames: int = 1
    thorax_empty_on_long_red_tiny: bool = True
    long_red_empty_frames: int = 10
    long_red_empty_seed_ratio: float = 0.60
    thorax_empty_on_red_collapse: bool = True
    red_collapse_empty_frames: int = 7
    red_collapse_empty_seed_ratio: float = 0.85
    red_collapse_empty_prev_iou_max: float = 0.35
    red_collapse_empty_seed_center_max: float = 18.0
    thorax_reentry_after_absence: bool = True
    reentry_min_absence_frames: int = 1
    reentry_seed_area_min: float = 0.80
    reentry_seed_area_max: float = 1.30
    reentry_seed_center_max: float = 12.0
    reentry_component_min: float = 0.95
    raw_relocation_rescue: bool = True
    raw_relocation_min_red: int = 2
    raw_relocation_prev_iou_min: float = 0.05
    raw_relocation_prev_iou_max: float = 0.20
    raw_relocation_temporal_iou_min: float = 0.10
    raw_relocation_area_min: float = 0.80
    raw_relocation_area_max: float = 2.00
    raw_relocation_center_min: float = 18.0
    raw_relocation_component_min: float = 0.95
    raw_relocation_thorax_min_red: int = 999
    raw_relocation_thorax_area_min: float = 0.80
    lost_raw_reacquire: bool = True
    lost_raw_reacquire_min_red: int = 2
    lost_raw_reacquire_seed_area_min: float = 0.70
    lost_raw_reacquire_seed_area_max: float = 2.20
    lost_raw_reacquire_thorax_seed_area_min: float = 0.90
    lost_raw_reacquire_prev_iou_max: float = 0.20
    lost_raw_reacquire_seed_center_max: float = 38.0
    lost_raw_reacquire_component_min: float = 0.95
    stable_update_admission: bool = True
    admission_nonthorax_only: bool = True
    admission_reacquire_seed_area_min: float = 1.00
    admission_reacquire_seed_area_max: float = 1.50
    admission_reacquire_temporal_iou_min: float = 0.65
    admission_component_min: float = 0.95
    admission_raw_relocation_seed_area_min: float = 0.80
    admission_raw_relocation_seed_area_max: float = 2.00
    admission_raw_relocation_temporal_iou_min: float = 0.10
    admission_raw_relocation_component_min: float = 0.95
    admission_lost_raw_seed_area_min: float = 0.80
    admission_lost_raw_seed_area_max: float = 1.50
    admission_lost_raw_temporal_iou_min: float = 0.10
    admission_lost_raw_component_min: float = 0.95


def _write_debug_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_idx",
        "gate_mode",
        "decision",
        "gate_reason",
        "initial_gate_reason",
        "final_gate_reason",
        "accepted_source",
        "area",
        "area_ratio",
        "rolling_area_median",
        "rolling_area_mad",
        "rolling_center_jump_median",
        "rolling_center_jump_mad",
        "prev_iou",
        "center_jump",
        "largest_component_ratio",
        "trusted_stable_override",
        "cotracker_visibility",
        "raw_prev_iou",
        "raw_temporal_iou",
        "raw_prev_area_ratio",
        "raw_seed_area_ratio",
        "raw_prev_center_jump",
        "raw_seed_center_jump",
        "red_reacquire_eligible",
        "raw_relocation_eligible",
        "stable_update_admitted",
        "stable_update_reason",
        "stable_update_score",
        "mask_quality_score",
        "quality_flags",
        "consecutive_sam2_empty_red",
        "consecutive_red_collapse_absence",
        "consecutive_absence_output",
        "target_absence_eligible",
        "reentry_eligible",
        "continuous_red_count",
        "stable_history_size",
        "lost_mode",
        "update_stable_stats",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def causal_quality_gate(
    frames: np.ndarray,
    sam2_masks: np.ndarray,
    first_mask: np.ndarray,
    config: PipelineConfig,
    gate_config: GateConfig | None = None,
    scanned_region: str = "",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    gate_config = gate_config or GateConfig(
        gate_mode=config.gate_mode,
        coldstart_frames=config.coldstart_frames,
        hybrid_frames=config.hybrid_frames,
        rolling_window=config.rolling_window,
        max_continuous_red=config.max_continuous_red,
        trusted_iou_min=config.trusted_iou_min,
        trusted_area_min=config.trusted_area_min,
        trusted_area_max=config.trusted_area_max,
        trusted_component_ratio_min=config.trusted_component_ratio_min,
        collapse_area_min=config.collapse_area_min,
        collapse_iou_max=config.collapse_iou_max,
    )
    region = str(scanned_region).lower()
    total_frames = sam2_masks.shape[2]
    output = np.zeros_like(sam2_masks, dtype=np.uint8)
    motion_debug = np.zeros_like(sam2_masks, dtype=np.uint8)
    debug_rows: list[dict] = []
    consecutive_sam2_empty_red = 0
    consecutive_red_collapse_absence = 0
    consecutive_absence_output = 0

    first_clean = clean_mask(first_mask, min_component_area=config.min_component_area)
    if first_clean.sum() == 0:
        first_clean = (first_mask > 0).astype(np.uint8)
    output[:, :, 0] = first_clean
    state = TrackingState.from_first_mask(first_clean)
    seed_center = mask_center(first_clean)
    debug_rows.append(
        {
            "frame_idx": 0,
            "gate_mode": "seed",
            "decision": "green",
            "gate_reason": "seed",
            "initial_gate_reason": "seed",
            "final_gate_reason": "seed",
            "accepted_source": "first_mask",
            "area": int(first_clean.sum()),
            "area_ratio": 1.0,
            "rolling_area_median": int(first_clean.sum()),
            "rolling_area_mad": 0.0,
            "rolling_center_jump_median": "",
            "rolling_center_jump_mad": "",
            "prev_iou": 1.0,
            "center_jump": 0.0,
            "largest_component_ratio": 1.0,
            "cotracker_visibility": "",
            "mask_quality_score": 1.0,
            "quality_flags": "seed",
            "consecutive_sam2_empty_red": 0,
            "consecutive_red_collapse_absence": 0,
            "consecutive_absence_output": 0,
            "target_absence_eligible": False,
            "reentry_eligible": False,
            "continuous_red_count": 0,
            "stable_history_size": 1,
            "lost_mode": False,
            "update_stable_stats": True,
        }
    )

    for frame_idx in range(1, total_frames):
        sam2_mask = clean_mask(sam2_masks[:, :, frame_idx], min_component_area=config.min_component_area)
        stats = classify_mask(sam2_mask, state, gate_config, frame_idx=frame_idx, cotracker_visibility=None)
        decision = stats["decision"]
        initial_gate_reason = stats.get("gate_reason", "")
        raw_prev_iou = ""
        raw_prev_area_ratio = ""
        raw_seed_area_ratio = ""
        raw_prev_center_jump = ""
        raw_seed_center_jump = ""
        raw_temporal_iou = ""
        red_reacquire_eligible = False
        raw_relocation_eligible = False
        stable_update_admitted = False
        stable_update_reason = ""
        stable_update_score = ""
        target_absence_eligible = False
        reentry_eligible = False
        accepted_source = "sam2"
        accepted = sam2_mask
        update_stable_stats = decision == "green"

        if decision == "green":
            consecutive_sam2_empty_red = 0
            consecutive_red_collapse_absence = 0
        elif decision == "yellow":
            # CoTracker prompt correction hook. The v1 implementation does not
            # require CoTracker weights, so it records the re-prompt opportunity
            # but falls through to the SAM2 mask when it passes non-severe checks.
            accepted_source = "sam2_cotracker_prompt_pending"
            update_stable_stats = False
            consecutive_sam2_empty_red = 0
            consecutive_red_collapse_absence = 0
        elif decision == "red":
            previous_area = int((state.previous_mask > 0).sum())
            previous_center = mask_center(state.previous_mask)
            raw_center = mask_center(sam2_mask)
            raw_prev_iou = iou(sam2_mask, state.previous_mask)
            prev_sam2_mask = clean_mask(sam2_masks[:, :, frame_idx - 1], min_component_area=config.min_component_area)
            raw_temporal_iou = iou(sam2_mask, prev_sam2_mask)
            raw_prev_area_ratio = int(sam2_mask.sum()) / max(1, previous_area)
            raw_seed_area_ratio = int(sam2_mask.sum()) / max(1, state.seed_area)
            raw_prev_center_jump = (
                float("inf")
                if previous_center is None or raw_center is None
                else float(np.linalg.norm(raw_center - previous_center))
            )
            raw_seed_center_jump = (
                float("inf")
                if seed_center is None or raw_center is None
                else float(np.linalg.norm(raw_center - seed_center))
            )
            sam2_empty_or_tiny = int(sam2_mask.sum()) < config.min_component_area
            if int(sam2_mask.sum()) == 0:
                consecutive_sam2_empty_red += 1
            else:
                consecutive_sam2_empty_red = 0
            collapse_absence_evidence = (
                region == "thorax"
                and config.thorax_empty_on_red_collapse
                and not sam2_empty_or_tiny
                and raw_seed_area_ratio <= config.red_collapse_empty_seed_ratio
                and raw_prev_iou <= config.red_collapse_empty_prev_iou_max
                and raw_seed_center_jump <= config.red_collapse_empty_seed_center_max
                and any(
                    reason in initial_gate_reason
                    for reason in ("area_collapse", "severe_area_drop", "severe_iou_drop")
                )
            )
            if collapse_absence_evidence:
                consecutive_red_collapse_absence += 1
            else:
                consecutive_red_collapse_absence = 0
            red_reacquire_eligible = (
                config.red_sam2_reacquire
                and not sam2_empty_or_tiny
                and raw_prev_iou >= config.red_reacquire_prev_iou_min
                and config.red_reacquire_prev_area_min <= raw_prev_area_ratio <= config.red_reacquire_prev_area_max
                and config.red_reacquire_seed_area_min <= raw_seed_area_ratio <= config.red_reacquire_seed_area_max
                and raw_prev_center_jump <= config.red_reacquire_center_max
                and largest_component_ratio(sam2_mask) >= config.red_reacquire_component_min
            )
            if red_reacquire_eligible:
                accepted = sam2_mask
                accepted_source = "sam2_red_reacquire"
                update_stable_stats = True
            elif (
                config.raw_relocation_rescue
                and state.continuous_red_count + 1 >= config.raw_relocation_min_red
                and not sam2_empty_or_tiny
                and config.raw_relocation_prev_iou_min <= raw_prev_iou <= config.raw_relocation_prev_iou_max
                and raw_temporal_iou >= config.raw_relocation_temporal_iou_min
                and config.raw_relocation_area_min <= raw_seed_area_ratio <= config.raw_relocation_area_max
                and raw_prev_center_jump >= config.raw_relocation_center_min
                and largest_component_ratio(sam2_mask) >= config.raw_relocation_component_min
                and not (
                    region == "thorax"
                    and (
                        consecutive_absence_output > 0
                        or raw_seed_area_ratio < config.raw_relocation_thorax_area_min
                        or state.continuous_red_count + 1 < config.raw_relocation_thorax_min_red
                    )
                )
            ):
                accepted = sam2_mask
                accepted_source = "raw_relocation_rescue"
                update_stable_stats = True
                raw_relocation_eligible = True
            elif (
                config.thorax_reentry_after_absence
                and region == "thorax"
                and consecutive_absence_output >= config.reentry_min_absence_frames
                and not sam2_empty_or_tiny
                and config.reentry_seed_area_min <= raw_seed_area_ratio <= config.reentry_seed_area_max
                and raw_seed_center_jump <= config.reentry_seed_center_max
                and largest_component_ratio(sam2_mask) >= config.reentry_component_min
            ):
                accepted = sam2_mask
                accepted_source = "thorax_reentry_sam2"
                update_stable_stats = True
                reentry_eligible = True
            elif (
                config.thorax_empty_on_sam2_empty
                and region == "thorax"
                and consecutive_sam2_empty_red >= config.empty_consecutive_frames
            ):
                accepted = np.zeros_like(state.last_stable_mask, dtype=np.uint8)
                accepted_source = "thorax_sam2_empty"
                target_absence_eligible = True
            elif (
                config.thorax_empty_on_red_collapse
                and region == "thorax"
                and state.continuous_red_count + 1 >= config.red_collapse_empty_frames
                and consecutive_red_collapse_absence >= 1
            ):
                accepted = np.zeros_like(state.last_stable_mask, dtype=np.uint8)
                accepted_source = "thorax_red_collapse_absence"
                target_absence_eligible = True
            elif (
                config.thorax_empty_on_long_red_tiny
                and region == "thorax"
                and state.continuous_red_count + 1 >= config.long_red_empty_frames
                and raw_seed_area_ratio <= config.long_red_empty_seed_ratio
            ):
                accepted = np.zeros_like(state.last_stable_mask, dtype=np.uint8)
                accepted_source = "thorax_long_red_tiny"
                target_absence_eligible = True
            elif (
                config.lost_raw_reacquire
                and state.continuous_red_count + 1 >= config.lost_raw_reacquire_min_red
                and not sam2_empty_or_tiny
                and max(
                    config.lost_raw_reacquire_seed_area_min,
                    config.lost_raw_reacquire_thorax_seed_area_min if region == "thorax" else 0.0,
                )
                <= raw_seed_area_ratio
                <= config.lost_raw_reacquire_seed_area_max
                and raw_prev_iou <= config.lost_raw_reacquire_prev_iou_max
                and raw_seed_center_jump <= config.lost_raw_reacquire_seed_center_max
                and largest_component_ratio(sam2_mask) >= config.lost_raw_reacquire_component_min
            ):
                accepted = sam2_mask
                accepted_source = "lost_raw_reacquire"
                update_stable_stats = True
            else:
                accepted_source = "motion_fallback"
                roi_padding = config.roi_padding
                if state.continuous_red_count >= 2:
                    roi_padding = int(round(config.roi_padding * 1.75))
                if config.motion_method == "tvl1":
                    accepted = local_tvl1_warp(
                        frames[:, :, frame_idx - 1],
                        frames[:, :, frame_idx],
                        state.last_stable_mask,
                        roi_padding=roi_padding,
                        min_component_area=config.min_component_area,
                    )
                else:
                    accepted = state.last_stable_mask.copy()
                    accepted_source = "last_stable"
                motion_debug[:, :, frame_idx] = accepted

                fallback_stats = classify_mask(
                    accepted, state, gate_config, frame_idx=frame_idx, cotracker_visibility=None
                )
                if fallback_stats["decision"] == "red":
                    accepted = state.last_stable_mask.copy()
                    accepted_source = "last_stable"
                if state.continuous_red_count >= config.max_continuous_red and config.lost_mode == "hold":
                    accepted = state.last_stable_mask.copy()
                    accepted_source = "lost_mode_last_stable"

        accepted = clean_mask(accepted, min_component_area=config.min_component_area)
        absence_sources = {"thorax_sam2_empty", "thorax_long_red_tiny", "thorax_red_collapse_absence"}
        if accepted.sum() == 0 and accepted_source not in absence_sources:
            accepted = state.last_stable_mask.copy()
            accepted_source = "last_stable"
        output[:, :, frame_idx] = accepted

        final_stats = classify_mask(accepted, state, gate_config, frame_idx=frame_idx, cotracker_visibility=None)
        mask_quality_score, quality_flags = _quality_score(
            final_stats,
            state.seed_area,
            intentional_absence=accepted_source in absence_sources,
        )
        forced_update_sources = {"sam2_red_reacquire", "raw_relocation_rescue", "lost_raw_reacquire"}
        if config.stable_update_admission and accepted_source in forced_update_sources:
            region_is_allowed = not config.admission_nonthorax_only or region != "thorax"
            if accepted_source == "raw_relocation_rescue":
                seed_area_min = config.admission_raw_relocation_seed_area_min
                seed_area_max = config.admission_raw_relocation_seed_area_max
                temporal_min = config.admission_raw_relocation_temporal_iou_min
                component_min = config.admission_raw_relocation_component_min
                admit_label = "admit_raw_relocation"
                reject_label = "reject_raw_relocation"
            elif accepted_source == "lost_raw_reacquire":
                seed_area_min = config.admission_lost_raw_seed_area_min
                seed_area_max = config.admission_lost_raw_seed_area_max
                temporal_min = config.admission_lost_raw_temporal_iou_min
                component_min = config.admission_lost_raw_component_min
                admit_label = "admit_lost_raw_reacquire"
                reject_label = "reject_lost_raw_reacquire"
            else:
                seed_area_min = config.admission_reacquire_seed_area_min
                seed_area_max = config.admission_reacquire_seed_area_max
                temporal_min = config.admission_reacquire_temporal_iou_min
                component_min = config.admission_component_min
                admit_label = "admit_red_reacquire"
                reject_label = "reject_red_reacquire"
            seed_area_is_allowed = raw_seed_area_ratio != "" and seed_area_min <= raw_seed_area_ratio <= seed_area_max
            temporal_is_allowed = raw_temporal_iou != "" and raw_temporal_iou >= temporal_min
            component_ratio = largest_component_ratio(accepted)
            component_is_allowed = component_ratio >= component_min
            stable_update_score = float(
                np.mean(
                    [
                        1.0 if region_is_allowed else 0.0,
                        1.0 if seed_area_is_allowed else 0.0,
                        1.0 if temporal_is_allowed else 0.0,
                        min(1.0, float(component_ratio)),
                    ]
                )
            )
            stable_update_admitted = bool(
                update_stable_stats
                and accepted.sum() > 0
                and region_is_allowed
                and seed_area_is_allowed
                and temporal_is_allowed
                and component_is_allowed
            )
            if stable_update_admitted:
                stable_update_reason = admit_label
            else:
                failed = []
                if not region_is_allowed:
                    failed.append("region")
                if not seed_area_is_allowed:
                    failed.append("seed_area")
                if not temporal_is_allowed:
                    failed.append("temporal")
                if not component_is_allowed:
                    failed.append("component")
                stable_update_reason = reject_label + ":" + "|".join(failed)

        force_stable_update = stable_update_admitted or (
            not config.stable_update_admission
            and accepted_source in forced_update_sources
        )
        if update_stable_stats and (final_stats["decision"] != "red" or force_stable_update):
            state.update_stable(accepted, final_stats["center_jump"], rolling_window=gate_config.rolling_window)
            update_stable_stats = True
            if not stable_update_reason:
                stable_update_reason = "standard_green_or_forced"
            stable_update_admitted = True
        else:
            if decision == "red":
                state.update_failure(accepted, gate_config)
            else:
                state.previous_mask = (accepted > 0).astype(np.uint8)
            update_stable_stats = False

        if accepted_source in absence_sources:
            consecutive_absence_output += 1
        elif accepted.sum() > 0:
            consecutive_absence_output = 0

        final_stats.update(
            {
                "frame_idx": frame_idx,
                "decision": decision,
                "gate_reason": initial_gate_reason,
                "initial_gate_reason": initial_gate_reason,
                "final_gate_reason": final_stats.get("gate_reason", ""),
                "accepted_source": accepted_source,
                "raw_prev_iou": raw_prev_iou,
                "raw_temporal_iou": raw_temporal_iou,
                "raw_prev_area_ratio": raw_prev_area_ratio,
                "raw_seed_area_ratio": raw_seed_area_ratio,
                "raw_prev_center_jump": raw_prev_center_jump,
                "raw_seed_center_jump": raw_seed_center_jump,
                "red_reacquire_eligible": red_reacquire_eligible,
                "raw_relocation_eligible": raw_relocation_eligible,
                "stable_update_admitted": stable_update_admitted,
                "stable_update_reason": stable_update_reason,
                "stable_update_score": stable_update_score,
                "mask_quality_score": mask_quality_score,
                "quality_flags": quality_flags,
                "consecutive_sam2_empty_red": consecutive_sam2_empty_red,
                "consecutive_red_collapse_absence": consecutive_red_collapse_absence,
                "consecutive_absence_output": consecutive_absence_output,
                "target_absence_eligible": target_absence_eligible,
                "reentry_eligible": reentry_eligible,
                "continuous_red_count": state.continuous_red_count,
                "stable_history_size": state.stable_history_size(),
                "lost_mode": state.lost_mode,
                "update_stable_stats": update_stable_stats,
            }
        )
        debug_rows.append(final_stats)

    return output.astype(np.uint8), motion_debug.astype(np.uint8), debug_rows


def run_causal_zerotune(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate,
    magnetic_field_strength,
    scanned_region,
    config: PipelineConfig,
    frame_image=None,
) -> tuple[np.ndarray, dict]:
    _ = frame_rate, magnetic_field_strength
    first_mask = target[:, :, 0].astype(np.uint8)
    start = time.perf_counter()
    sam2_masks = run_sam2_causal(
        frames=frames,
        first_mask=first_mask,
        sam2_root=config.sam2_root,
        checkpoint=config.checkpoint,
        config=config.config,
        work_dir=config.work_dir,
        device=config.device,
        image_size=config.image_size,
        bbox_padding=config.bbox_padding,
        vos_optimized=config.vos_optimized,
        preprocessing_mode=config.preprocessing_mode,
        preprocessing_percentile_low=config.preprocessing_percentile_low,
        preprocessing_percentile_high=config.preprocessing_percentile_high,
    )
    pred, motion_masks, debug_rows = causal_quality_gate(
        frames, sam2_masks, first_mask, config, scanned_region=scanned_region
    )
    elapsed = time.perf_counter() - start

    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_prediction(config.output_dir / "prediction.mha", pred, frame_image)
        write_prediction(config.output_dir / "sam2_branch.mha", sam2_masks, frame_image)
        if config.save_intermediates:
            write_prediction(config.output_dir / "motion_fallback.mha", motion_masks, frame_image)
        _write_debug_csv(config.output_dir / "fusion_debug.csv", debug_rows)

    return pred, {
        "runtime_seconds": float(elapsed),
        "debug_rows": debug_rows,
        "sam2_shape": list(sam2_masks.shape),
        "prediction_shape": list(pred.shape),
        "device": config.device,
        "motion_method": config.motion_method,
        "checkpoint": os.fspath(config.checkpoint),
        "preprocessing_mode": config.preprocessing_mode,
        "preprocessing_percentile_low": config.preprocessing_percentile_low,
        "preprocessing_percentile_high": config.preprocessing_percentile_high,
    }
