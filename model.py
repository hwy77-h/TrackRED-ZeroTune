from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from zerotune.pipeline import PipelineConfig, run_causal_zerotune


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAM2_ROOT = PROJECT_ROOT.parent / "Reproduce" / "trackrad" / "TrackRAD2025-main"
DEFAULT_CHECKPOINT = DEFAULT_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:
    """TrackRAD-compatible causal ZeroTune entrypoint."""
    sam2_root = _path_from_env("ZEROTUNE_SAM2_ROOT", DEFAULT_SAM2_ROOT)
    checkpoint = _path_from_env("ZEROTUNE_SAM2_CHECKPOINT", DEFAULT_CHECKPOINT)
    config_name = os.environ.get("ZEROTUNE_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
    device = os.environ.get("ZEROTUNE_DEVICE", _default_device())
    work_dir = _path_from_env("ZEROTUNE_WORK_DIR", PROJECT_ROOT / "work" / "grand_challenge")
    output_dir_env = os.environ.get("ZEROTUNE_OUTPUT_DIR")
    output_dir = Path(output_dir_env) if output_dir_env else None

    cfg = PipelineConfig(
        sam2_root=sam2_root,
        checkpoint=checkpoint,
        config=config_name,
        work_dir=work_dir,
        output_dir=output_dir,
        device=device,
        image_size=int(os.environ.get("ZEROTUNE_IMAGE_SIZE", "512")),
        bbox_padding=int(os.environ.get("ZEROTUNE_BBOX_PADDING", "0")),
        preprocessing_mode=os.environ.get("ZEROTUNE_PREPROCESSING_MODE", "trackrad_percentile"),
        preprocessing_percentile_low=float(os.environ.get("ZEROTUNE_PREPROCESSING_PERCENTILE_LOW", "0.5")),
        preprocessing_percentile_high=float(os.environ.get("ZEROTUNE_PREPROCESSING_PERCENTILE_HIGH", "99.5")),
        motion_method=os.environ.get("ZEROTUNE_MOTION_METHOD", "tvl1"),
        save_intermediates=os.environ.get("ZEROTUNE_SAVE_INTERMEDIATES", "0") == "1",
        min_component_area=int(os.environ.get("ZEROTUNE_MIN_COMPONENT_AREA", "25")),
        roi_padding=int(os.environ.get("ZEROTUNE_ROI_PADDING", "32")),
        gate_mode=os.environ.get("ZEROTUNE_GATE_MODE", "coldstart_adaptive"),
        coldstart_frames=int(os.environ.get("ZEROTUNE_COLDSTART_FRAMES", "5")),
        hybrid_frames=int(os.environ.get("ZEROTUNE_HYBRID_FRAMES", "10")),
        rolling_window=int(os.environ.get("ZEROTUNE_ROLLING_WINDOW", "8")),
        max_continuous_red=int(os.environ.get("ZEROTUNE_MAX_CONTINUOUS_RED", "5")),
        lost_mode=os.environ.get("ZEROTUNE_LOST_MODE", "hold"),
        trusted_iou_min=float(os.environ.get("ZEROTUNE_TRUSTED_IOU_MIN", "0.90")),
        trusted_area_min=float(os.environ.get("ZEROTUNE_TRUSTED_AREA_MIN", "0.85")),
        trusted_area_max=float(os.environ.get("ZEROTUNE_TRUSTED_AREA_MAX", "1.15")),
        trusted_component_ratio_min=float(os.environ.get("ZEROTUNE_TRUSTED_COMPONENT_RATIO_MIN", "0.95")),
        collapse_area_min=float(os.environ.get("ZEROTUNE_COLLAPSE_AREA_MIN", "0.80")),
        collapse_iou_max=float(os.environ.get("ZEROTUNE_COLLAPSE_IOU_MAX", "0.94")),
        red_sam2_reacquire=os.environ.get("ZEROTUNE_RED_SAM2_REACQUIRE", "1") == "1",
        red_reacquire_prev_iou_min=float(os.environ.get("ZEROTUNE_RED_REACQUIRE_PREV_IOU_MIN", "0.65")),
        red_reacquire_prev_area_min=float(os.environ.get("ZEROTUNE_RED_REACQUIRE_PREV_AREA_MIN", "0.50")),
        red_reacquire_prev_area_max=float(os.environ.get("ZEROTUNE_RED_REACQUIRE_PREV_AREA_MAX", "1.80")),
        red_reacquire_seed_area_min=float(os.environ.get("ZEROTUNE_RED_REACQUIRE_SEED_AREA_MIN", "0.30")),
        red_reacquire_seed_area_max=float(os.environ.get("ZEROTUNE_RED_REACQUIRE_SEED_AREA_MAX", "3.00")),
        red_reacquire_center_max=float(os.environ.get("ZEROTUNE_RED_REACQUIRE_CENTER_MAX", "12.0")),
        red_reacquire_component_min=float(os.environ.get("ZEROTUNE_RED_REACQUIRE_COMPONENT_MIN", "0.95")),
        thorax_empty_on_sam2_empty=os.environ.get("ZEROTUNE_THORAX_EMPTY_ON_SAM2_EMPTY", "1") == "1",
        empty_consecutive_frames=int(os.environ.get("ZEROTUNE_EMPTY_CONSECUTIVE_FRAMES", "1")),
        thorax_empty_on_long_red_tiny=os.environ.get("ZEROTUNE_THORAX_EMPTY_ON_LONG_RED_TINY", "1") == "1",
        long_red_empty_frames=int(os.environ.get("ZEROTUNE_LONG_RED_EMPTY_FRAMES", "10")),
        long_red_empty_seed_ratio=float(os.environ.get("ZEROTUNE_LONG_RED_EMPTY_SEED_RATIO", "0.60")),
        thorax_empty_on_red_collapse=os.environ.get("ZEROTUNE_THORAX_EMPTY_ON_RED_COLLAPSE", "1") == "1",
        red_collapse_empty_frames=int(os.environ.get("ZEROTUNE_RED_COLLAPSE_EMPTY_FRAMES", "7")),
        red_collapse_empty_seed_ratio=float(os.environ.get("ZEROTUNE_RED_COLLAPSE_EMPTY_SEED_RATIO", "0.85")),
        red_collapse_empty_prev_iou_max=float(os.environ.get("ZEROTUNE_RED_COLLAPSE_EMPTY_PREV_IOU_MAX", "0.35")),
        red_collapse_empty_seed_center_max=float(os.environ.get("ZEROTUNE_RED_COLLAPSE_EMPTY_SEED_CENTER_MAX", "18.0")),
        thorax_reentry_after_absence=os.environ.get("ZEROTUNE_THORAX_REENTRY_AFTER_ABSENCE", "1") == "1",
        reentry_min_absence_frames=int(os.environ.get("ZEROTUNE_REENTRY_MIN_ABSENCE_FRAMES", "1")),
        reentry_seed_area_min=float(os.environ.get("ZEROTUNE_REENTRY_SEED_AREA_MIN", "0.80")),
        reentry_seed_area_max=float(os.environ.get("ZEROTUNE_REENTRY_SEED_AREA_MAX", "1.30")),
        reentry_seed_center_max=float(os.environ.get("ZEROTUNE_REENTRY_SEED_CENTER_MAX", "12.0")),
        reentry_component_min=float(os.environ.get("ZEROTUNE_REENTRY_COMPONENT_MIN", "0.95")),
        raw_relocation_rescue=os.environ.get("ZEROTUNE_RAW_RELOCATION_RESCUE", "1") == "1",
        raw_relocation_min_red=int(os.environ.get("ZEROTUNE_RAW_RELOCATION_MIN_RED", "2")),
        raw_relocation_prev_iou_min=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_PREV_IOU_MIN", "0.05")),
        raw_relocation_prev_iou_max=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_PREV_IOU_MAX", "0.20")),
        raw_relocation_temporal_iou_min=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_TEMPORAL_IOU_MIN", "0.10")),
        raw_relocation_area_min=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_AREA_MIN", "0.80")),
        raw_relocation_area_max=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_AREA_MAX", "2.00")),
        raw_relocation_center_min=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_CENTER_MIN", "18.0")),
        raw_relocation_component_min=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_COMPONENT_MIN", "0.95")),
        raw_relocation_thorax_min_red=int(os.environ.get("ZEROTUNE_RAW_RELOCATION_THORAX_MIN_RED", "999")),
        raw_relocation_thorax_area_min=float(os.environ.get("ZEROTUNE_RAW_RELOCATION_THORAX_AREA_MIN", "0.80")),
        lost_raw_reacquire=os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE", "1") == "1",
        lost_raw_reacquire_min_red=int(os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE_MIN_RED", "2")),
        lost_raw_reacquire_seed_area_min=float(os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE_SEED_AREA_MIN", "0.70")),
        lost_raw_reacquire_seed_area_max=float(os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE_SEED_AREA_MAX", "2.20")),
        lost_raw_reacquire_thorax_seed_area_min=float(
            os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE_THORAX_SEED_AREA_MIN", "0.90")
        ),
        lost_raw_reacquire_prev_iou_max=float(os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE_PREV_IOU_MAX", "0.20")),
        lost_raw_reacquire_seed_center_max=float(
            os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE_SEED_CENTER_MAX", "38.0")
        ),
        lost_raw_reacquire_component_min=float(os.environ.get("ZEROTUNE_LOST_RAW_REACQUIRE_COMPONENT_MIN", "0.95")),
        stable_update_admission=os.environ.get("ZEROTUNE_STABLE_UPDATE_ADMISSION", "1") == "1",
        admission_nonthorax_only=os.environ.get("ZEROTUNE_ADMISSION_NONTHORAX_ONLY", "1") == "1",
        admission_reacquire_seed_area_min=float(os.environ.get("ZEROTUNE_ADMISSION_REACQUIRE_SEED_AREA_MIN", "1.00")),
        admission_reacquire_seed_area_max=float(os.environ.get("ZEROTUNE_ADMISSION_REACQUIRE_SEED_AREA_MAX", "1.50")),
        admission_reacquire_temporal_iou_min=float(os.environ.get("ZEROTUNE_ADMISSION_REACQUIRE_TEMPORAL_IOU_MIN", "0.65")),
        admission_component_min=float(os.environ.get("ZEROTUNE_ADMISSION_COMPONENT_MIN", "0.95")),
        admission_raw_relocation_seed_area_min=float(
            os.environ.get("ZEROTUNE_ADMISSION_RAW_RELOCATION_SEED_AREA_MIN", "0.80")
        ),
        admission_raw_relocation_seed_area_max=float(
            os.environ.get("ZEROTUNE_ADMISSION_RAW_RELOCATION_SEED_AREA_MAX", "2.00")
        ),
        admission_raw_relocation_temporal_iou_min=float(
            os.environ.get("ZEROTUNE_ADMISSION_RAW_RELOCATION_TEMPORAL_IOU_MIN", "0.10")
        ),
        admission_raw_relocation_component_min=float(
            os.environ.get("ZEROTUNE_ADMISSION_RAW_RELOCATION_COMPONENT_MIN", "0.95")
        ),
        admission_lost_raw_seed_area_min=float(os.environ.get("ZEROTUNE_ADMISSION_LOST_RAW_SEED_AREA_MIN", "0.80")),
        admission_lost_raw_seed_area_max=float(os.environ.get("ZEROTUNE_ADMISSION_LOST_RAW_SEED_AREA_MAX", "1.50")),
        admission_lost_raw_temporal_iou_min=float(
            os.environ.get("ZEROTUNE_ADMISSION_LOST_RAW_TEMPORAL_IOU_MIN", "0.10")
        ),
        admission_lost_raw_component_min=float(os.environ.get("ZEROTUNE_ADMISSION_LOST_RAW_COMPONENT_MIN", "0.95")),
    )
    try:
        prediction, _ = run_causal_zerotune(
            frames=frames,
            target=target,
            frame_rate=frame_rate,
            magnetic_field_strength=magnetic_field_strength,
            scanned_region=scanned_region,
            config=cfg,
        )
        return prediction.astype(np.uint8)
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
