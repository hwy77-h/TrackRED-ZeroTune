from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mask_ops import iou, largest_component_ratio, mask_center


@dataclass
class GateConfig:
    gate_mode: str = "coldstart_adaptive"
    coldstart_frames: int = 5
    hybrid_frames: int = 10
    rolling_window: int = 8
    stable_area_min: float = 0.55
    stable_area_max: float = 1.80
    severe_area_min: float = 0.35
    severe_area_max: float = 2.50
    stable_iou_min: float = 0.35
    severe_iou_max: float = 0.20
    component_ratio_min: float = 0.65
    min_area_px: int = 20
    cotracker_visibility_min: float = 0.45
    coldstart_area_min: float = 0.70
    coldstart_area_max: float = 1.40
    coldstart_iou_min: float = 0.50
    coldstart_component_ratio_min: float = 0.80
    adaptive_mad_scale: float = 3.0
    max_continuous_red: int = 5
    trusted_iou_min: float = 0.90
    trusted_area_min: float = 0.85
    trusted_area_max: float = 1.15
    trusted_component_ratio_min: float = 0.95
    collapse_area_min: float = 0.80
    collapse_iou_max: float = 0.94


@dataclass
class TrackingState:
    seed_area: int
    last_stable_mask: np.ndarray
    last_stable_area: int
    last_stable_center: np.ndarray | None
    previous_mask: np.ndarray
    continuous_red_count: int = 0
    lost_mode: bool = False
    area_history: list[int] = field(default_factory=list)
    center_jump_history: list[float] = field(default_factory=list)

    @classmethod
    def from_first_mask(cls, first_mask: np.ndarray):
        area = int((first_mask > 0).sum())
        center = mask_center(first_mask)
        return cls(
            seed_area=area,
            last_stable_mask=(first_mask > 0).astype(np.uint8),
            last_stable_area=area,
            last_stable_center=center,
            previous_mask=(first_mask > 0).astype(np.uint8),
            area_history=[area],
        )

    def update_stable(self, mask: np.ndarray, center_jump: float, rolling_window: int) -> None:
        self.last_stable_mask = (mask > 0).astype(np.uint8)
        self.last_stable_area = int(self.last_stable_mask.sum())
        self.last_stable_center = mask_center(self.last_stable_mask)
        self.previous_mask = self.last_stable_mask
        self.continuous_red_count = 0
        self.lost_mode = False
        self.area_history.append(self.last_stable_area)
        self.center_jump_history.append(float(center_jump))
        keep = max(rolling_window, 1)
        self.area_history = self.area_history[-keep:]
        self.center_jump_history = self.center_jump_history[-keep:]

    def update_failure(self, mask: np.ndarray, config: GateConfig) -> None:
        self.previous_mask = (mask > 0).astype(np.uint8)
        self.continuous_red_count += 1
        if self.continuous_red_count > config.max_continuous_red:
            self.lost_mode = True

    def stable_history_size(self) -> int:
        return len(self.area_history)


def _center_jump(mask: np.ndarray, reference_center: np.ndarray | None) -> float:
    center = mask_center(mask)
    if center is None or reference_center is None:
        return float("inf")
    return float(np.linalg.norm(center - reference_center))


def _median_mad(values: list[int] | list[float], window: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values[-window:], dtype=np.float32)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median, max(mad, 1e-6)


def _select_gate_mode(frame_idx: int, state: TrackingState, config: GateConfig) -> str:
    if config.gate_mode != "coldstart_adaptive":
        return "fixed"
    if frame_idx <= config.coldstart_frames or state.stable_history_size() < 3:
        return "coldstart"
    if frame_idx <= config.hybrid_frames:
        return "hybrid"
    return "adaptive"


def _adaptive_area_limits(state: TrackingState, config: GateConfig) -> tuple[float, float, float | None, float | None]:
    median, mad = _median_mad(state.area_history, config.rolling_window)
    if median is None or mad is None:
        return config.stable_area_min, config.stable_area_max, median, mad
    lower_area = max(config.min_area_px, median - config.adaptive_mad_scale * mad)
    upper_area = median + config.adaptive_mad_scale * mad
    return lower_area / max(1, state.last_stable_area), upper_area / max(1, state.last_stable_area), median, mad


def _adaptive_center_limit(state: TrackingState, config: GateConfig) -> tuple[float, float | None, float | None]:
    median, mad = _median_mad(state.center_jump_history, config.rolling_window)
    fallback = max(20.0, 0.35 * np.sqrt(max(1, state.seed_area)))
    if median is None or mad is None:
        return fallback, median, mad
    return max(8.0, median + config.adaptive_mad_scale * mad), median, mad


def classify_mask(
    mask: np.ndarray,
    state: TrackingState,
    config: GateConfig,
    frame_idx: int = 0,
    cotracker_visibility: float | None = None,
) -> dict:
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    prev_iou = iou(mask, state.previous_mask)
    area_ratio = area / max(1, state.last_stable_area)
    center_jump = _center_jump(mask, state.last_stable_center)
    component_ratio = largest_component_ratio(mask)
    gate_mode = _select_gate_mode(frame_idx, state, config)

    fixed_center_limit = max(20.0, 0.35 * np.sqrt(max(1, state.seed_area)))
    severe_center_limit = max(40.0, 0.75 * np.sqrt(max(1, state.seed_area)))
    coldstart_center_limit = max(12.0, 0.20 * np.sqrt(max(1, state.seed_area)))
    adaptive_area_min, adaptive_area_max, area_median, area_mad = _adaptive_area_limits(state, config)
    adaptive_center_limit, center_median, center_mad = _adaptive_center_limit(state, config)

    if gate_mode == "coldstart":
        stable_area_min = config.coldstart_area_min
        stable_area_max = config.coldstart_area_max
        stable_iou_min = config.coldstart_iou_min
        stable_center_limit = coldstart_center_limit
        component_ratio_min = config.coldstart_component_ratio_min
    elif gate_mode == "hybrid":
        stable_area_min = max(config.stable_area_min, adaptive_area_min)
        stable_area_max = min(config.stable_area_max, adaptive_area_max)
        stable_iou_min = config.stable_iou_min
        stable_center_limit = min(fixed_center_limit, adaptive_center_limit)
        component_ratio_min = config.component_ratio_min
    elif gate_mode == "adaptive":
        stable_area_min = adaptive_area_min
        stable_area_max = adaptive_area_max
        stable_iou_min = config.stable_iou_min
        stable_center_limit = adaptive_center_limit
        component_ratio_min = config.component_ratio_min
    else:
        stable_area_min = config.stable_area_min
        stable_area_max = config.stable_area_max
        stable_iou_min = config.stable_iou_min
        stable_center_limit = fixed_center_limit
        component_ratio_min = config.component_ratio_min

    red_reasons = []
    if area < max(config.min_area_px, int(0.10 * max(1, state.seed_area))):
        red_reasons.append("empty_or_tiny_area")
    if area_ratio < config.severe_area_min:
        red_reasons.append("severe_area_drop")
    if area_ratio > config.severe_area_max:
        red_reasons.append("severe_area_growth")
    if area_ratio < config.collapse_area_min and prev_iou < config.collapse_iou_max:
        red_reasons.append("area_collapse")
    if prev_iou < config.severe_iou_max:
        red_reasons.append("severe_iou_drop")
    if center_jump > severe_center_limit:
        red_reasons.append("severe_center_jump")
    if component_ratio < config.component_ratio_min and area > 0:
        red_reasons.append("fragmented_mask")
    red = bool(red_reasons)
    green = (
        not red
        and stable_area_min <= area_ratio <= stable_area_max
        and prev_iou >= stable_iou_min
        and center_jump <= stable_center_limit
        and component_ratio >= component_ratio_min
    )
    trusted_stable = (
        not red
        and gate_mode in {"hybrid", "adaptive", "fixed"}
        and config.trusted_area_min <= area_ratio <= config.trusted_area_max
        and prev_iou >= config.trusted_iou_min
        and component_ratio >= config.trusted_component_ratio_min
    )
    cotracker_ok = cotracker_visibility is None or cotracker_visibility >= config.cotracker_visibility_min

    if green or trusted_stable:
        decision = "green"
    elif red or not cotracker_ok:
        decision = "red"
    else:
        decision = "yellow"

    return {
        "decision": decision,
        "gate_mode": gate_mode,
        "area": area,
        "area_ratio": float(area_ratio),
        "rolling_area_median": "" if area_median is None else float(area_median),
        "rolling_area_mad": "" if area_mad is None else float(area_mad),
        "rolling_center_jump_median": "" if center_median is None else float(center_median),
        "rolling_center_jump_mad": "" if center_mad is None else float(center_mad),
        "prev_iou": float(prev_iou),
        "center_jump": float(center_jump),
        "largest_component_ratio": float(component_ratio),
        "trusted_stable_override": bool(trusted_stable and not green),
        "gate_reason": "green" if green else "trusted_stable" if trusted_stable else ";".join(red_reasons) if red_reasons else "uncertain_yellow",
        "stable_area_min": float(stable_area_min),
        "stable_area_max": float(stable_area_max),
        "stable_center_limit": float(stable_center_limit),
        "severe_center_limit": float(severe_center_limit),
        "cotracker_visibility": "" if cotracker_visibility is None else float(cotracker_visibility),
        "continuous_red_count": int(state.continuous_red_count),
        "stable_history_size": int(state.stable_history_size()),
        "lost_mode": bool(state.lost_mode),
    }
