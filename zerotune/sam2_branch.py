from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .io_utils import write_video_frames
from .mask_ops import bbox


def mask_to_objects(mask: np.ndarray) -> dict[int, np.ndarray]:
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    return {int(obj_id): mask == obj_id for obj_id in object_ids}


def prompt_geometry(mask: np.ndarray, padding: int = 0):
    box = bbox(mask, padding=padding)
    if box is None:
        raise ValueError("Cannot build prompts from an empty mask.")
    x0, y0, x1, y1 = box
    points = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    return points, labels, np.array([x0, y0, x1, y1], dtype=np.float32)


def combine_objects(per_obj_mask: dict[int, np.ndarray], height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint8)
    for obj_id in sorted(per_obj_mask, reverse=True):
        obj_mask = per_obj_mask[obj_id].reshape(height, width)
        out[obj_mask] = obj_id
    return out


def add_mask_prompts(predictor, state, per_obj_input, frame_idx: int, bbox_padding: int = 0) -> None:
    for obj_id, object_mask in per_obj_input.items():
        points, labels, box = prompt_geometry(object_mask, padding=bbox_padding)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=int(obj_id),
            points=points,
            labels=labels,
            box=box,
        )
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=int(obj_id),
            mask=object_mask,
        )


def run_sam2_causal(
    frames: np.ndarray,
    first_mask: np.ndarray,
    sam2_root: Path,
    checkpoint: Path,
    config: str,
    work_dir: Path,
    device: str = "cpu",
    image_size: int = 512,
    bbox_padding: int = 0,
    vos_optimized: bool = False,
    preprocessing_mode: str = "trackrad_percentile",
    preprocessing_percentile_low: float = 0.5,
    preprocessing_percentile_high: float = 99.5,
) -> np.ndarray:
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    from sam2.build_sam import build_sam2_video_predictor

    frame_dir = work_dir / "frames"
    write_video_frames(
        frames,
        frame_dir,
        preprocessing_mode=preprocessing_mode,
        percentile_low=preprocessing_percentile_low,
        percentile_high=preprocessing_percentile_high,
    )

    predictor = build_sam2_video_predictor(
        config,
        str(checkpoint),
        device=device,
        mode="eval",
        hydra_overrides_extra=[
            f"++model.image_size={image_size}",
            f"++model.memory_attention.layer.self_attention.feat_sizes=[{image_size // 16},{image_size // 16}]",
            f"++model.memory_attention.layer.cross_attention.feat_sizes=[{image_size // 16},{image_size // 16}]",
        ],
        apply_postprocessing=True,
        vos_optimized=vos_optimized,
    )

    state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)
    height = state["video_height"]
    width = state["video_width"]
    output = np.zeros(frames.shape, dtype=np.uint8)
    per_obj_input = mask_to_objects(first_mask.astype(np.uint8))
    if not per_obj_input:
        raise RuntimeError("First-frame target mask is empty.")
    add_mask_prompts(predictor, state, per_obj_input, frame_idx=0, bbox_padding=bbox_padding)

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        per_obj_output = {
            int(obj_id): (mask_logits[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(obj_ids)
        }
        output[:, :, frame_idx] = combine_objects(per_obj_output, height, width)
    return output
