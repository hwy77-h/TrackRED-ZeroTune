from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from model import run_algorithm
from zerotune.io_utils import read_mha, write_prediction


INPUT_SERIES_DIR = Path("/input/images/mri-linacs")
INPUT_TARGET_DIR = Path("/input/images/mri-linac-target")
INPUT_B_FIELD = Path("/input/b-field-strength.json")
INPUT_FRAME_RATE = Path("/input/frame-rate.json")
INPUT_SCANNED_REGION = Path("/input/scanned-region.json")
OUTPUT_DIR = Path("/output/images/mri-linac-series-targets")
OUTPUT_PATH = OUTPUT_DIR / "output.mha"


def _first_image_file(directory: Path) -> Path:
    for pattern in ("*.mha", "*.mhd", "*.tiff", "*.tif"):
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No image file found in {directory}")


def _load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_3d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {array.shape}")
    return array


class ZeroTuneTrackRADContainer:
    def __init__(self) -> None:
        self.series_dir = INPUT_SERIES_DIR
        self.target_dir = INPUT_TARGET_DIR
        self.b_field_path = INPUT_B_FIELD
        self.frame_rate_path = INPUT_FRAME_RATE
        self.scanned_region_path = INPUT_SCANNED_REGION
        self.output_dir = OUTPUT_DIR
        self.output_path = OUTPUT_PATH

    def run(self) -> None:
        start = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        series_path = _first_image_file(self.series_dir)
        target_path = _first_image_file(self.target_dir)
        frames, frame_image = read_mha(series_path)
        target, _ = read_mha(target_path)
        frames = _ensure_3d(frames)
        target = _ensure_3d(target).astype(np.uint8)

        frame_rate = _load_json(self.frame_rate_path)
        magnetic_field_strength = _load_json(self.b_field_path)
        scanned_region = _load_json(self.scanned_region_path)

        print(f"ZeroTune Grand Challenge inference")
        print(f"Series: {series_path} shape={frames.shape} dtype={frames.dtype}")
        print(f"Target: {target_path} shape={target.shape} dtype={target.dtype}")
        print(
            "Metadata: "
            f"frame_rate={frame_rate}, "
            f"magnetic_field_strength={magnetic_field_strength}, "
            f"scanned_region={scanned_region}"
        )

        prediction = run_algorithm(
            frames=frames,
            target=target,
            frame_rate=frame_rate,
            magnetic_field_strength=magnetic_field_strength,
            scanned_region=scanned_region,
        )
        if prediction.shape != frames.shape:
            raise ValueError(f"Prediction shape {prediction.shape} does not match input frames {frames.shape}")
        prediction = prediction.astype(np.uint8, copy=False)

        write_prediction(self.output_path, prediction, frame_image)
        elapsed = time.perf_counter() - start
        print(f"Output saved to {self.output_path}")
        print(f"Total runtime: {elapsed:.2f}s ({elapsed / max(1, frames.shape[2]):.4f}s/frame)")


if __name__ == "__main__":
    ZeroTuneTrackRADContainer().run()
