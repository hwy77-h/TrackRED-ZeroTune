from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


SPLITS = {
    "pre-testing": "trackrad2025_labeled_pre-testing_data",
    "testing": "trackrad2025_labeled_testing_data",
    "training": "trackrad2025_labeled_training_data",
}


@dataclass
class MetaImageInfo:
    dim_size: tuple[int, ...]
    element_type: str
    element_spacing: tuple[float, ...] | None = None
    offset: tuple[float, ...] | None = None
    transform_matrix: tuple[float, ...] | None = None


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_match(directory: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No files matching {patterns} in {directory}")


def read_mha(path: Path):
    path = Path(path)
    try:
        sitk_read_path = _ascii_shadow_file(path) if _needs_ascii_shadow(path) else path
        image = sitk.ReadImage(_sitk_path(sitk_read_path))
        return sitk.GetArrayFromImage(image), image
    except Exception:
        return read_mha_fallback(path)


def write_prediction(path: Path, output: np.ndarray, frame_image=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame_image, MetaImageInfo):
        write_mha_fallback(path, output, frame_image)
        return
    out_img = sitk.GetImageFromArray(output.astype(np.uint8))
    if frame_image is not None:
        out_img.SetSpacing(frame_image.GetSpacing())
        out_img.SetOrigin(frame_image.GetOrigin())
        out_img.SetDirection(frame_image.GetDirection())
    if _needs_ascii_shadow(path):
        tmp_path = Path(tempfile.gettempdir()) / f"zerotune_{uuid.uuid4().hex}{path.suffix}"
        try:
            sitk.WriteImage(out_img, _sitk_path(tmp_path, must_exist=False), useCompression=True)
            shutil.copyfile(tmp_path, path)
            try:
                tmp_path.unlink()
            except OSError:
                pass
        except Exception:
            write_mha_fallback(path, output, None)
    else:
        try:
            sitk.WriteImage(out_img, _sitk_path(path, must_exist=False), useCompression=True)
        except Exception:
            write_mha_fallback(path, output, None)


def _dtype_from_element_type(element_type: str) -> np.dtype:
    mapping = {
        "MET_UCHAR": np.uint8,
        "MET_CHAR": np.int8,
        "MET_USHORT": np.uint16,
        "MET_SHORT": np.int16,
        "MET_UINT": np.uint32,
        "MET_INT": np.int32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }
    if element_type not in mapping:
        raise ValueError(f"Unsupported MHA ElementType: {element_type}")
    return np.dtype(mapping[element_type])


def _parse_meta_value(value: str):
    value = value.strip()
    if " " in value:
        return tuple(value.split())
    return value


def read_mha_fallback(path: Path):
    with open(path, "rb") as f:
        content = f.read()

    marker = b"ElementDataFile = LOCAL"
    marker_idx = content.find(marker)
    if marker_idx < 0:
        raise ValueError(f"Fallback reader only supports LOCAL MHA files: {path}")
    data_start = content.find(b"\n", marker_idx)
    if data_start < 0:
        raise ValueError(f"Malformed MHA header: {path}")
    data_start += 1
    header_text = content[:data_start].decode("ascii", errors="ignore")
    data = content[data_start:]

    meta = {}
    for line in header_text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()

    dim_size = tuple(int(v) for v in meta["DimSize"].split())
    element_type = meta["ElementType"]
    dtype = _dtype_from_element_type(element_type)
    compressed = meta.get("CompressedData", "False").lower() == "true"
    if compressed:
        data = zlib.decompress(data)
    arr = np.frombuffer(data, dtype=dtype)
    expected = int(np.prod(dim_size))
    if arr.size != expected:
        raise ValueError(f"MHA data size mismatch for {path}: got {arr.size}, expected {expected}")
    arr = arr.reshape(tuple(reversed(dim_size)))
    info = MetaImageInfo(
        dim_size=dim_size,
        element_type=element_type,
        element_spacing=tuple(float(v) for v in meta.get("ElementSpacing", "").split()) or None,
        offset=tuple(float(v) for v in meta.get("Offset", "").split()) or None,
        transform_matrix=tuple(float(v) for v in meta.get("TransformMatrix", "").split()) or None,
    )
    return arr.copy(), info


def write_mha_fallback(path: Path, output: np.ndarray, reference: MetaImageInfo | None = None) -> None:
    arr = output.astype(np.uint8)
    dim_size = tuple(reversed(arr.shape))
    spacing = reference.element_spacing if reference and reference.element_spacing else tuple([1.0] * arr.ndim)
    offset = reference.offset if reference and reference.offset else tuple([0.0] * arr.ndim)
    matrix = reference.transform_matrix if reference and reference.transform_matrix else tuple(np.eye(arr.ndim).ravel())
    header = "\n".join(
        [
            "ObjectType = Image",
            f"NDims = {arr.ndim}",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = " + " ".join(f"{v:g}" for v in matrix),
            "Offset = " + " ".join(f"{v:g}" for v in offset),
            "CenterOfRotation = " + " ".join(["0"] * arr.ndim),
            "AnatomicalOrientation = RAI",
            "ElementSpacing = " + " ".join(f"{v:g}" for v in spacing),
            "DimSize = " + " ".join(str(v) for v in dim_size),
            "ElementType = MET_UCHAR",
            "ElementDataFile = LOCAL",
            "",
        ]
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes(order="C"))


def _needs_ascii_shadow(path: Path) -> bool:
    try:
        os.fspath(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _ascii_shadow_file(path: Path) -> Path:
    suffixes = "".join(path.suffixes) or path.suffix
    tmp_path = Path(tempfile.gettempdir()) / f"zerotune_{uuid.uuid4().hex}{suffixes}"
    shutil.copyfile(path, tmp_path)
    return tmp_path


def _sitk_path(path: Path, must_exist: bool = True) -> str:
    """Return a SimpleITK-friendly path on Windows.

    Some SimpleITK builds fail on non-ASCII Windows paths. Use the 8.3 short
    path when available; otherwise fall back to the normal filesystem path.
    """
    path = Path(path)
    if sys.platform != "win32":
        return os.fspath(path)
    target = path if must_exist else path.parent
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buffer = ctypes.create_unicode_buffer(4096)
        result = get_short(os.fspath(target), buffer, len(buffer))
        if result:
            short_target = Path(buffer.value)
            if must_exist:
                return os.fspath(short_target)
            return os.fspath(short_target / path.name)
    except Exception:
        pass
    return os.fspath(path)


PREPROCESSING_MODES = ("trackrad_percentile", "foreground_minmax", "raw_minmax", "none_uint8")


def normalize_frame(
    frame: np.ndarray,
    percentile_low=0.5,
    percentile_high=99.5,
    mode: str = "trackrad_percentile",
) -> np.ndarray:
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unsupported preprocessing mode: {mode}. Expected one of {PREPROCESSING_MODES}.")

    if mode == "none_uint8":
        return np.clip(frame, 0, 255).astype(np.uint8)

    foreground = frame > 0
    if not np.any(foreground):
        return np.zeros_like(frame, dtype=np.uint8)

    if mode == "raw_minmax":
        low = float(np.min(frame))
        high = float(np.max(frame))
    elif mode == "foreground_minmax":
        values = frame[foreground]
        low = float(np.min(values))
        high = float(np.max(values))
    else:
        values = frame[foreground]
        low = float(np.percentile(values, percentile_low))
        high = float(np.percentile(values, percentile_high))

    clipped = np.clip(frame, low, high)
    if high > low:
        out = (clipped - low) / (high - low) * 255.0
    else:
        out = np.zeros_like(clipped)
    out[~foreground] = 0
    return out.astype(np.uint8)


def write_video_frames(
    frames: np.ndarray,
    frame_dir: Path,
    preprocessing_mode: str = "trackrad_percentile",
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> None:
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames.shape[2]):
        frame = normalize_frame(
            frames[:, :, i],
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            mode=preprocessing_mode,
        )
        Image.fromarray(frame).save(frame_dir / f"{i:05d}.jpg", quality=95)


def resolve_case(dataset_root: Path, split: str, case: str | None) -> Path:
    split_dir = dataset_root / SPLITS[split]
    if case:
        return split_dir / case
    cases = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not cases:
        raise FileNotFoundError(f"No cases found in {split_dir}")
    return cases[0]
