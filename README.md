# TrackRAD-ZeroTune Algorithm

This folder is the minimal Grand Challenge submission package for the
TrackRAD-ZeroTune causal online tracker.

## Contents

```text
Dockerfile
inference_GC.py
model.py
zerotune/
requirements.txt
```

The algorithm uses a zero-fine-tuning pipeline:

```text
generic SAM2 causal propagation
-> adaptive drift gate
-> recovery / fallback
-> output.mha
```

Offline research scripts, reports, local outputs, datasets, and checkpoints are
intentionally excluded from this folder.

## Grand Challenge I/O

The container entrypoint is:

```text
python inference_GC.py
```

Expected input paths:

```text
/input/images/mri-linacs
/input/images/mri-linac-target
/input/b-field-strength.json
/input/frame-rate.json
/input/scanned-region.json
```

Output path:

```text
/output/images/mri-linac-series-targets/output.mha
```

## Checkpoint

The Dockerfile expects the generic SAM2 checkpoint at:

```text
/opt/ml/model/sam2.1_hiera_small.pt
```

This checkpoint is not included in the GitHub repository. Provide it through the
Grand Challenge model storage or adjust `ZEROTUNE_SAM2_CHECKPOINT`.

Local source checkpoint used during development:

```text
F:\2026summer实习\TrackRAD\Reproduce\trackrad\TrackRAD2025-main\checkpoints\sam2.1_hiera_small.pt
```

## Notes

- This package does not use TrackRAD fine-tuned weights.
- The online entrypoint does not call bidirectional SAM2, full-sequence
  CoTracker, DINOv3, or offline pseudo-label code.
- The Dockerfile clones SAM2 during image build. If the platform build has no
  network access, vendor the SAM2 source into the repository and update
  `ZEROTUNE_SAM2_ROOT`.
