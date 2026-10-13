"""IMPA generative morphology prediction (style transfer) -- skeleton.

IMPA (IMage Perturbation Autoencoder) is a conditional GAN built on StarGANv2.
It decomposes a cell image into *content* (perturbation-agnostic features such
as cell orientation and shape) and *style* (perturbation- or batch-specific
information), then swaps the style to synthesise a counterfactual image.

Because it is a style-transfer model, IMPA -- unlike MorphoDiff or CellFlux --
**requires an input image**: the content encoder must be fed a source cell
(typically a control/DMSO image) before a target perturbation can be rendered.
It cannot sample from pure noise.

This module is currently a **skeleton**: the public interface and result
contract are stable, but inference is stubbed out. The mock path runs today
and records a placeholder manifest; supplying a real checkpoint raises a clear
``NotImplementedError`` at the single inference fill-in point.

Status / blockers
-----------------
Checkpoints are downloaded and verified (``IMPA/checkpoints/``):

* ``nets.ckpt`` (202 MB) -- StarGANv2 weights, loads as ``generator``,
  ``style_encoder``, ``discriminator``, ``mapping_network``.
* ``embeddings.ckpt`` (105 KB) -- perturbation embedding matrix of shape
  ``(26, 1024)``. Note it stores **weights only, no compound names**, so the
  compound-to-index mapping must come from the IMPA data index CSV.

Three things are still missing before real inference can run:

1. **Control images** (the content source). BBBC021 controls live inside 55
   plate-level ZIPs (~41 GB total) and are not downloaded.
2. **Compound-to-index mapping** -- lives in ``bbbc021_df_all.csv``, part of the
   IMPA preprocessing bundle (Zenodo, ~12.5 GB).
3. **The ``theislab/IMPA`` runtime** (StarGANv2 + hydra dependency stack).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# IMPA's BBBC021 configuration renders 128x128 three-channel images.
IMPA_IMAGE_SIZE: int = 128
IMPA_CHANNELS: int = 3


@dataclass(frozen=True)
class IMPAResult:
    """Serializable result contract for one IMPA generation run."""

    condition: str
    output_dir: Path
    predicted_images_dir: Path
    manifest_path: Path
    checkpoint_path: Path | None
    control_images_dir: Path | None
    prediction_count: int
    mode: str  # 'mock' until inference is implemented, then 'real'
    model_loaded: bool
    device: str
    image_size: int
    seed: int | None
    note: str


def predict_impa(
    *,
    condition: str,
    output_dir: Path,
    control_images_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    embedding_path: Path | None = None,
    device: str = 'cpu',
    image_size: int = IMPA_IMAGE_SIZE,
    seed: int | None = None,
) -> IMPAResult:
    """Synthesise a perturbed cell image by style transfer from a control cell.

    Parameters
    ----------
    condition:
        Target perturbation name (a BBBC021 compound).
    output_dir:
        Run root. Predicted images and the manifest are written here.
    control_images_dir:
        Directory of control (DMSO) single-cell images. IMPA **requires** this
        -- it is the content source for style transfer. Not bundled with the
        checkpoint.
    checkpoint_path:
        Path to ``nets.ckpt`` (generator/discriminator weights). When absent or
        missing, runs in mock mode without importing torch.
    embedding_path:
        Path to ``embeddings.ckpt`` (perturbation embeddings). Defaults to a
        sibling of ``checkpoint_path``.
    device:
        Torch device (``cpu`` or a CUDA index).
    image_size:
        Spatial size of the generated square image (default 128).
    seed:
        Optional RNG seed.
    """
    if not condition or not condition.strip():
        raise ValueError('condition must be a non-empty perturbation name (for example a BBBC021 compound).')

    output_dir = output_dir.expanduser().resolve()
    predicted_images_dir = output_dir / 'predicted_images'
    predicted_images_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_path.expanduser().resolve() if checkpoint_path is not None else None
    controls = control_images_dir.expanduser().resolve() if control_images_dir is not None else None
    if controls is not None and not controls.is_dir():
        raise FileNotFoundError(f'control_images_dir does not exist: {controls}')

    model_loaded = checkpoint is not None and checkpoint.is_file()

    if model_loaded:
        _run_style_transfer(
            checkpoint=checkpoint,
            embedding_path=embedding_path.expanduser().resolve() if embedding_path is not None else None,
            controls=controls,
            condition=condition,
            output_dir=predicted_images_dir,
            device=device,
            image_size=image_size,
            seed=seed,
        )

    prediction_count = _count_predictions(predicted_images_dir)
    mode = 'real' if model_loaded else 'mock'
    note = (
        'Placeholder result: IMPA inference is not implemented yet. It additionally requires '
        'control images (content source) that are not bundled with the checkpoint.'
        if not model_loaded
        else f'IMPA generated {prediction_count} image(s) for {condition!r}.'
    )

    manifest_path = output_dir / 'impa_prediction_manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'condition': condition,
                'checkpoint_path': str(checkpoint) if checkpoint is not None else None,
                'control_images_dir': str(controls) if controls is not None else None,
                'predicted_images_dir': str(predicted_images_dir),
                'device': device,
                'image_size': image_size,
                'seed': seed,
                'mode': mode,
                'model_loaded': model_loaded,
                'prediction_count': prediction_count,
                'note': note,
                'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            },
            indent=2,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )

    return IMPAResult(
        condition=condition,
        output_dir=output_dir,
        predicted_images_dir=predicted_images_dir,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint,
        control_images_dir=controls,
        prediction_count=prediction_count,
        mode=mode,
        model_loaded=model_loaded,
        device=device,
        image_size=image_size,
        seed=seed,
        note=note,
    )


def _run_style_transfer(
    *,
    checkpoint: Path,
    embedding_path: Path | None,
    controls: Path | None,
    condition: str,
    output_dir: Path,
    device: str,
    image_size: int,
    seed: int | None,
) -> None:
    """Run IMPA style transfer against loaded weights.

    TODO(skeleton): implement. This is the single fill-in point. The
    implementation should:

    1. Load the IMPA generator from ``checkpoint`` and the perturbation
       embeddings from ``embedding_path`` (``theislab/IMPA`` runtime).
    2. Encode each control image in ``controls`` into the content space.
    3. Look up the target perturbation's style code for ``condition``.
    4. Decode content + style into a counterfactual image and save it.
    """
    if controls is None:
        raise ValueError(
            'IMPA is a style-transfer model: control_images_dir is required as the content source.'
        )
    raise NotImplementedError(
        'IMPA inference is not implemented yet. This is a skeleton: the theislab/IMPA runtime '
        'and the BBBC021 control images are pending.'
    )


def _count_predictions(predicted_images_dir: Path) -> int:
    if not predicted_images_dir.is_dir():
        return 0
    return sum(
        1
        for path in predicted_images_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.tif', '.tiff', '.png', '.npz'}
    )


def impa_result_to_dict(result: IMPAResult) -> dict[str, Any]:
    """Serialize an ``IMPAResult`` into a JSON-ready dictionary."""
    return {
        'condition': result.condition,
        'output_dir': str(result.output_dir),
        'predicted_images_dir': str(result.predicted_images_dir),
        'manifest_path': str(result.manifest_path),
        'checkpoint_path': str(result.checkpoint_path) if result.checkpoint_path is not None else None,
        'control_images_dir': str(result.control_images_dir) if result.control_images_dir is not None else None,
        'prediction_count': result.prediction_count,
        'mode': result.mode,
        'model_loaded': result.model_loaded,
        'device': result.device,
        'image_size': result.image_size,
        'seed': result.seed,
        'note': result.note,
    }
