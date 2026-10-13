"""CellFlux generative morphology prediction (flow matching).

CellFlux is a flow-matching model that transports a starting point to a
perturbed cell state by integrating a learned velocity field. Unlike
MorphoDiff (latent diffusion), it does not denoise; it solves an ODE.

Inference is conditioned on a chemical perturbation: the compound's 1024-bit
Morgan fingerprint is looked up from a CSV, projected through a linear layer,
and added to the timestep embedding of a UNet velocity field.

Two paths mirror :mod:`cellpaint_pipeline.generative.morphodiff`:

* **mock path** -- no checkpoint -> placeholder manifest, no torch import.
* **real path** -- valid checkpoint -> load UNet + fingerprint + ODE sampling.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# CellFlux BBBC021 operates on 3-channel 128x128 images.
CELLFLUX_IMAGE_SIZE: int = 128
CELLFLUX_CHANNELS: int = 3
# BBBC021 Morgan fingerprint width (matches emb_fp.csv).
CELLFLUX_FINGERPRINT_DIM: int = 1024
DEFAULT_NUM_STEPS: int = 50

_EMBEDDING_CSV = 'emb_fp.csv'

# Where the ODE integration starts. Upstream CellFlux trains and evaluates on
# (control, treated) image pairs and transports the *control* distribution to
# the perturbed one, so the official BBBC021 evaluation runs with
# USE_INITIAL_CONTROL_PLUS_NOISE. Starting from pure noise is a different
# regime the released checkpoint was not fitted for.
USE_INITIAL_NOISE = 0                 # x_0 = randn()          -- framework default
USE_INITIAL_CONTROL = 1               # x_0 = control image
USE_INITIAL_CONTROL_PLUS_NOISE = 2    # x_0 = control image + noise_level * randn()

DEFAULT_NOISE_LEVEL: float = 1.0      # matches --noise_level=1.0 in upstream eval


@dataclass(frozen=True)
class CellFluxResult:
    """Serializable result contract for one CellFlux generation run."""

    condition: str
    output_dir: Path
    predicted_images_dir: Path
    manifest_path: Path
    checkpoint_path: Path | None
    prediction_count: int
    mode: str  # 'mock' until a real checkpoint is supplied, then 'real'
    model_loaded: bool
    device: str
    num_steps: int
    image_size: int
    cfg_scale: float
    seed: int | None
    note: str
    use_initial: int = USE_INITIAL_NOISE
    noise_level: float = DEFAULT_NOISE_LEVEL
    control_image_path: Path | None = None


def predict_cellflux(
    *,
    condition: str,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    embedding_csv: Path | None = None,
    device: str = 'cpu',
    num_steps: int = DEFAULT_NUM_STEPS,
    image_size: int = CELLFLUX_IMAGE_SIZE,
    cfg_scale: float = 0.2,
    seed: int | None = None,
    control_image_path: Path | None = None,
    use_initial: int = USE_INITIAL_NOISE,
    noise_level: float = DEFAULT_NOISE_LEVEL,
) -> CellFluxResult:
    """Generate a perturbed cell image via CellFlux flow matching.

    Parameters
    ----------
    condition:
        Target perturbation name. Must exist in the fingerprint CSV (35 BBBC021
        compounds such as ``taxol``, ``nocodazole``, ``cytochalasin-d``).
    output_dir:
        Run root. Predicted images and manifest are written here.
    checkpoint_path:
        Path to ``checkpoint.pth`` (a torch checkpoint dict with a ``model``
        key). When absent or missing, runs in mock mode without importing torch.
    embedding_csv:
        Path to ``emb_fp.csv`` (compound -> 1024-bit fingerprint). Defaults to
        ``emb_fp.csv`` next to the checkpoint.
    device:
        Torch device (``cpu`` or a CUDA index).
    num_steps:
        Number of ODE integration steps (time grid length).
    image_size:
        Spatial size of the generated square image (default 128).
    cfg_scale:
        Classifier-free-guidance scale (0 disables CFG).
    seed:
        Optional RNG seed.
    control_image_path:
        Source cell image for the ODE start point. Required when ``use_initial``
        is 1 or 2. Accepts ``.npy`` (upstream's format, pixel values 0-255) or a
        regular image file such as PNG/TIFF.
    use_initial:
        Where integration starts. ``USE_INITIAL_NOISE`` (0, default) starts from
        pure noise; ``USE_INITIAL_CONTROL`` (1) starts from ``control_image_path``;
        ``USE_INITIAL_CONTROL_PLUS_NOISE`` (2) adds ``noise_level``-scaled Gaussian
        noise to it. **Upstream evaluates BBBC021 with mode 2** -- CellFlux is a
        control-to-treated transport, so mode 0 is outside its trained regime.
    noise_level:
        Gaussian noise scale added to the control image in mode 2.
    """
    if not condition or not condition.strip():
        raise ValueError('condition must be a non-empty perturbation name (for example a BBBC021 compound).')
    if use_initial not in (USE_INITIAL_NOISE, USE_INITIAL_CONTROL, USE_INITIAL_CONTROL_PLUS_NOISE):
        raise ValueError(
            f'use_initial must be 0, 1 or 2 (got {use_initial!r}). '
            '0 = pure noise, 1 = control image, 2 = control image + noise.'
        )
    if use_initial in (USE_INITIAL_CONTROL, USE_INITIAL_CONTROL_PLUS_NOISE) and control_image_path is None:
        raise ValueError(
            f'use_initial={use_initial} needs a control_image_path: CellFlux transports a control '
            'cell to the perturbed state, so the ODE has to start from a real cell image.'
        )

    output_dir = output_dir.expanduser().resolve()
    predicted_images_dir = output_dir / 'predicted_images'
    predicted_images_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_path.expanduser().resolve() if checkpoint_path is not None else None
    model_loaded = checkpoint is not None and checkpoint.is_file()

    if model_loaded:
        _run_inference(
            checkpoint=checkpoint,
            embedding_csv=embedding_csv.expanduser().resolve() if embedding_csv is not None else None,
            condition=condition,
            output_dir=predicted_images_dir,
            device=device,
            num_steps=num_steps,
            image_size=image_size,
            cfg_scale=cfg_scale,
            seed=seed,
            control_image_path=control_image_path.expanduser().resolve() if control_image_path is not None else None,
            use_initial=use_initial,
            noise_level=noise_level,
        )

    prediction_count = _count_predictions(predicted_images_dir)
    mode = 'real' if model_loaded else 'mock'
    note = (
        'Placeholder result: CellFlux inference requires a valid checkpoint_path plus the torch/flow_matching runtime.'
        if not model_loaded
        else f'CellFlux generated {prediction_count} image(s) for {condition!r}.'
    )

    manifest_path = output_dir / 'cellflux_prediction_manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'condition': condition,
                'checkpoint_path': str(checkpoint) if checkpoint is not None else None,
                'predicted_images_dir': str(predicted_images_dir),
                'device': device,
                'num_steps': num_steps,
                'image_size': image_size,
                'cfg_scale': cfg_scale,
                'seed': seed,
                'use_initial': use_initial,
                'noise_level': noise_level,
                'control_image_path': str(control_image_path) if control_image_path is not None else None,
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

    return CellFluxResult(
        condition=condition,
        output_dir=output_dir,
        predicted_images_dir=predicted_images_dir,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint,
        prediction_count=prediction_count,
        mode=mode,
        model_loaded=model_loaded,
        device=device,
        num_steps=num_steps,
        image_size=image_size,
        cfg_scale=cfg_scale,
        seed=seed,
        note=note,
        use_initial=use_initial,
        noise_level=noise_level,
        control_image_path=control_image_path,
    )


def _run_inference(
    *,
    checkpoint: Path,
    embedding_csv: Path | None,
    condition: str,
    output_dir: Path,
    device: str,
    num_steps: int,
    image_size: int,
    cfg_scale: float,
    seed: int | None,
    control_image_path: Path | None,
    use_initial: int,
    noise_level: float,
) -> None:
    import torch
    from flow_matching.solver import ODESolver
    from flow_matching.utils import ModelWrapper

    from cellpaint_pipeline.generative.cellflux.model_configs import MODEL_CONFIGS
    from cellpaint_pipeline.generative.cellflux.unet import UNetModel

    class CFGScaledModel(ModelWrapper):
        """CFG wrapper: expands scalar t to batch, applies guidance when set."""

        def forward(self, x, t, cfg_scale, extra):
            t = torch.zeros(x.shape[0], device=x.device) + t
            if cfg_scale != 0.0:
                conditional = self.model(x, t, extra=extra)
                condition_free = self.model(x, t, extra={})
                return (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
            return self.model(x, t, extra=extra)

    # 1. model
    model = UNetModel(**MODEL_CONFIGS['bbbc021'])
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # 2. fingerprint
    fingerprint = _lookup_fingerprint(_resolve_embedding_csv(checkpoint, embedding_csv), condition)
    z = fingerprint.to(device).unsqueeze(0)  # (1, 1024)

    # 3. start point
    if seed is not None:
        torch.manual_seed(seed)
    x_0 = _build_initial_state(
        use_initial=use_initial,
        control_image_path=control_image_path,
        image_size=image_size,
        device=device,
        noise_level=noise_level,
    )

    # 4. ODE sampling
    solver = ODESolver(velocity_model=CFGScaledModel(model))
    time_grid = torch.linspace(0, 1, num_steps, device=device)
    samples = solver.sample(
        x_init=x_0,
        step_size=None,
        method='heun2',
        time_grid=time_grid,
        cfg_scale=cfg_scale,
        extra={'concat_conditioning': z},
    )

    # 5. save
    import numpy as np
    from PIL import Image
    img_np = (
        torch.clamp(samples[0] * 0.5 + 0.5, 0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255
    ).astype('uint8')
    # Some BBBC021 condition names contain characters that are not legal in a
    # filename -- `mevinolin/lovastatin` is the one that bites: used raw it
    # turns the output path into a nested directory `mevinolin/` that does not
    # exist, and save() raises FileNotFoundError.
    safe_condition = _safe_filename(condition)
    out = output_dir / f'{safe_condition}-generated-0.png'
    Image.fromarray(img_np, 'RGB').save(out)


def _safe_filename(condition: str) -> str:
    """Make a condition name usable as a filename component."""
    safe = condition
    for char in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        safe = safe.replace(char, '-')
    return safe or 'condition'


def _build_initial_state(
    *,
    use_initial: int,
    control_image_path: Path | None,
    image_size: int,
    device: str,
    noise_level: float,
):
    """Build the ODE start point `x_0`, mirroring upstream's `use_initial` switch.

    Upstream (CellFlux/training/eval_loop.py):

        use_initial == 1 -> x_0 = x_real_ctrl
        use_initial == 2 -> x_0 = x_real_ctrl + randn * noise_level
        otherwise        -> x_0 = randn

    The released checkpoint was trained on (control, treated) pairs, so modes 1
    and 2 are the in-distribution ways to run it; mode 0 samples outside that.
    """
    import torch

    if use_initial == USE_INITIAL_NOISE:
        return torch.randn(1, CELLFLUX_CHANNELS, image_size, image_size, device=device)

    control = _load_control_image(control_image_path, image_size, device)
    if use_initial == USE_INITIAL_CONTROL:
        return control
    # USE_INITIAL_CONTROL_PLUS_NOISE
    return control + torch.randn(control.shape, dtype=torch.float32, device=device) * noise_level


def _load_control_image(path: Path, image_size: int, device: str):
    """Load a control cell image into the model's input range, ``[-1, 1]``.

    Mirrors upstream preprocessing (CellFlux/training/data_utils.py:
    ``X = (X + noise) / 255`` followed by ``Normalize(mean=0.5, std=0.5)``).
    Upstream's uniform dither before the divide is a training-time augmentation
    and is not reproduced here.

    Accepts ``.npy`` in upstream's ``(H, W, 3)`` layout, or a PNG/TIFF.
    """
    import numpy as np
    import torch

    if path is None or not path.is_file():
        raise FileNotFoundError(f'control image not found: {path}')

    if path.suffix.lower() == '.npy':
        arr = np.load(path)
    else:
        from PIL import Image

        with Image.open(path) as img:
            arr = np.asarray(img.convert('RGB'))

    arr = np.asarray(arr, dtype=np.float32)

    # Upstream stores HWC; tolerate CHW too.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] < CELLFLUX_CHANNELS:
        raise ValueError(f'control image must be HxWx3 (or x4), got shape {arr.shape}')
    arr = arr[..., :CELLFLUX_CHANNELS]

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).float()

    if tuple(tensor.shape[-2:]) != (image_size, image_size):
        tensor = torch.nn.functional.interpolate(
            tensor, size=(image_size, image_size), mode='bilinear', align_corners=False
        )

    # Upstream stores pixels as 0-255; tolerate arrays already in [0, 1].
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return (tensor - 0.5) / 0.5  # -> [-1, 1]


def _resolve_embedding_csv(checkpoint: Path, embedding_csv: Path | None) -> Path:
    if embedding_csv is not None and embedding_csv.is_file():
        return embedding_csv
    candidates = (
        checkpoint.parent / _EMBEDDING_CSV,
        checkpoint.parent.parent / _EMBEDDING_CSV,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'CellFlux fingerprint CSV not found next to checkpoint {checkpoint}. Looked in {[str(p) for p in candidates]}'
    )


def _lookup_fingerprint(embedding_csv: Path, compound: str) -> Any:
    import torch
    with open(embedding_csv, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header (empty index column + 0..1023)
        for row in reader:
            if row and row[0] == compound:
                return torch.tensor([float(v) for v in row[1:]], dtype=torch.float32)
    known = _compound_names(embedding_csv)
    raise ValueError(f'Unknown BBBC021 compound {compound!r}. Known compounds: {", ".join(known)}')


def _compound_names(embedding_csv: Path) -> list[str]:
    with open(embedding_csv, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row[0] for row in reader if row and row[0]]


def _count_predictions(predicted_images_dir: Path) -> int:
    if not predicted_images_dir.is_dir():
        return 0
    return sum(
        1
        for path in predicted_images_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.tif', '.tiff', '.png', '.npz'}
    )


def cellflux_result_to_dict(result: CellFluxResult) -> dict[str, Any]:
    return {
        'condition': result.condition,
        'output_dir': str(result.output_dir),
        'predicted_images_dir': str(result.predicted_images_dir),
        'manifest_path': str(result.manifest_path),
        'checkpoint_path': str(result.checkpoint_path) if result.checkpoint_path is not None else None,
        'prediction_count': result.prediction_count,
        'mode': result.mode,
        'model_loaded': result.model_loaded,
        'device': result.device,
        'num_steps': result.num_steps,
        'image_size': result.image_size,
        'cfg_scale': result.cfg_scale,
        'seed': result.seed,
        'use_initial': result.use_initial,
        'noise_level': result.noise_level,
        'control_image_path': str(result.control_image_path) if result.control_image_path is not None else None,
        'note': result.note,
    }
