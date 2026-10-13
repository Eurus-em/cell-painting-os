"""Generative morphology prediction with MorphoDiff.

MorphoDiff is a latent diffusion model (fine-tuned from Stable Diffusion 1.4)
that generates cell morphology images conditioned on a chemical perturbation.
It takes a compound name (for example a BBBC021 compound such as ``nocodazole``
or ``cytochalasin-d``), looks up that compound's pre-computed perturbation
embedding, and samples a treated-cell image from noise.

The pretrained checkpoint referenced by this module was trained on the BBBC021
Cell Painting benchmark. Inference is *conditioned generation*: the input is a
compound name, not a control image (the model does not do image-to-image).

The module is split into two paths:

* **mock path** -- when ``checkpoint_path`` is absent or does not exist, the
  function records a placeholder manifest without loading any heavy runtime
  dependency (torch / diffusers are imported lazily, only in the real path).
* **real path** -- when a valid ``checkpoint_path`` is supplied, the function
  loads the MorphoDiff pipeline and samples ``num_images`` images.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# BBBC021 (the dataset behind the pretrained checkpoint) images MCF-7 cells in
# three fluorescence channels: DNA (Hoechst), F-actin (phalloidin), and
# beta-tubulin. The checkpoint VAE expects 3-channel 512x512 inputs.
MORPHODIFF_CHANNELS: tuple[str, ...] = ('DNA', 'F-actin', 'Tubulin')

# Default reverse-diffusion step count used unless a caller overrides it.
DEFAULT_INFERENCE_STEPS: int = 50

# The perturbation embedding is padded/repeated into the shape Stable Diffusion
# expects for its cross-attention conditioning: sequence length 77, width 768.
_PERTURBATION_SEQ_LEN: int = 77
_PERTURBATION_WIDTH: int = 768

# BBBC021 compound embeddings live in a CSV keyed by compound name. This file
# sits next to the checkpoint directory (the checkpoint zip also ships it under
# morphodiff/required_file/).
_BBBC021_EMBEDDING_CSV = 'perturbation_embedding_bbbc021.csv'


@dataclass(frozen=True)
class MorphoDiffResult:
    """Serializable result contract for one MorphoDiff generation run."""

    condition: str
    output_dir: Path
    predicted_images_dir: Path
    manifest_path: Path
    checkpoint_path: Path | None
    control_images_dir: Path | None
    prediction_count: int
    mode: str  # 'mock' until the real checkpoint is wired in, then 'real'
    model_loaded: bool
    device: str
    num_inference_steps: int
    seed: int | None
    note: str
    # Defaulted so existing constructions of this result stay valid; records
    # which embedding file inference actually used.
    embedding_csv: Path | None = None


def predict_morphodiff(
    *,
    condition: str,
    output_dir: Path,
    control_images_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    embedding_csv: Path | None = None,
    device: str = 'cpu',
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    num_images: int = 1,
    guidance_scale: float = 1.0,
    seed: int | None = None,
) -> MorphoDiffResult:
    """Generate treated-cell morphology conditioned on a compound name.

    This is the public interface for the MorphoDiff generative skill. It is
    intentionally stable: the real inference path fills in behind the same
    signature.

    Parameters
    ----------
    condition:
        Target perturbation name (a BBBC021 compound such as ``nocodazole``,
        ``taxol``, ``cytochalasin-d``). Required. Must exist in the BBBC021
        perturbation-embedding CSV.
    output_dir:
        Run root. Predicted images and the prediction manifest are written here.
    control_images_dir:
        Unused by the current conditioned-generation model. Kept for interface
        compatibility; MorphoDiff samples from noise conditioned on ``condition``
        rather than transforming a control image.
    checkpoint_path:
        Path to the unpacked MorphoDiff checkpoint directory (the one containing
        ``model_index.json``). When absent or missing, the function runs in
        ``mock`` mode without importing torch.
    embedding_csv:
        Optional explicit path to the BBBC021 perturbation-embedding CSV. When
        omitted, the CSV is discovered next to the checkpoint; when given but
        missing, that is an error rather than a silent fallback, so a mistyped
        path does not quietly change which embeddings are used.
    device:
        Torch device for inference (``cpu`` or a CUDA index). CPU is the safe
        default; sampling 512x512 images on CPU is slow.
    num_inference_steps:
        Number of reverse-diffusion sampling steps.
    num_images:
        Number of images to generate for the condition.
    guidance_scale:
        Classifier-free-guidance scale. ``1.0`` disables CFG (the framework's
        historical default); upstream MorphoDiff leaves diffusers' default of
        ``7.5``, which amplifies the perturbation conditioning at the cost of
        diversity. Values above 1.0 need an unconditional branch -- see
        ``_run_diffusion_inference``.
    seed:
        Optional RNG seed for reproducible sampling.
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

    model_loaded = checkpoint is not None and checkpoint.is_dir()

    resolved_embedding_csv: Path | None = None
    if model_loaded:
        resolved_embedding_csv = _run_diffusion_inference(
            checkpoint=checkpoint,
            condition=condition,
            embedding_csv=(
                embedding_csv.expanduser().resolve() if embedding_csv is not None else None
            ),
            output_dir=predicted_images_dir,
            device=device,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    prediction_count = _count_predictions(predicted_images_dir)
    mode = 'real' if model_loaded else 'mock'
    note = (
        'Placeholder result: MorphoDiff inference requires a valid checkpoint_path '
        'plus the torch/diffusers runtime. Provide a real checkpoint to generate images.'
        if not model_loaded
        else f'MorphoDiff generated {prediction_count} image(s) for {condition!r}.'
    )

    manifest_path = output_dir / 'morphodiff_prediction_manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'condition': condition,
                'checkpoint_path': str(checkpoint) if checkpoint is not None else None,
                'embedding_csv': str(resolved_embedding_csv) if resolved_embedding_csv is not None else None,
                'control_images_dir': str(controls) if controls is not None else None,
                'predicted_images_dir': str(predicted_images_dir),
                'device': device,
                'num_inference_steps': num_inference_steps,
                'num_images': num_images,
                'guidance_scale': guidance_scale,
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

    return MorphoDiffResult(
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
        num_inference_steps=num_inference_steps,
        seed=seed,
        note=note,
        embedding_csv=resolved_embedding_csv,
    )


def _run_diffusion_inference(
    *,
    checkpoint: Path,
    condition: str,
    embedding_csv: Path | None,
    output_dir: Path,
    device: str,
    num_inference_steps: int,
    num_images: int,
    guidance_scale: float,
    seed: int | None,
) -> Path:
    """Load the MorphoDiff pipeline and sample conditioned images.

    Mirrors the upstream ``generate_img.py`` inference path:

    1. look up the compound's 512-dim perturbation embedding from the CSV;
    2. pad to 768 and repeat to ``(1, 77, 768)`` to stand in for the SD text
       embedding;
    3. load vae / unet_ema / scheduler and run reverse diffusion conditioned on
       that embedding, then decode to images and save them under ``output_dir``.
    """
    import torch

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers import StableDiffusionPipeline

    class MorphoDiffPipeline(StableDiffusionPipeline):
        """StableDiffusionPipeline whose text encoder is the perturbation encoder."""

        def __init__(
            self,
            vae,
            text_encoder,
            unet,
            scheduler,
            feature_extractor=None,
            tokenizer=None,
            safety_checker=None,
            image_encoder=None,
            requires_safety_checker=False,
        ):
            super().__init__(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
                requires_safety_checker=requires_safety_checker,
            )

        def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        ):
            # The base pipeline unpacks (prompt_embeds, negative_prompt_embeds)
            # and does the CFG concatenation itself, so the unconditional branch
            # has to be returned here rather than as None.
            #
            # The perturbation encoder only knows compound names, so the
            # negative prompt (which is empty by default) falls back to the
            # "no perturbation" code instead of being looked up as a compound.
            def embed(text):
                if text is None or (isinstance(text, str) and not text.strip()):
                    return self.text_encoder.unconditional_embedding()
                return self.text_encoder(text)

            embeddings = embed(prompt).to(device)
            negative_embeddings = embed(negative_prompt).to(device) if do_classifier_free_guidance else None
            if num_images_per_prompt > 1:
                embeddings = embeddings.repeat(num_images_per_prompt, 1, 1)
                if negative_embeddings is not None:
                    negative_embeddings = negative_embeddings.repeat(num_images_per_prompt, 1, 1)
            return embeddings, negative_embeddings

    resolved_embedding_csv = _resolve_embedding_csv(checkpoint, explicit=embedding_csv)
    encoder = _PerturbationEncoder(resolved_embedding_csv)

    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder='unet_ema')
    scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder='scheduler')

    pipeline = MorphoDiffPipeline(
        vae=vae,
        text_encoder=encoder,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.to(device)

    if seed is not None:
        torch.manual_seed(seed)

    for i in range(num_images):
        image = pipeline(
            prompt=condition,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        image.images[0].save(output_dir / f'{condition}-generated-{i}.png')

    return resolved_embedding_csv


def _resolve_embedding_csv(checkpoint: Path, *, explicit: Path | None = None) -> Path:
    """Locate the BBBC021 perturbation-embedding CSV.

    An explicit path wins when supplied. A missing explicit path is an error
    rather than a silent fall-through to discovery: quietly using a different
    embedding file than the caller asked for would change every generated image
    without any visible signal. Without an explicit path, the CSV is discovered
    next to the checkpoint.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(
                f'Configured BBBC021 perturbation embedding CSV does not exist: {explicit}. '
                f'Leave generative_runtime.perturbation_embedding_csv unset to discover it '
                f'next to the checkpoint instead.'
            )
        return explicit

    candidates = (
        checkpoint.parent / _BBBC021_EMBEDDING_CSV,
        checkpoint.parent.parent / _BBBC021_EMBEDDING_CSV,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'BBBC021 perturbation embedding CSV not found next to checkpoint {checkpoint}. '
        f'Looked in: {[str(p) for p in candidates]}. '
        f'Set generative_runtime.perturbation_embedding_csv to point at it explicitly.'
    )


def _load_perturbation_table(embedding_csv: Path) -> dict[str, list[float]]:
    """Read ``compound, latent_1, ..., latent_N`` into a compound -> vector map."""
    table: dict[str, list[float]] = {}
    with open(embedding_csv, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if not row or not row[0]:
                continue
            table[row[0]] = [float(value) for value in row[1:]]
    return table


class _PerturbationEncoder:
    """Maps a compound name to its padded/repeated SD-style conditioning tensor."""

    def __init__(self, embedding_csv: Path):
        self._table = _load_perturbation_table(embedding_csv)

    def __call__(self, compound: str):
        import torch
        from torch.nn import functional as F

        if compound not in self._table:
            known = ', '.join(sorted(self._table)[:20])
            raise ValueError(
                f'Unknown BBBC021 compound {compound!r}. Known compounds include: {known}, ...'
            )
        embedding = torch.tensor(self._table[compound], dtype=torch.float32)  # (173,)
        padding_left = (_PERTURBATION_WIDTH - len(embedding)) // 2
        padding_right = _PERTURBATION_WIDTH - len(embedding) - padding_left
        padded = F.pad(embedding, (padding_left, padding_right), 'constant', 1)  # (768,)
        padded = padded.unsqueeze(0).repeat(1, _PERTURBATION_SEQ_LEN, 1)  # (1, 77, 768)
        return padded

    def unconditional_embedding(self):
        """The 'no perturbation' condition, used as the CFG negative branch.

        Upstream MorphoDiff conditions its unconditional (``naive``) model on an
        all-ones tensor of the same ``(1, 77, 768)`` shape, and its rxrx1 encoder
        codes an ``EMPTY`` perturbation as ``np.ones(512)``. This mirrors that
        convention.

        Caveat: the BBBC021 embedding table has no such row, so this code was
        never trained on directly -- it is the upstream convention reused, not a
        validated "untreated" condition.
        """
        import torch

        return torch.ones(1, _PERTURBATION_SEQ_LEN, _PERTURBATION_WIDTH)


def _count_predictions(predicted_images_dir: Path) -> int:
    if not predicted_images_dir.is_dir():
        return 0
    return sum(
        1
        for path in predicted_images_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.tif', '.tiff', '.png', '.npz'}
    )


def morphodiff_result_to_dict(result: MorphoDiffResult) -> dict[str, Any]:
    """Serialize a ``MorphoDiffResult`` into a JSON-ready dictionary."""
    return {
        'condition': result.condition,
        'output_dir': str(result.output_dir),
        'predicted_images_dir': str(result.predicted_images_dir),
        'manifest_path': str(result.manifest_path),
        'checkpoint_path': str(result.checkpoint_path) if result.checkpoint_path is not None else None,
        'embedding_csv': str(result.embedding_csv) if result.embedding_csv is not None else None,
        'control_images_dir': str(result.control_images_dir) if result.control_images_dir is not None else None,
        'prediction_count': result.prediction_count,
        'mode': result.mode,
        'model_loaded': result.model_loaded,
        'device': result.device,
        'num_inference_steps': result.num_inference_steps,
        'seed': result.seed,
        'note': result.note,
    }
