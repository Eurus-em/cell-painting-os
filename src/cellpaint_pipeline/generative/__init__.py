"""Generative morphology prediction skills.

Deep-learning models that predict cell morphology responses under perturbation,
exposed as task-oriented pipeline skills.

Generative morphology is a **layered architecture**, not a single model. Each
layer solves a different sub-problem and can be swapped independently:

1. **velocity-field backbone** -- how the control distribution is transported to
   the perturbed distribution. CellFlux implements this with flow matching.
2. **encoder / VAE** -- how images enter and leave latent space with
   per-organelle fidelity.
3. **condition injection** -- how a perturbation is encoded and fed to the
   backbone. MorphoDiff uses cross-attention on a latent embedding.

Do not treat these models as interchangeable replacements for one another.
When swapping a layer, change only that layer.

Model status
------------

======================  ==========  ============================================
Model                   Status      Notes
======================  ==========  ============================================
``morphodiff``          working     Latent diffusion (SD 1.4). BBBC021 compounds,
                                    512x512, 3-channel. Checkpoint + inference
                                    verified end to end.
``cellflux``            working     Flow matching velocity field. BBBC021
                                    compounds, 128x128, 3-channel. Checkpoint +
                                    inference verified end to end.
``impa``                skeleton    StarGANv2 style transfer. Checkpoints exist
                                    but inference needs the theislab/IMPA
                                    runtime plus control images (not bundled).
======================  ==========  ============================================

Every model module follows the same two-path contract: a **mock path** when no
checkpoint is supplied (runs today, writes a placeholder manifest, imports no
torch), and a **real path** when a valid checkpoint is given.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'CELLFLUX_CHANNELS',
    'CELLFLUX_IMAGE_SIZE',
    'MORPHODIFF_CHANNELS',
    'available_generative_models',
    'get_generative_model_definition',
    'generative_model_definition_to_dict',
]

# Lightweight, torch-free metadata so callers can discover models without
# importing the heavy inference modules.
_GENERATIVE_MODELS: dict[str, dict[str, Any]] = {
    'morphodiff': {
        'key': 'morphodiff',
        'paradigm': 'latent-diffusion',
        'condition_injection': 'cross-attention',
        'status': 'working',
        'dataset': 'bbbc021',
        'image_size': 512,
        'channels': ('DNA', 'F-actin', 'Tubulin'),
        'needs_control_images': False,
        'module': 'cellpaint_pipeline.generative.morphodiff',
        'entrypoint': 'predict_morphodiff',
        'notes': 'Stable Diffusion 1.4 fine-tune. Checkpoint and inference verified end to end.',
    },
    'cellflux': {
        'key': 'cellflux',
        'paradigm': 'flow-matching',
        'condition_injection': 'linear-add',
        'status': 'working',
        'dataset': 'bbbc021',
        'image_size': 128,
        'channels': ('DNA', 'F-actin', 'Tubulin'),
        'needs_control_images': False,
        'module': 'cellpaint_pipeline.generative.cellflux',
        'entrypoint': 'predict_cellflux',
        'notes': 'Flow-matching velocity field. Checkpoint and inference verified end to end.',
    },
    'impa': {
        'key': 'impa',
        'paradigm': 'style-transfer',
        'condition_injection': 'adain-style-code',
        'status': 'skeleton',
        'dataset': 'bbbc021',
        'image_size': 128,
        'channels': ('DNA', 'F-actin', 'Tubulin'),
        'needs_control_images': True,
        'module': 'cellpaint_pipeline.generative.impa',
        'entrypoint': 'predict_impa',
        'notes': 'StarGANv2 content/style decomposition. Needs the theislab/IMPA runtime and control images.',
    },
}


def available_generative_models(*, include_skeleton: bool = True) -> list[str]:
    """Return the registered generative model keys."""
    return [
        key
        for key, payload in _GENERATIVE_MODELS.items()
        if include_skeleton or payload['status'] != 'skeleton'
    ]


def get_generative_model_definition(model_key: str) -> dict[str, Any]:
    """Return the metadata payload for one generative model."""
    if model_key not in _GENERATIVE_MODELS:
        available = ', '.join(available_generative_models())
        raise KeyError(f'Unknown generative model: {model_key}. Available: {available}')
    return dict(_GENERATIVE_MODELS[model_key])


def generative_model_definition_to_dict(model_key: str) -> dict[str, Any]:
    """Return a JSON-ready payload for one generative model."""
    payload = get_generative_model_definition(model_key)
    payload['channels'] = list(payload['channels'])
    return payload


# Re-exported constants for callers that want channel/size metadata without
# importing the full inference modules.
MORPHODIFF_CHANNELS: tuple[str, ...] = ('DNA', 'F-actin', 'Tubulin')
CELLFLUX_CHANNELS: tuple[str, ...] = ('DNA', 'F-actin', 'Tubulin')
CELLFLUX_IMAGE_SIZE: int = 128
