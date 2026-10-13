# `gen-predict-morphology`

`gen-predict-morphology` generates cell morphology images conditioned on a chemical perturbation.

The first generative backend is MorphoDiff, a latent diffusion model fine-tuned from Stable Diffusion 1.4 and trained on the BBBC021 Cell Painting benchmark. It conditions on a compound name, looks up that compound's pre-computed perturbation embedding, and samples treated-cell images from noise.

## Status

This skill is **implemented and tested end-to-end**:

- the registration, runner, CLI, and documentation are in place
- real diffusion inference is wired into `_run_diffusion_inference`
- verified against the `bbbc021_morphodiff_ckpt` checkpoint (model loads, a `nocodazole` image is generated on CPU)

Without a `checkpoint_path` the skill still runs in `mock` mode and records a placeholder manifest (no torch/diffusers import is triggered).

### Relationship to the batch chain

This skill predates the batch skills and remains available as the single-condition shortcut. It is equivalent to [gen-plan-conditions](gen_plan_conditions.md) followed by [gen-run-morphodiff](gen_run_morphodiff.md) for one condition.

Prefer the batch chain when you want more than one compound, a resumable run, normalized outputs, or a contact sheet.

## Purpose

Use this skill when you want:

- generated images of how cells would look under a chemical perturbation
- the generative branch of the Cell Painting pipeline (the first model is MorphoDiff, with CellFlux, MorphGen, and IMPA planned to follow)

Note: this is *conditioned generation*, not image-to-image. The input is a compound name; the model samples from noise, it does not transform a control image.

## Inputs

This skill reads:

- a target perturbation `condition` — a BBBC021 compound name such as `nocodazole`, `taxol`, `cytochalasin-d` (must exist in the perturbation-embedding CSV)
- a `checkpoint_path` pointing at the unpacked MorphoDiff checkpoint directory (the one containing `model_index.json`)
- the BBBC021 perturbation-embedding CSV (`perturbation_embedding_bbbc021.csv`): taken from `generative_runtime.perturbation_embedding_csv` when that key is set, otherwise discovered automatically next to the checkpoint
- an optional output directory

## Outputs

The main user-facing results are:

- `predicted_images/` — generated treated-cell images (PNG)
- `morphodiff_prediction_manifest.json` — a manifest describing the condition, checkpoint, and prediction counts

## Interface

The public function behind this skill is:

```python
from cellpaint_pipeline.generative.morphodiff import predict_morphodiff

result = predict_morphodiff(
    condition="nocodazole",
    output_dir=Path(".../skill_run"),
    checkpoint_path=Path(".../bbbc021_morphodiff_ckpt/checkpoint"),
    device="cpu",
    num_inference_steps=50,
    num_images=1,
    seed=42,
)
```

Inference requires a runtime with `torch` + `diffusers` + `safetensors`. Both the `cellpainting-claw` and `cellpainting-generative` environments in this repository carry those packages. CPU sampling of a 512x512 image takes roughly `num_inference_steps × 3.5s` — a 50-step image is a few minutes on CPU; a GPU makes this seconds.

## How it works

1. look up the compound's 512-dim perturbation embedding from the CSV
2. pad to 768 and repeat to `(1, 77, 768)` to stand in for the Stable Diffusion text embedding
3. run reverse diffusion with the loaded `unet_ema` + `vae`, then decode to an image
