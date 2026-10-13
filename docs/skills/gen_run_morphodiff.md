# `gen-run-morphodiff`

`gen-run-morphodiff` samples generated cell images for every condition in a generation plan.

It is the step that actually loads the MorphoDiff checkpoint and runs reverse diffusion, writing one directory per condition under `runs/`.

## Purpose

Use this skill when you want:

- the generated cell images for a batch of compounds
- a batch that can be resumed instead of restarted
- the raw per-condition outputs before they are normalized for downstream use

## Main Outcome

After this skill finishes, each condition in the plan has a predicted-image directory plus a per-condition manifest recording the settings that produced it.

## Inputs

This skill reads:

- a project config
- a generation plan from [gen-plan-conditions](gen_plan_conditions.md), or a condition list it can plan from directly
- a MorphoDiff checkpoint, from the plan, from `--checkpoint-path`, or from `generative_runtime`

## Outputs

The main user-facing results are:

- `runs/<condition>/predicted_images/*.png` — generated images
- `runs/<condition>/morphodiff_prediction_manifest.json` — per-condition settings and counts
- `generation_run_manifest.json` — the batch-level record
- a batch log under `<output_root>/logs/generative/`

Condition names are sanitized into single path segments, so a name such as `mevinolin/lovastatin` becomes `mevinolin-lovastatin` instead of producing a path error.

## Resuming a batch

Sampling 512x512 images on CPU is slow, so a condition whose existing manifest already records the same checkpoint, device, step count, guidance scale, image count, and seed is **skipped**. Re-running the skill after an interruption therefore continues where it stopped rather than starting over. Pass `--overwrite` to force regeneration.

## Mock mode

Without a usable checkpoint the stage writes placeholder manifests and generates no images. No torch or diffusers import is triggered, so the stage is safe to run in environments that lack the generative runtime.

## Next Skills

- [gen-collect-generated-images](gen_collect_generated_images.md)
