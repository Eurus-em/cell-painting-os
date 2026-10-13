# `gen-collect-generated-images`

`gen-collect-generated-images` normalizes a batch of generated images into a stable layout and records what they are.

Generated images are only useful to another step if that step can tell what the channels mean and how the images were produced. This skill writes that down.

## Purpose

Use this skill when you want:

- generated images in a predictable directory layout instead of scattered run folders
- a single manifest that downstream segmentation or profiling steps can read
- the generation parameters attached to the images rather than remembered separately

## Main Outcome

After this skill finishes, every generated image sits under `collected/<condition>/` and a single manifest describes the whole set.

## Inputs

This skill reads:

- a project config
- a runs directory produced by [gen-run-morphodiff](gen_run_morphodiff.md)
- an optional output directory

## Outputs

The main user-facing results are:

- `collected/<condition>/<image>.png` — normalized copies of the generated images
- `generation_manifest.json` — the downstream-facing description

`generation_manifest.json` records, per condition and per image:

- the compound name and its path-safe form
- the channel semantics: `["DNA", "F-actin", "Tubulin"]` with `channel_axis: "rgb"`
- image size, per-image width/height/mode/bit depth
- `device`, `num_inference_steps`, `guidance_scale`, `num_images`, `seed`
- the checkpoint path and the source manifest

## Downstream boundary

The collected set carries the metadata a downstream step needs, but it is **not** a drop-in input for the segmentation branch.

`cp-prepare-segmentation-inputs` builds its load-data table from the configured segmentation backend's own source CSV, filtered by `Metadata_Plate` / `Metadata_Well` / `Metadata_Site`. It expects per-channel files named in CellProfiler convention plus illumination-correction arrays. Generated images are one RGB PNG per compound, with no plate/well/site identity and no illumination files, so bridging the two requires an explicit format-conversion step that has not been built.

Treat this skill's output as "documented and normalized", not "ready for the segmentation backend".

## Next Skills

- [gen-summarize-batch](gen_summarize_batch.md)
