# `gen-summarize-batch`

`gen-summarize-batch` produces a readable summary and a labelled contact sheet for a collected batch.

## Purpose

Use this skill when you want:

- a quick human-readable view of what a batch produced
- a contact sheet for visual review instead of opening images one by one
- per-condition counts and parameters in one small file

## Main Outcome

After this skill finishes, a reviewer can see the whole batch at a glance and read its parameters without inspecting raw image paths.

## Inputs

This skill reads:

- a project config
- a `generation_manifest.json` from [gen-collect-generated-images](gen_collect_generated_images.md)
- an optional output directory

## Outputs

The main user-facing results are:

- `generation_summary.json` — per-condition counts and generation parameters
- `generation_contact_sheet.png` — generated images laid out in a labelled grid

The contact sheet labels each tile with its compound name and tightens the grid when there are fewer images than the requested column count. If Pillow is unavailable, or there is nothing to draw, the sheet is skipped and the summary is still written.

## Interpreting the output

Low step counts and `guidance_scale=1.0` produce images that are visually similar across conditions. That is a property of the sampling settings, not of this skill — see the framework notes on classifier-free guidance for the conditioning trade-off.

## Next Skills

None. This is the end of the generative batch chain. To feed the images into analysis, see the downstream boundary note in [gen-collect-generated-images](gen_collect_generated_images.md).
