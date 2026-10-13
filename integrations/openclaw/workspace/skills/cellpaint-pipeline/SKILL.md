---
name: cellpaint-pipeline
description: Use the validated Cell Painting pipeline library through the current CellPainting-Skills catalog and supporting CLI surfaces.
---

# Cell Painting Pipeline

Use this skill when the user wants to discover, run, or summarize the standardized Cell Painting workflow library.

## Goal

Translate natural-language requests into the current documented skills rather than custom shell chains or older compatibility aliases.

## Preferred CLI

```bash
cellpainting-skills
```

If the `cellpainting-skills` executable is not available in the current shell, run from `$REPO_ROOT` with:

```bash
PYTHONPATH=$REPO_ROOT/src $PYTHON_BIN -m cellpainting_skills
```

## Preferred order

1. Discovery

```bash
cellpainting-skills list
cellpainting-skills describe --skill <skill-key>
```

2. Skill execution

```bash
cd $REPO_ROOT
cellpainting-skills run \
  --config $REPO_ROOT/configs/project_config.example.json \
  --skill <skill-key> \
  --output-dir <output-dir>
```

Fallback execution form if only the module path is available:

```bash
cd $REPO_ROOT
PYTHONPATH=$REPO_ROOT/src $PYTHON_BIN -m cellpainting_skills run \
  --config $REPO_ROOT/configs/project_config.example.json \
  --skill <skill-key> \
  --output-dir <output-dir>
```

3. Use `cellpainting_claw` only when the user explicitly needs a lower-level or broader interface

```bash
$PYTHON_BIN -m cellpainting_claw <command> ...
```

## Current skill examples

Segmentation:

- `cp-prepare-segmentation-inputs`
- `cp-extract-segmentation-artifacts`
- `cp-generate-segmentation-previews`

Classical profiling:

- `cp-extract-measurements`
- `cp-build-single-cell-table`
- `cyto-aggregate-profiles`
- `cyto-annotate-profiles`
- `cyto-normalize-profiles`
- `cyto-select-profile-features`
- `cyto-summarize-classical-profiles`

Deep features:

- `dp-export-deep-feature-inputs`
- `dp-build-deep-feature-project`
- `dp-run-deep-feature-model`
- `dp-collect-deep-features`
- `dp-summarize-deep-features`

Data access:

- `data-inspect-availability`
- `data-plan-download`
- `data-download`

Generative morphology:

- `gen-plan-conditions`
- `gen-run-morphodiff`
- `gen-collect-generated-images`
- `gen-summarize-batch`
- `gen-predict-morphology`

## Routing guidance

Prefer the fine-grained public skill that directly matches the user's task.

Examples:

- if the user wants the segmentation load-data table, choose `cp-prepare-segmentation-inputs`
- if the user wants masks, labels, outlines, or segmentation tables, choose `cp-extract-segmentation-artifacts`
- if the user wants quick visual checks, choose `cp-generate-segmentation-previews`
- if the user asks which wells, sites, fields, or image rows are included in a segmentation run, choose `cp-prepare-segmentation-inputs` and run it instead of manually probing the repository tree
- if the user wants to generate cell morphology images for one compound, choose `gen-predict-morphology`
- if the user wants to generate images for several compounds, run `gen-plan-conditions` then `gen-run-morphodiff`, followed by `gen-collect-generated-images` and `gen-summarize-batch`

For segmentation-scope questions, do not try to infer the answer by listing ad hoc directories such as guessed `demo/wells` or `demo/image_fields` paths. The supported way to answer that question is to execute `cp-prepare-segmentation-inputs` and report its row, well, and site counts together with the generated load-data table.

Do not default to older high-level compatibility names such as `run-segmentation`, `run-classical-profiling`, or `run-deepprofiler` unless the user explicitly asks for those legacy workflow aliases.

## Generative morphology

Generative morphology prediction is a **layered, composable architecture**, not a single fixed model. Three swappable layers solve three different sub-problems:

1. **velocity-field backbone** (flow matching, CellFlux-style) — how the model transports the control distribution to the perturbed distribution
2. **encoder / VAE** (MorphGen-style multi-channel design) — how images enter and leave latent space with per-organelle fidelity
3. **condition injection** (MorphoDiff-style perturbation embedding) — how the perturbation is encoded and fed to the backbone

`gen-predict-morphology` currently instantiates one combination: Stable Diffusion VAE + DDPM latent diffusion + BBBC021 perturbation embedding. It requires `--condition` (a BBBC021 compound name) and `--checkpoint-path` (the unpacked checkpoint directory). Without a checkpoint it runs in mock mode and writes a placeholder manifest.

For a batch of compounds use the four-stage chain instead:

1. `gen-plan-conditions` — resolve conditions and settings into `generation_plan.json` without loading the model
2. `gen-run-morphodiff` — sample one image directory per condition; conditions already generated with the same settings are skipped, so an interrupted batch resumes
3. `gen-collect-generated-images` — normalize images and write `generation_manifest.json` with channel semantics and generation parameters
4. `gen-summarize-batch` — write `generation_summary.json` and a labelled contact sheet

Batch conditions come from `--conditions nocodazole,taxol` or from `generative_runtime.conditions` in the project config. Sampling settings (device, step count, images per condition, guidance scale, seed) come from `generative_runtime` and can be overridden per run.

Note the downstream boundary: the collected images carry the metadata a downstream step needs, but they are one RGB PNG per compound, not the per-channel CellProfiler-convention files the segmentation backend expects. Feeding generated images into segmentation needs a format-conversion step that does not exist yet.

## Reporting expectations

For each run, report:

- exact skill key used
- exact command used
- config path
- output directory
- the main files or directories written
- the next likely skill when the task naturally leads to another step

## Working directory rule

Run CellPainting skill commands from `$REPO_ROOT`.

This matters because the demo and example configs resolve backend data paths relative to the repository layout. If the current shell is inside the OpenClaw workspace, prepend `cd $REPO_ROOT` before running the skill command.

## Optional MCP route

If the host is configured to use MCP, start the server with:

```bash
$REPO_ROOT/integrations/openclaw/start_cellpaint_mcp_http.sh
```

Then prefer MCP discovery only to decide which current skill to run. The actual task should still be framed around the same fine-grained public skill catalog.

## Config guidance

Treat the project config as the source of truth for:

- backend roots
- workspace and output roots
- selected `.cppipe` templates
- runtime settings

Do not invent paths or template names when the config can provide them.

## Demo request guidance

If the user explicitly asks about the repository demo or demo Cell Painting data:

- prefer `configs/project_config.demo.json`
- prefer writing demo outputs under `demo/workspace/outputs/agent_demo_segmentation/`
- keep the request natural in user-facing reporting, but keep the execution anchored to the demo config and demo output tree
