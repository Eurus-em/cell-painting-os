# Cell Painting Operator

You operate the validated `CellPainting-Claw` workspace at:

- `$REPO_ROOT`

Your job is to route user requests to the stable library interfaces instead of reaching into internal workflow modules.

## Preferred execution order

1. Discover available surfaces.
2. Prefer task-oriented skills when they match the user request.
3. Use public API entrypoints when the request is broader or needs explicit contract mapping.
4. Use lower-level CLI commands only when the higher-level surfaces are not sufficient.

## Preferred commands

Prefer task-level skills first:

```bash
cd $REPO_ROOT
cellpainting-skills list
cellpainting-skills describe --skill <skill-key>
cellpainting-skills run --config $REPO_ROOT/configs/project_config.example.json --skill <skill-key> --output-dir <output-dir>
```

If the `cellpainting-skills` executable is not available in the current shell, use:

```bash
cd $REPO_ROOT
PYTHONPATH=$REPO_ROOT/src $PYTHON_BIN -m cellpainting_skills <command> ...
```

Use the `cellpainting-claw` interpreter directly:

```bash
$PYTHON_BIN -m cellpainting_claw list-mcp-tools
$PYTHON_BIN -m cellpainting_claw show-public-api-contract
$PYTHON_BIN -m cellpainting_claw run-pipeline-skill --config $REPO_ROOT/configs/project_config.example.json --skill run-segmentation
```

## Stable top-level surfaces

- `run-pipeline-skill`
- `run-public-api-entrypoint`
- `run-deepprofiler-pipeline`
- `run-generative-pipeline`
- `serve-mcp`

## Reporting rule

After each pipeline execution, summarize:

- what entrypoint ran
- which config file was used
- which output directory was written
- whether pycytominer-style or DeepProfiler outputs were produced

## Generative morphology

When the user asks to generate cell images conditioned on a chemical perturbation:

```bash
cd $REPO_ROOT
cellpainting-skills run \
  --config $REPO_ROOT/configs/project_config.demo.json \
  --skill gen-predict-morphology \
  --condition <bbbc021-compound> \
  --checkpoint-path <checkpoint-dir> \
  --output-dir <output-dir>
```

- `--condition` must be one of the 98 BBBC021 compounds (for example `nocodazole`, `taxol`, `cytochalasin-d`).
- `--checkpoint-path` points at the unpacked checkpoint directory (the one containing `model_index.json`).
- Generation requires torch + diffusers. Without a checkpoint, the skill runs in mock mode and writes a placeholder manifest.

For more than one compound, use the batch chain instead of looping the single-condition skill:

```bash
cd $REPO_ROOT
cellpainting-skills run --config $REPO_ROOT/configs/project_config.demo.json \
  --skill gen-plan-conditions --conditions nocodazole,taxol
cellpainting-skills run --config $REPO_ROOT/configs/project_config.demo.json \
  --skill gen-run-morphodiff --generation-plan-path <plan>.json
cellpainting-skills run --config $REPO_ROOT/configs/project_config.demo.json \
  --skill gen-collect-generated-images --runs-root <runs-dir>
cellpainting-skills run --config $REPO_ROOT/configs/project_config.demo.json \
  --skill gen-summarize-batch --generation-manifest-path <manifest>.json
```

`gen-run-morphodiff` skips conditions that already produced images with the same settings, so a long CPU batch can be resumed. The collected manifest records channel semantics and generation parameters, but the images are one RGB PNG per compound — feeding them into the segmentation backend needs a format conversion that does not exist yet.

Generative morphology is a **layered architecture**, not a single model:

1. velocity-field backbone (flow matching, CellFlux-style)
2. encoder / VAE (MorphGen-style multi-channel design)
3. condition injection (MorphoDiff-style perturbation embedding)

Do not treat CellFlux, MorphGen, or MorphoDiff as interchangeable replacements. When the user asks to swap a layer, change only that layer — not the whole model.
