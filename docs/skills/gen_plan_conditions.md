# `gen-plan-conditions`

`gen-plan-conditions` turns a list of perturbations into a concrete generation plan before any sampling happens.

It resolves every runtime setting the later stages need, checks whether a usable checkpoint is present, and records the result as `generation_plan.json`. It loads no model runtime, so it is cheap to run and safe to run repeatedly.

## Purpose

Use this skill when you want:

- to line up a batch of compounds before spending CPU or GPU time
- to find out whether a run will sample real images or fall back to mock mode
- a single artifact that records exactly which settings a batch used

## Main Outcome

After this skill finishes, the batch is fully specified: conditions, checkpoint, device, sampling parameters, and expected cost.

## Inputs

This skill reads:

- a project config such as `configs/project_config.example.json`
- the `generative_runtime` block from that config, which supplies the model, checkpoint path, perturbation-embedding CSV, device, step count, guidance scale, images per condition, and seed
- a condition list, either from `--conditions` or from `generative_runtime.conditions`
- an optional output directory

Every generation setting can be overridden on the command line. Conditions are required: with neither `--conditions` nor a config list, the skill raises a clear error instead of guessing.

## Outputs

The main user-facing results are:

- `generation_plan.json` — conditions plus every resolved setting
- a `pipeline_skill_manifest.json` describing the run itself

The plan records `mode` as `real` when the checkpoint directory exists and `mock` otherwise, along with `channels`, `image_size`, and a rough CPU time estimate.

## Interface

```bash
cellpainting-skills run --config configs/project_config.example.json \
  --skill gen-plan-conditions \
  --conditions nocodazole,taxol
```

## Next Skills

- [gen-run-morphodiff](gen_run_morphodiff.md)
