"""Batch generation pipeline for the generative branch.

This module is the generative-branch counterpart of
:mod:`cellpaint_pipeline.deepprofiler_pipeline`: it orchestrates four stages
that turn a list of perturbation names into a collected, downstream-consumable
image set.

======================  =========================================================
Stage                   Function
======================  =========================================================
1. Plan conditions      :func:`plan_generation_conditions`
2. Run the model        :func:`run_generation_batch`
3. Collect images       :func:`collect_generated_images`
4. Summarize the batch  :func:`summarize_generation_batch`
======================  =========================================================

:func:`run_generative_pipeline` chains all four.

The model itself lives in :mod:`cellpaint_pipeline.generative`; this module owns
only orchestration, so no heavy runtime is imported here. Stage 2 calls
:func:`~cellpaint_pipeline.generative.morphodiff.predict_morphodiff`, which keeps
its two-path contract: without a checkpoint it records placeholder manifests and
never imports torch.

Layout under the run root (``config.generative_output_root`` by default)::

    generation_plan.json
    runs/<condition>/predicted_images/<condition>-generated-<i>.png
    runs/<condition>/morphodiff_prediction_manifest.json
    collected/<condition>/<condition>-generated-<i>.png
    collected/generation_manifest.json
    generation_summary.json
    generation_contact_sheet.png
    generative_pipeline_manifest.json
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.generative import available_generative_models
from cellpaint_pipeline.generative import get_generative_model_definition
from cellpaint_pipeline.generative.morphodiff import MORPHODIFF_CHANNELS
from cellpaint_pipeline.generative.morphodiff import MorphoDiffResult
from cellpaint_pipeline.generative.morphodiff import morphodiff_result_to_dict
from cellpaint_pipeline.generative.morphodiff import predict_morphodiff

GENERATIVE_PIPELINE_MANIFEST = 'generative_pipeline_manifest.json'
GENERATION_PLAN_FILENAME = 'generation_plan.json'
GENERATION_MANIFEST_FILENAME = 'generation_manifest.json'
GENERATION_SUMMARY_FILENAME = 'generation_summary.json'
CONTACT_SHEET_FILENAME = 'generation_contact_sheet.png'
RUN_MANIFEST_FILENAME = 'generation_run_manifest.json'

# The only model with a wired inference path in this pipeline. CellFlux and IMPA
# are registered in cellpaint_pipeline.generative but are out of scope here.
SUPPORTED_MODEL = 'morphodiff'

# Rough CPU cost per reverse-diffusion step for one 512x512 image, used only for
# the estimate recorded in the plan. Real GPU runs are far faster.
_SECONDS_PER_STEP_CPU = 3.5

# Manifest keys that must match for a condition to count as already generated.
# 'embedding_csv' is included so that pointing the config at a different
# embedding table regenerates the batch instead of reusing images that were
# conditioned on the old one.
_RESUME_KEYS = (
    'checkpoint_path',
    'embedding_csv',
    'device',
    'num_inference_steps',
    'guidance_scale',
    'num_images',
    'seed',
)


@dataclass(frozen=True)
class GenerationPlanResult:
    plan_path: Path
    condition_count: int
    plan: dict[str, Any]


@dataclass(frozen=True)
class GenerationRunResult:
    runs_root: Path
    manifest_path: Path
    log_path: Path | None
    per_condition: tuple[MorphoDiffResult, ...]
    skipped_conditions: tuple[str, ...]
    image_count: int
    mode: str


@dataclass(frozen=True)
class GenerationCollectionResult:
    collected_root: Path
    manifest_path: Path
    image_count: int
    condition_count: int


@dataclass(frozen=True)
class GenerationSummaryResult:
    summary_path: Path
    contact_sheet_path: Path | None
    condition_count: int
    image_count: int


@dataclass(frozen=True)
class GenerativePipelineResult:
    output_dir: Path
    manifest_path: Path
    plan_path: Path
    runs_root: Path
    run_manifest_path: Path
    collected_root: Path
    generation_manifest_path: Path
    summary_path: Path
    contact_sheet_path: Path | None
    log_path: Path | None
    mode: str
    condition_count: int
    image_count: int
    ok: bool


def plan_generation_conditions(
    config: ProjectConfig,
    *,
    conditions: str | Sequence[str] | None = None,
    output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    num_images: int | None = None,
    seed: int | None = None,
) -> GenerationPlanResult:
    """Stage 1: resolve the condition list and runtime settings into a plan.

    Writes ``generation_plan.json`` and loads no model runtime. Conditions come
    from the ``conditions`` argument when given, otherwise from
    ``generative_runtime.conditions`` in the project config. Every other
    argument overrides the matching ``generative_runtime`` key.
    """
    run_root = _resolve_run_root(config, output_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    runtime = _resolve_runtime(
        config,
        overrides={
            'checkpoint_path': str(checkpoint_path.expanduser().resolve()) if checkpoint_path is not None else None,
            'device': device,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'num_images': num_images,
            'seed': seed,
        },
    )
    resolved_conditions = _resolve_conditions(config, conditions)
    model = runtime['model']
    _require_supported_model(model)

    checkpoint = _optional_path(runtime.get('checkpoint_path'))
    checkpoint_present = checkpoint is not None and checkpoint.is_dir()
    mode = 'real' if checkpoint_present else 'mock'

    embedding_csv = _optional_path(runtime.get('perturbation_embedding_csv'))
    model_definition = get_generative_model_definition(model)

    per_image_seconds = _SECONDS_PER_STEP_CPU * runtime['num_inference_steps']
    estimated_seconds = round(
        per_image_seconds * runtime['num_images'] * len(resolved_conditions), 1
    )

    plan: dict[str, Any] = {
        'implementation': 'cellpaint_pipeline.generative_pipeline',
        'model': model,
        'mode': mode,
        'conditions': list(resolved_conditions),
        'condition_count': len(resolved_conditions),
        'checkpoint_path': str(checkpoint) if checkpoint is not None else None,
        'checkpoint_present': checkpoint_present,
        'perturbation_embedding_csv': str(embedding_csv) if embedding_csv is not None else None,
        'device': runtime['device'],
        'num_inference_steps': runtime['num_inference_steps'],
        'guidance_scale': runtime['guidance_scale'],
        'num_images': runtime['num_images'],
        'seed': runtime['seed'],
        'channels': list(MORPHODIFF_CHANNELS),
        'image_size': model_definition.get('image_size'),
        'estimated_seconds_cpu': estimated_seconds,
        'note': _plan_note(mode, checkpoint, embedding_csv),
        'generated_utc': _utc_now(),
    }

    plan_path = run_root / GENERATION_PLAN_FILENAME
    _write_json(plan_path, plan)
    return GenerationPlanResult(
        plan_path=plan_path,
        condition_count=len(resolved_conditions),
        plan=plan,
    )


def run_generation_batch(
    config: ProjectConfig,
    *,
    plan_path: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> GenerationRunResult:
    """Stage 2: sample images for every condition in the plan.

    Each condition is written to ``runs/<condition>/`` through
    :func:`~cellpaint_pipeline.generative.morphodiff.predict_morphodiff`. A
    condition whose existing manifest already records the same generation
    parameters is skipped, so a long CPU batch can be resumed instead of
    restarted.
    """
    run_root = _resolve_run_root(config, output_dir)
    if plan_path is not None:
        resolved_plan_path = plan_path.expanduser().resolve()
        plan = _read_json(resolved_plan_path)
    else:
        plan_result = plan_generation_conditions(config, output_dir=run_root)
        resolved_plan_path = plan_result.plan_path
        plan = plan_result.plan

    conditions = [str(item) for item in plan.get('conditions') or []]
    if not conditions:
        raise ValueError(
            'The generation plan contains no conditions. '
            f'Regenerate it with a non-empty condition list: {resolved_plan_path}'
        )

    checkpoint = _optional_path(plan.get('checkpoint_path'))
    checkpoint_present = checkpoint is not None and checkpoint.is_dir()
    embedding_csv = _optional_path(plan.get('perturbation_embedding_csv'))
    device = _resolve_device(str(plan.get('device') or 'cpu'), checkpoint_present=checkpoint_present)
    log_path = _open_batch_log(config, conditions)

    runs_root = run_root / 'runs'
    runs_root.mkdir(parents=True, exist_ok=True)

    per_condition: list[MorphoDiffResult] = []
    skipped: list[str] = []
    image_count = 0

    _append_log(log_path, f'batch start: {len(conditions)} condition(s), mode={plan.get("mode")}, device={device}')
    for index, condition in enumerate(conditions, start=1):
        condition_dir = runs_root / safe_condition_name(condition)
        expected = {
            'checkpoint_path': str(checkpoint) if checkpoint is not None else None,
            'embedding_csv': str(embedding_csv) if embedding_csv is not None else None,
            # Compare the resolved device: a previous run recorded 'cpu', not 'auto'.
            'device': device,
            'num_inference_steps': plan.get('num_inference_steps'),
            'guidance_scale': plan.get('guidance_scale'),
            'num_images': plan.get('num_images'),
            'seed': plan.get('seed'),
        }

        if not overwrite and _condition_already_generated(condition_dir, expected):
            skipped.append(condition)
            _append_log(log_path, f'[{index}/{len(conditions)}] {condition}: skipped (up to date)')
            continue

        started = time.monotonic()
        result = predict_morphodiff(
            condition=condition,
            output_dir=condition_dir,
            checkpoint_path=checkpoint,
            embedding_csv=embedding_csv,
            device=device,
            num_inference_steps=plan.get('num_inference_steps') or 50,
            num_images=plan.get('num_images') or 1,
            guidance_scale=plan.get('guidance_scale') if plan.get('guidance_scale') is not None else 1.0,
            seed=plan.get('seed'),
        )
        elapsed = round(time.monotonic() - started, 1)
        per_condition.append(result)
        image_count += result.prediction_count
        _append_log(
            log_path,
            f'[{index}/{len(conditions)}] {condition}: {result.prediction_count} image(s) '
            f'in {elapsed}s (mode={result.mode})',
        )

    run_manifest_path = run_root / RUN_MANIFEST_FILENAME
    run_manifest = {
        'implementation': 'cellpaint_pipeline.generative_pipeline',
        'plan_path': str(resolved_plan_path),
        'runs_root': str(runs_root),
        'model': plan.get('model'),
        'mode': plan.get('mode'),
        'conditions': conditions,
        'generated_conditions': [result.condition for result in per_condition],
        'skipped_conditions': skipped,
        'results': [morphodiff_result_to_dict(result) for result in per_condition],
        'image_count': image_count,
        'log_path': str(log_path) if log_path is not None else None,
        'generated_utc': _utc_now(),
    }
    _write_json(run_manifest_path, run_manifest)
    _append_log(log_path, f'batch done: {image_count} image(s), {len(skipped)} skipped')

    return GenerationRunResult(
        runs_root=runs_root,
        manifest_path=run_manifest_path,
        log_path=log_path,
        per_condition=tuple(per_condition),
        skipped_conditions=tuple(skipped),
        image_count=image_count,
        mode=str(plan.get('mode') or 'mock'),
    )


def collect_generated_images(
    config: ProjectConfig,
    *,
    runs_root: Path | None = None,
    output_dir: Path | None = None,
) -> GenerationCollectionResult:
    """Stage 3: normalize generated images into a downstream-consumable set.

    Copies each condition's images into ``collected/<condition>/`` under stable
    names and writes ``generation_manifest.json``. That manifest records the
    channel semantics, image size, and generation parameters a downstream
    segmentation or profiling step needs to interpret the images.
    """
    run_root = _resolve_run_root(config, output_dir)
    resolved_runs_root = (
        runs_root.expanduser().resolve() if runs_root is not None else run_root / 'runs'
    )
    if not resolved_runs_root.is_dir():
        raise FileNotFoundError(
            f'No generated runs found at {resolved_runs_root}. Run stage 2 first.'
        )

    collected_root = run_root / 'collected'
    collected_root.mkdir(parents=True, exist_ok=True)

    image_size: int | None = None
    conditions: list[dict[str, Any]] = []
    total_images = 0

    for condition_dir in sorted(path for path in resolved_runs_root.iterdir() if path.is_dir()):
        source_manifest = condition_dir / 'morphodiff_prediction_manifest.json'
        source_payload = _read_json(source_manifest) if source_manifest.is_file() else {}
        condition = str(source_payload.get('condition') or condition_dir.name)
        safe_name = safe_condition_name(condition)

        image_paths = sorted(
            path
            for path in (condition_dir / 'predicted_images').glob('*')
            if path.is_file() and path.suffix.lower() in {'.png', '.tif', '.tiff'}
        )
        if not image_paths:
            continue

        target_dir = collected_root / safe_name
        target_dir.mkdir(parents=True, exist_ok=True)

        images: list[dict[str, Any]] = []
        for source_path in image_paths:
            target_path = target_dir / source_path.name
            shutil.copy2(source_path, target_path)
            metadata = _image_metadata(target_path)
            if image_size is None and metadata.get('width'):
                image_size = metadata['width']
            images.append({'file': target_path.name, 'path': str(target_path), **metadata})

        total_images += len(images)
        conditions.append(
            {
                'condition': condition,
                'safe_name': safe_name,
                'image_count': len(images),
                'images': images,
                'device': source_payload.get('device'),
                'num_inference_steps': source_payload.get('num_inference_steps'),
                'guidance_scale': source_payload.get('guidance_scale'),
                'num_images': source_payload.get('num_images'),
                'seed': source_payload.get('seed'),
                'mode': source_payload.get('mode'),
                'checkpoint_path': source_payload.get('checkpoint_path'),
                'embedding_csv': source_payload.get('embedding_csv'),
                'source_manifest': str(source_manifest) if source_manifest.is_file() else None,
            }
        )

    manifest_path = collected_root / GENERATION_MANIFEST_FILENAME
    manifest = {
        'implementation': 'cellpaint_pipeline.generative_pipeline',
        'model': 'morphodiff',
        'collected_root': str(collected_root),
        'runs_root': str(resolved_runs_root),
        'condition_count': len(conditions),
        'image_count': total_images,
        'channels': list(MORPHODIFF_CHANNELS),
        'channel_axis': 'rgb',
        'image_size': image_size,
        'conditions': conditions,
        'note': (
            'Generated images are synthetic. Each PNG is RGB with channel 0=DNA, '
            '1=F-actin, 2=Tubulin; the image size and per-condition generation '
            'parameters are recorded above so downstream steps can consume this '
            'directory directly.'
        ),
        'generated_utc': _utc_now(),
    }
    _write_json(manifest_path, manifest)

    return GenerationCollectionResult(
        collected_root=collected_root,
        manifest_path=manifest_path,
        image_count=total_images,
        condition_count=len(conditions),
    )


def summarize_generation_batch(
    config: ProjectConfig,
    *,
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
    tile: int = 256,
    columns: int = 4,
) -> GenerationSummaryResult:
    """Stage 4: write a readable summary and a labelled contact sheet."""
    run_root = _resolve_run_root(config, output_dir)
    resolved_manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else run_root / 'collected' / GENERATION_MANIFEST_FILENAME
    )
    if not resolved_manifest_path.is_file():
        raise FileNotFoundError(
            f'No generation manifest at {resolved_manifest_path}. Run stage 3 first.'
        )

    manifest = _read_json(resolved_manifest_path)
    conditions = list(manifest.get('conditions') or [])
    total_images = sum(int(item.get('image_count') or 0) for item in conditions)

    summary = {
        'implementation': 'cellpaint_pipeline.generative_pipeline',
        'model': manifest.get('model'),
        'manifest_path': str(resolved_manifest_path),
        'collected_root': manifest.get('collected_root'),
        'condition_count': len(conditions),
        'image_count': total_images,
        'channels': manifest.get('channels'),
        'image_size': manifest.get('image_size'),
        'conditions': [
            {
                'condition': item.get('condition'),
                'safe_name': item.get('safe_name'),
                'image_count': item.get('image_count'),
                'seed': item.get('seed'),
                'guidance_scale': item.get('guidance_scale'),
                'num_inference_steps': item.get('num_inference_steps'),
                'mode': item.get('mode'),
            }
            for item in conditions
        ],
        'generated_utc': _utc_now(),
    }

    summary_path = run_root / GENERATION_SUMMARY_FILENAME
    _write_json(summary_path, summary)

    tiles: list[tuple[str, Path]] = []
    for item in conditions:
        for image in item.get('images') or []:
            path = image.get('path')
            if path:
                tiles.append((str(item.get('condition')), Path(path)))
    contact_sheet_path = build_contact_sheet(
        tiles, run_root / CONTACT_SHEET_FILENAME, columns=columns, tile=tile
    )

    return GenerationSummaryResult(
        summary_path=summary_path,
        contact_sheet_path=contact_sheet_path,
        condition_count=len(conditions),
        image_count=total_images,
    )


def run_generative_pipeline(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    conditions: str | Sequence[str] | None = None,
    checkpoint_path: Path | None = None,
    device: str | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    num_images: int | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> GenerativePipelineResult:
    """Run all four generative-branch stages and write the pipeline manifest."""
    run_root = _resolve_run_root(config, output_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    plan_result = plan_generation_conditions(
        config,
        conditions=conditions,
        output_dir=run_root,
        checkpoint_path=checkpoint_path,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=num_images,
        seed=seed,
    )
    run_result = run_generation_batch(
        config, plan_path=plan_result.plan_path, output_dir=run_root, overwrite=overwrite
    )
    collection_result = collect_generated_images(config, runs_root=run_result.runs_root, output_dir=run_root)
    summary_result = summarize_generation_batch(
        config, manifest_path=collection_result.manifest_path, output_dir=run_root
    )

    manifest_path = run_root / GENERATIVE_PIPELINE_MANIFEST
    payload = {
        'implementation': 'cellpaint_pipeline.generative_pipeline',
        'package_version': _package_version(),
        'output_dir': str(run_root),
        'plan_path': str(plan_result.plan_path),
        'runs_root': str(run_result.runs_root),
        'run_manifest_path': str(run_result.manifest_path),
        'collected_root': str(collection_result.collected_root),
        'generation_manifest_path': str(collection_result.manifest_path),
        'summary_path': str(summary_result.summary_path),
        'contact_sheet_path': (
            str(summary_result.contact_sheet_path) if summary_result.contact_sheet_path else None
        ),
        'log_path': str(run_result.log_path) if run_result.log_path else None,
        'model': plan_result.plan.get('model'),
        'mode': run_result.mode,
        'condition_count': plan_result.condition_count,
        'image_count': collection_result.image_count,
        'ok': True,
    }
    _write_json(manifest_path, payload)

    return GenerativePipelineResult(
        output_dir=run_root,
        manifest_path=manifest_path,
        plan_path=plan_result.plan_path,
        runs_root=run_result.runs_root,
        run_manifest_path=run_result.manifest_path,
        collected_root=collection_result.collected_root,
        generation_manifest_path=collection_result.manifest_path,
        summary_path=summary_result.summary_path,
        contact_sheet_path=summary_result.contact_sheet_path,
        log_path=run_result.log_path,
        mode=run_result.mode,
        condition_count=plan_result.condition_count,
        image_count=collection_result.image_count,
        ok=True,
    )


def generative_pipeline_result_to_dict(result: GenerativePipelineResult) -> dict[str, object]:
    return {
        'implementation': 'cellpaint_pipeline.generative_pipeline',
        'output_dir': str(result.output_dir),
        'manifest_path': str(result.manifest_path),
        'plan_path': str(result.plan_path),
        'runs_root': str(result.runs_root),
        'run_manifest_path': str(result.run_manifest_path),
        'collected_root': str(result.collected_root),
        'generation_manifest_path': str(result.generation_manifest_path),
        'summary_path': str(result.summary_path),
        'contact_sheet_path': str(result.contact_sheet_path) if result.contact_sheet_path else None,
        'log_path': str(result.log_path) if result.log_path else None,
        'mode': result.mode,
        'condition_count': result.condition_count,
        'image_count': result.image_count,
        'ok': result.ok,
    }


def safe_condition_name(condition: str) -> str:
    """Turn a condition name into a single safe path segment.

    Some BBBC021 conditions contain characters that are legal in a name but not
    in a path segment -- ``mevinolin/lovastatin`` previously produced a
    ``FileNotFoundError`` because the slash was interpolated straight into an
    output path.
    """
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '-', condition).strip('-')
    return cleaned or 'condition'


def build_contact_sheet(
    tiles: Sequence[tuple[str, Path]],
    output: Path,
    *,
    columns: int = 4,
    tile: int = 256,
) -> Path | None:
    """Lay generated images out in a labelled grid.

    Returns ``None`` when Pillow is unavailable or there is nothing to draw,
    matching the behaviour of the standalone export script this was ported from.
    """
    if not tiles:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    # Keep the grid tight when there are fewer images than the requested column
    # count, so a single-image batch is not padded out to a mostly empty canvas.
    columns = max(1, min(int(columns), len(tiles)))
    tile = max(16, int(tile))
    pad, label_height = 8, 34
    rows = (len(tiles) + columns - 1) // columns

    sheet = Image.new(
        'RGB',
        (columns * tile + (columns + 1) * pad, rows * (tile + label_height) + (rows + 1) * pad),
        (18, 18, 20),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)

    drawn = 0
    for index, (label, path) in enumerate(tiles):
        try:
            with Image.open(path) as image:
                thumbnail = image.convert('RGB').resize((tile, tile), Image.LANCZOS)
        except Exception:
            continue
        row, column = divmod(index, columns)
        x = pad + column * (tile + pad)
        y = pad + row * (tile + label_height + pad)
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 4, y + tile + 6), label, fill=(235, 235, 235), font=font)
        drawn += 1

    if drawn == 0:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def _resolve_run_root(config: ProjectConfig, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return config.generative_output_root


def _resolve_runtime(
    config: ProjectConfig,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge explicit overrides over the config block over the built-in defaults."""
    runtime = dict(config.generative_runtime or {})
    if overrides:
        runtime.update({key: value for key, value in overrides.items() if value is not None})
    runtime['model'] = str(runtime.get('model') or SUPPORTED_MODEL)
    runtime['device'] = str(runtime.get('device') or 'cpu')
    runtime['num_inference_steps'] = int(runtime.get('num_inference_steps') or 50)
    guidance = runtime.get('guidance_scale')
    runtime['guidance_scale'] = 1.0 if guidance is None else float(guidance)
    runtime['num_images'] = int(runtime.get('num_images') or 1)
    seed = runtime.get('seed')
    runtime['seed'] = None if seed is None else int(seed)
    return runtime


def _resolve_conditions(
    config: ProjectConfig, conditions: str | Sequence[str] | None
) -> list[str]:
    if conditions is None:
        raw = (config.generative_runtime or {}).get('conditions') or []
        if isinstance(raw, str):
            raw = raw.split(',')
        resolved = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(conditions, str):
        resolved = [item.strip() for item in conditions.split(',') if item.strip()]
    else:
        resolved = [str(item).strip() for item in conditions if str(item).strip()]

    if not resolved:
        raise ValueError(
            'No conditions given. Pass a comma-separated list via --conditions, '
            'or set generative_runtime.conditions in the project config.'
        )
    return resolved


def _resolve_device(requested: str, *, checkpoint_present: bool) -> str:
    """Turn ``auto`` into a concrete torch device name.

    Torch rejects ``auto`` outright, so it must be resolved before it reaches the
    pipeline. This probes torch only when a real run is about to happen, which
    keeps the mock path free of heavy imports.
    """
    if requested != 'auto':
        return requested
    if not checkpoint_present:
        return 'cpu'
    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _require_supported_model(model: str) -> None:
    if model == SUPPORTED_MODEL:
        return
    available = ', '.join(available_generative_models())
    raise NotImplementedError(
        f'The batch generation pipeline is wired for {SUPPORTED_MODEL!r} only; got {model!r}. '
        f'Registered generative models: {available}. '
        'CellFlux and IMPA reuse this pipeline surface but need their own inference wiring.'
    )


def _plan_note(mode: str, checkpoint: Path | None, embedding_csv: Path | None) -> str:
    if mode == 'mock':
        return (
            'Mock mode: no checkpoint directory was found, so running this plan writes '
            'placeholder manifests and generates no images.'
        )
    if embedding_csv is None or not embedding_csv.is_file():
        return (
            f'Checkpoint found at {checkpoint}, but the BBBC021 perturbation embedding CSV '
            'was not resolved. Inference needs it to condition on a compound name.'
        )
    return f'Real mode: checkpoint at {checkpoint}.'


def _condition_already_generated(condition_dir: Path, expected: dict[str, Any]) -> bool:
    manifest_path = condition_dir / 'morphodiff_prediction_manifest.json'
    if not manifest_path.is_file():
        return False
    payload = _read_json(manifest_path)
    for key in _RESUME_KEYS:
        if payload.get(key) != expected.get(key):
            return False
    return True


def _image_metadata(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {}
    try:
        with Image.open(path) as image:
            metadata: dict[str, Any] = {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
            }
            bits = getattr(image, 'bits', None)
            if bits is not None:
                metadata['bit_depth'] = bits
            return metadata
    except Exception:
        return {}


def _open_batch_log(config: ProjectConfig, conditions: Sequence[str]) -> Path | None:
    """Create the batch log, mirroring the subprocess runner's naming."""
    try:
        log_dir = config.log_root / 'generative'
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log_path = log_dir / f'{timestamp}_generative_batch.log'
        log_path.write_text(
            f'generative batch: {len(conditions)} condition(s)\n', encoding='utf-8'
        )
        return log_path
    except OSError:
        return None


def _append_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(f'{_utc_now()} {message}\n')
    except OSError:
        pass


def _read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    return path


def _optional_path(value: Any) -> Path | None:
    if value in (None, ''):
        return None
    return Path(str(value)).expanduser().resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _package_version() -> str:
    from cellpaint_pipeline import __version__

    return __version__


__all__ = [
    'CONTACT_SHEET_FILENAME',
    'GENERATION_MANIFEST_FILENAME',
    'GENERATION_PLAN_FILENAME',
    'GENERATION_SUMMARY_FILENAME',
    'GENERATIVE_PIPELINE_MANIFEST',
    'GenerationCollectionResult',
    'GenerationPlanResult',
    'GenerationRunResult',
    'GenerationSummaryResult',
    'GenerativePipelineResult',
    'build_contact_sheet',
    'collect_generated_images',
    'generative_pipeline_result_to_dict',
    'plan_generation_conditions',
    'run_generation_batch',
    'run_generative_pipeline',
    'safe_condition_name',
    'summarize_generation_batch',
]
