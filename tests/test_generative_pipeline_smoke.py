from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cellpaint_pipeline as package_api
from cellpaint_pipeline.cli import main
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.generative.morphodiff import MorphoDiffResult
from cellpaint_pipeline.generative_pipeline import (
    GenerativePipelineResult,
    collect_generated_images,
    plan_generation_conditions,
    run_generation_batch,
    run_generative_pipeline,
    safe_condition_name,
)

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'


def _write_png(path: Path, *, size: int = 8) -> None:
    """Write a tiny real PNG so collection can read back width/height/mode."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (size, size), (10, 20, 30)).save(path)


def _fake_predict(*, condition: str, output_dir: Path, **kwargs) -> MorphoDiffResult:
    """Stand-in for predict_morphodiff that writes one real image per condition."""
    predicted_images_dir = output_dir / 'predicted_images'
    image_path = predicted_images_dir / f'{safe_condition_name(condition)}-generated-0.png'
    _write_png(image_path)
    manifest_path = output_dir / 'morphodiff_prediction_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                'condition': condition,
                'checkpoint_path': str(kwargs.get('checkpoint_path')) if kwargs.get('checkpoint_path') else None,
                # Mirrors the real predict_morphodiff manifest, which records the
                # embedding CSV it used. _RESUME_KEYS compares this key.
                'embedding_csv': str(kwargs['embedding_csv']) if kwargs.get('embedding_csv') else None,
                'device': kwargs.get('device'),
                'num_inference_steps': kwargs.get('num_inference_steps'),
                'num_images': kwargs.get('num_images'),
                'guidance_scale': kwargs.get('guidance_scale'),
                'seed': kwargs.get('seed'),
                'mode': 'real',
                'prediction_count': 1,
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    return MorphoDiffResult(
        condition=condition,
        output_dir=output_dir,
        predicted_images_dir=predicted_images_dir,
        manifest_path=manifest_path,
        checkpoint_path=kwargs.get('checkpoint_path'),
        control_images_dir=None,
        prediction_count=1,
        mode='real',
        model_loaded=True,
        device=str(kwargs.get('device')),
        num_inference_steps=int(kwargs.get('num_inference_steps') or 50),
        seed=kwargs.get('seed'),
        note='fake',
    )


class GenerativePipelineSmokeTests(unittest.TestCase):
    def test_package_api_exports_generative_pipeline(self) -> None:
        self.assertTrue(callable(package_api.run_generative_pipeline))
        self.assertTrue(callable(package_api.generative_pipeline_result_to_dict))

    def test_safe_condition_name_flattens_path_separators(self) -> None:
        # 'mevinolin/lovastatin' previously produced a FileNotFoundError because
        # the slash was interpolated straight into an output path.
        self.assertEqual(safe_condition_name('mevinolin/lovastatin'), 'mevinolin-lovastatin')
        self.assertEqual(safe_condition_name('nocodazole'), 'nocodazole')
        self.assertEqual(safe_condition_name('  weird name!  '), 'weird-name')

    def test_plan_requires_conditions(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                plan_generation_conditions(config, output_dir=Path(tmpdir) / 'plan')
        self.assertIn('No conditions given', str(ctx.exception))

    def test_run_generative_pipeline_dispatches_and_resumes(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'generative'

            with patch(
                'cellpaint_pipeline.generative_pipeline.predict_morphodiff',
                side_effect=_fake_predict,
            ) as predict_mock:
                result = run_generative_pipeline(
                    config,
                    conditions='nocodazole,mevinolin/lovastatin',
                    output_dir=output_dir,
                    num_inference_steps=4,
                    num_images=1,
                    # Pin the device so this test never probes for CUDA (which
                    # would import torch and perturb test_mock_mode_imports_no_torch).
                    device='cpu',
                )

                self.assertTrue(result.ok)
                self.assertEqual(result.condition_count, 2)
                self.assertEqual(result.image_count, 2)
                self.assertTrue(result.manifest_path.exists())
                self.assertEqual(predict_mock.call_count, 2)
                # num_inference_steps reached the model through the plan.
                self.assertEqual(predict_mock.call_args.kwargs['num_inference_steps'], 4)

                payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
                self.assertEqual(payload['condition_count'], 2)
                self.assertEqual(payload['image_count'], 2)
                self.assertEqual(payload['mode'], 'real')

                # Resume: same plan and settings must not re-sample anything.
                rerun = run_generation_batch(
                    config, plan_path=result.plan_path, output_dir=output_dir
                )
                self.assertEqual(rerun.image_count, 0)
                self.assertEqual(len(rerun.skipped_conditions), 2)
                self.assertEqual(predict_mock.call_count, 2)

    def test_collected_manifest_is_downstream_consumable(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'generative'
            with patch(
                'cellpaint_pipeline.generative_pipeline.predict_morphodiff',
                side_effect=_fake_predict,
            ):
                run_generation_batch(
                    config,
                    plan_path=plan_generation_conditions(
                        config, conditions='nocodazole', output_dir=output_dir, device='cpu'
                    ).plan_path,
                    output_dir=output_dir,
                )
            collection = collect_generated_images(config, output_dir=output_dir)

            manifest = json.loads(collection.manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['channels'], ['DNA', 'F-actin', 'Tubulin'])
            self.assertEqual(manifest['channel_axis'], 'rgb')
            self.assertEqual(manifest['image_size'], 8)
            self.assertEqual(manifest['condition_count'], 1)
            condition = manifest['conditions'][0]
            self.assertEqual(condition['condition'], 'nocodazole')
            self.assertEqual(condition['num_inference_steps'], 50)
            self.assertTrue(Path(condition['images'][0]['path']).is_file())

    def test_mock_mode_imports_no_torch(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'generative'
            result = run_generative_pipeline(
                config,
                conditions='nocodazole',
                output_dir=output_dir,
                checkpoint_path=Path(tmpdir) / 'no_such_checkpoint',
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.mode, 'mock')
            self.assertEqual(result.image_count, 0)
            # The two-path contract: a missing checkpoint must not pull in torch.
            self.assertNotIn('torch', sys.modules)

    def test_auto_device_is_resolved_before_reaching_the_model(self) -> None:
        # The example config ships device='auto', which torch rejects outright
        # ("Expected one of cpu, cuda, ... device type"). The pipeline must
        # resolve it before the model sees it, and must do so without importing
        # torch while running the mock path.
        config = ProjectConfig.from_json(CONFIG_PATH)
        self.assertEqual(config.generative_runtime.get('device'), 'auto')
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'generative'
            plan = plan_generation_conditions(
                config,
                conditions='nocodazole',
                output_dir=output_dir,
                # Force mock mode so this stays fast and torch-free.
                checkpoint_path=Path(tmpdir) / 'no_such_checkpoint',
            )
            # The plan records intent; resolution happens when the model is called.
            self.assertEqual(plan.plan['device'], 'auto')

            with patch(
                'cellpaint_pipeline.generative_pipeline.predict_morphodiff',
                side_effect=_fake_predict,
            ) as predict_mock:
                run_generation_batch(config, plan_path=plan.plan_path, output_dir=output_dir)

            device = predict_mock.call_args.kwargs['device']
            self.assertNotEqual(device, 'auto')
            self.assertIn(device, {'cpu', 'cuda'})

    def test_explicit_embedding_csv_wins_and_a_missing_one_is_an_error(self) -> None:
        # The config key must actually reach inference rather than being
        # decorative, and a mistyped path must not silently fall back to
        # whatever CSV happens to sit next to the checkpoint.
        from cellpaint_pipeline.generative.morphodiff import _resolve_embedding_csv

        checkpoint = PROJECT_ROOT.parent / 'checkpoints' / 'bbbc021_morphodiff_ckpt' / 'checkpoint'
        if not checkpoint.is_dir():
            self.skipTest('MorphoDiff checkpoint not present in this checkout')
        csv_path = checkpoint.parent.parent / 'perturbation_embedding_bbbc021.csv'
        self.assertTrue(csv_path.is_file())

        # No explicit path: auto-discovery, unchanged.
        self.assertEqual(_resolve_embedding_csv(checkpoint), csv_path.resolve())
        # Explicit path: used as given.
        self.assertEqual(_resolve_embedding_csv(checkpoint, explicit=csv_path), csv_path.resolve())
        # Explicit but missing: loud failure, not a silent fallback.
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                _resolve_embedding_csv(checkpoint, explicit=Path(tmpdir) / 'nope.csv')

    def test_configured_embedding_csv_is_forwarded_to_the_model(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        configured = config.generative_runtime.get('perturbation_embedding_csv')
        self.assertIsNotNone(configured, 'example config should set perturbation_embedding_csv')

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'generative'
            with patch(
                'cellpaint_pipeline.generative_pipeline.predict_morphodiff',
                side_effect=_fake_predict,
            ) as predict_mock:
                run_gen = run_generation_batch(
                    config,
                    plan_path=plan_generation_conditions(
                        config, conditions='nocodazole', output_dir=output_dir, device='cpu'
                    ).plan_path,
                    output_dir=output_dir,
                )
            self.assertEqual(predict_mock.call_count, 1)
            self.assertEqual(str(predict_mock.call_args.kwargs['embedding_csv']), configured)
            self.assertEqual(run_gen.image_count, 1)

    @patch('cellpaint_pipeline.cli.run_generative_pipeline')
    def test_cli_run_generative_pipeline_command(self, run_pipeline_mock) -> None:
        output_dir = PROJECT_ROOT / 'outputs' / 'tests_cli_generative_pipeline'
        run_pipeline_mock.return_value = GenerativePipelineResult(
            output_dir=output_dir,
            manifest_path=output_dir / 'generative_pipeline_manifest.json',
            plan_path=output_dir / 'generation_plan.json',
            runs_root=output_dir / 'runs',
            run_manifest_path=output_dir / 'generation_run_manifest.json',
            collected_root=output_dir / 'collected',
            generation_manifest_path=output_dir / 'collected' / 'generation_manifest.json',
            summary_path=output_dir / 'generation_summary.json',
            contact_sheet_path=output_dir / 'generation_contact_sheet.png',
            log_path=output_dir / 'logs' / 'generative_batch.log',
            mode='real',
            condition_count=2,
            image_count=2,
            ok=True,
        )
        returncode = main([
            'run-generative-pipeline',
            '--config',
            str(CONFIG_PATH),
            '--conditions',
            'nocodazole,taxol',
            '--num-inference-steps',
            '4',
            '--output-dir',
            str(output_dir),
        ])
        self.assertEqual(returncode, 0)
        run_pipeline_mock.assert_called_once()
        _, kwargs = run_pipeline_mock.call_args
        self.assertEqual(kwargs['conditions'], 'nocodazole,taxol')
        self.assertEqual(kwargs['num_inference_steps'], 4)
        self.assertEqual(kwargs['output_dir'], output_dir.resolve())


if __name__ == '__main__':
    unittest.main()
