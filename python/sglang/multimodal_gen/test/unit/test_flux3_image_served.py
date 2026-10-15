# SPDX-License-Identifier: Apache-2.0
"""GPU-free package and registration tests for FLUX.3 image-small."""

import argparse
import json
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


def _config_dict():
    return {
        "format_version": 1,
        "model": "flux-3-image-klein",
        "quantization": None,
        "torch_dtype": "bfloat16",
        "dit": {
            "hidden": 3072,
            "heads": 24,
            "depth": 5,
            "depth_single": 28,
            "mlp_ratio": 3.0,
            "axes_dim": [32, 32, 32, 32],
            "theta": 10000,
            "in_channels": {"image": 128, "image_cond": 128, "text": 20480},
            "out_modes": ["image"],
            "vec_in_dim": None,
            "image_rope_scale": 0.5,
        },
        "text_encoder": {
            "path": "text_encoder",
            "tokenizer": "tokenizer",
            "layers": [4, 8, 12, 16, 20, 24, 28, 32],
            "pad_multiple": 80,
            "max_tokens": 8192,
            "dtype": "bfloat16",
        },
        "vae": {"path": "ae.safetensors", "dtype": "float32"},
        "sampler": {
            "steps": 50,
            "guidance": 3.5,
            "alpha": 6.93,
            "size": [1024, 1024],
        },
    }


class TestFlux3ServedRegistration(unittest.TestCase):
    def test_pipeline_and_config_classes_are_registered(self):
        from sglang.multimodal_gen.registry import (
            _PIPELINE_REGISTRY,
            _discover_and_register_pipelines,
            get_pipeline_config_classes,
        )

        _discover_and_register_pipelines()
        self.assertIn("Flux3ImagePipeline", _PIPELINE_REGISTRY)
        config_cls, params_cls = get_pipeline_config_classes("Flux3ImagePipeline")
        self.assertEqual(config_cls.__name__, "Flux3ImagePipelineConfig")
        self.assertEqual(params_cls.__name__, "Flux3ImageSamplingParams")

    def test_config_defaults_match_package(self):
        from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
        from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
            Flux3ImagePipelineConfig,
        )

        config = Flux3ImagePipelineConfig()
        config.check_pipeline_config()
        self.assertEqual(config.task_type, ModelTaskType.TI2I)
        self.assertEqual(config.served_modes(), frozenset({"t2i", "i2i"}))
        # Sampler knobs are unset until the served package is known.
        self.assertIsNone(config.flux3_cfg_scale)
        self.assertIsNone(config.flux3_steps)
        self.assertTrue(config.flux3_compile_text_encoder)

    def test_sampler_defaults_follow_the_served_package(self):
        """The distilled flow map runs 16 steps without CFG; fixed 50-step CFG
        defaults in the pipeline config used to override the package sampler."""
        import tempfile
        from pathlib import Path

        from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
            Flux3ImagePipelineConfig,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(_config_dict()))
            (root / "manifest.json").write_text(
                json.dumps({"format_version": 1, "model": "flux-3-image-klein"})
            )
            for name in ("model.safetensors", "ae.safetensors"):
                (root / name).touch()
            (root / "text_encoder").mkdir()
            (root / "tokenizer").mkdir()
            sd = root / "variants" / "sd"
            sd.mkdir(parents=True)
            sd_config = _config_dict()
            sd_config["dit"]["flow_map"] = True
            sd_config["sampler"].update(steps=16, guidance=None)
            (sd / "config.json").write_text(json.dumps(sd_config))
            (sd / "manifest.json").write_text(
                json.dumps({"format_version": 1, "model": "flux-3-image-klein"})
            )
            (sd / "model.safetensors").touch()

            def validated(variant, **overrides):
                config = Flux3ImagePipelineConfig(**overrides)
                config.validate_server_args(
                    SimpleNamespace(
                        model_path=str(root),
                        model_variant=variant,
                        enable_cfg_parallel=False,
                    )
                )
                return config

            base = validated(None)
            self.assertEqual(
                (base.flux3_steps, base.flux3_cfg_scale),
                (_config_dict()["sampler"]["steps"], 3.5),
            )
            distilled = validated("sd")
            self.assertEqual(
                (distilled.flux3_steps, distilled.flux3_cfg_scale),
                (16, None),
            )
            self.assertEqual(validated("sd", flux3_steps=4).flux3_steps, 4)
            with self.assertRaisesRegex(ValueError, "without CFG"):
                validated("sd", flux3_cfg_scale=3.5)
            config = Flux3ImagePipelineConfig()
            with self.assertRaisesRegex(ValueError, "cfg-parallel"):
                config.validate_server_args(
                    SimpleNamespace(
                        model_path=str(root),
                        model_variant="sd",
                        enable_cfg_parallel=True,
                    )
                )

    def test_cli_rejects_unknown_mode(self):
        from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
            Flux3ImagePipelineConfig,
        )

        parser = argparse.ArgumentParser()
        Flux3ImagePipelineConfig.add_pipeline_cli_args(parser)
        self.assertEqual(parser.parse_args(["--flux3-mode", "i2i"]).flux3_mode, "i2i")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--flux3-mode", "unsupported"])


class TestFlux3ServedOutput(unittest.TestCase):
    def test_decode_output_is_normalized_once_before_worker_materialization(self):
        from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            Flux3ImageDecodeStage,
        )

        image = torch.tensor([[[-1.0]], [[0.0]], [[1.0]]])
        displayed = Flux3ImageDecodeStage._to_display(image)
        req = SimpleNamespace(
            enable_frame_interpolation=False,
            enable_upscaling=False,
        )

        frame = GPUWorker._materialize_frame_output(displayed[0], OutputBatch(), req)

        self.assertTrue(displayed.dtype.is_floating_point)
        self.assertEqual(tuple(displayed.shape), (1, 3, 1, 1))
        self.assertEqual(frame[0, 0, 0].tolist(), [0, 127, 255])


class TestFlux3Package(unittest.TestCase):
    def test_config_and_dit_mapping(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            Flux3ImagePackage,
            Flux3ImagePackageConfig,
        )
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            flux3_image_dit_config_from_package,
        )

        package_config = Flux3ImagePackageConfig.from_dict(_config_dict())
        package = Flux3ImagePackage(
            root=None,
            config=package_config,
            weights=None,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            manifest=None,
        )
        config = flux3_image_dit_config_from_package(package)
        arch = config.arch_config
        self.assertIsNone(arch.vec_in_dim)
        self.assertEqual(arch.out_modes, ["image"])
        self.assertEqual(arch.context_in_dim, 20480)
        self.assertEqual(arch.num_channels_latents, 128)
        # The native DiT owns its checkpoint mapping (packed to_qkv).
        self.assertEqual(arch.param_names_mapping, {})

    def test_resolver_requires_complete_root_package(self, tmp_path=None):
        # unittest supplies no tmp_path; TemporaryDirectory keeps this portable.
        import tempfile
        from pathlib import Path

        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(_config_dict()))
            (root / "manifest.json").write_text(
                json.dumps({"format_version": 1, "model": "flux-3-image-klein"})
            )
            (root / "model.safetensors").touch()
            (root / "ae.safetensors").touch()
            (root / "text_encoder").mkdir()
            (root / "tokenizer").mkdir()
            package = resolve_flux3_image_package(root)
            self.assertEqual(package.weights.name, "model.safetensors")
            self.assertEqual(package.vae.name, "ae.safetensors")

    def test_resolver_selects_quantized_variants_with_shared_root_assets(self):
        import tempfile
        from pathlib import Path

        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root_config = _config_dict()
            (root / "config.json").write_text(json.dumps(root_config))
            (root / "manifest.json").write_text(
                json.dumps({"format_version": 1, "model": "flux-3-image-klein"})
            )
            (root / "model.safetensors").touch()
            (root / "ae.safetensors").touch()
            (root / "text_encoder").mkdir()
            (root / "tokenizer").mkdir()

            variant = root / "variants" / "fp8r"
            variant.mkdir(parents=True)
            fp8r_config = dict(root_config)
            fp8r_config["quantization"] = "fp8r"
            (variant / "config.json").write_text(json.dumps(fp8r_config))
            (variant / "model.safetensors").touch()
            (variant / "hf_quant_config.json").write_text(
                json.dumps(
                    {
                        "producer": {"name": "bfl"},
                        "quantization": {
                            "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
                            "exclude_modules": [],
                        },
                    }
                )
            )
            (variant / "manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "model": "flux-3-image-klein",
                        "quantization": "fp8r",
                        "dtype": "float8_e4m3fn",
                        "scale_convention": ("weight.float() * weight_scale[:, None]"),
                        "tensors": 591,
                        "fp8_matrices": 234,
                        "weight_scales": 234,
                        "bf16_matrices": 37,
                    }
                )
            )

            package = resolve_flux3_image_package(root, variant="fp8r")
            direct = resolve_flux3_image_package(variant)
            self.assertEqual(package.variant, "fp8r")
            self.assertEqual(package.directory, variant)
            self.assertEqual(package.weights, variant / "model.safetensors")
            self.assertEqual(package.vae, root / "ae.safetensors")
            self.assertEqual(package.text_encoder, root / "text_encoder")
            self.assertEqual(package.tokenizer, root / "tokenizer")
            self.assertEqual(package.quant_config, variant / "hf_quant_config.json")
            self.assertEqual(direct, package)

            nvfp4 = root / "variants" / "nvfp4"
            nvfp4.mkdir()
            nvfp4_config = dict(root_config)
            nvfp4_config["quantization"] = "nvfp4"
            (nvfp4 / "config.json").write_text(json.dumps(nvfp4_config))
            (nvfp4 / "model.safetensors").touch()
            (nvfp4 / "hf_quant_config.json").write_text(
                json.dumps(
                    {
                        "producer": {"name": "bfl"},
                        "quantization": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                            "exclude_modules": [],
                        },
                    }
                )
            )
            (nvfp4 / "manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "model": "flux-3-image-klein",
                        "quantization": "nvfp4",
                        "dtype": "nvfp4: weight uint8",
                        "group_size": 16,
                        "scale_convention": (
                            "e2m1(weight) * weight_scale[:, k // 16] * "
                            "weight_scale_2; weight_scale_2 = amax(weight) / "
                            "(448 * 6); input_scale = p99.9 of per-call input "
                            "amax / (448 * 6)"
                        ),
                        "tensors": 1059,
                        "nvfp4_matrices": 234,
                        "weight_scales": 234,
                        "weight_scale_2": 234,
                        "input_scales": 234,
                        "bf16_matrices": 37,
                    }
                )
            )
            nvfp4_package = resolve_flux3_image_package(root, variant="nvfp4")
            self.assertEqual(nvfp4_package.variant, "nvfp4")
            self.assertEqual(nvfp4_package.directory, nvfp4)
            self.assertEqual(nvfp4_package.vae, root / "ae.safetensors")

            # Re-exports keep more layers in BF16 and add calibration-only clauses.
            manifest = json.loads((nvfp4 / "manifest.json").read_text())
            manifest.update(
                scale_convention=manifest["scale_convention"].replace(
                    "; input_scale",
                    "; weight_scale = the positive finite e4m3 code minimizing "
                    "the block's squared error; input_scale",
                ),
                weight_scales={"method": "mse_fp8_scale_sweep", "candidates": 126},
                weight_scale_tensors=198,
                tensors=951,
                nvfp4_matrices=198,
                weight_scale_2=198,
                input_scales=198,
                bf16_matrices=73,
            )
            (nvfp4 / "manifest.json").write_text(json.dumps(manifest))
            resolve_flux3_image_package(root, variant="nvfp4")

            manifest["input_scales"] = 197
            (nvfp4 / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "one scale set each"):
                resolve_flux3_image_package(root, variant="nvfp4")

            manifest["input_scales"] = 198
            manifest["scale_convention"] = manifest["scale_convention"].replace(
                "(448 * 6)", "448", 1
            )
            (nvfp4 / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "scale_convention lacks"):
                resolve_flux3_image_package(root, variant="nvfp4")

            with self.assertRaisesRegex(ValueError, "unsupported.*bad"):
                resolve_flux3_image_package(root, variant="bad")


class TestFlux3TextEncoder(unittest.TestCase):
    def test_batch_encodes_each_prompt_with_reference_positions(self):
        """Pad rows reach the DiT, so they must match the reference encoder: each
        prompt alone in its own bucket, with real tokens at 0..n-1 and pads at 1.
        Batched padding and the transformers default positions both moved them."""
        from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import (
            Flux3TextEncoder,
        )

        class FakeTokenizer:
            lengths = {"short": 3, "long": 6}

            def __call__(self, text, **kwargs):
                if "return_tensors" not in kwargs:
                    length = min(self.lengths[text], kwargs["max_length"])
                    return {"input_ids": list(range(length))}
                length = kwargs["max_length"]
                attention_mask = torch.zeros((1, length), dtype=torch.long)
                attention_mask[0, : self.lengths[text]] = 1
                return {
                    "input_ids": torch.arange(length)[None],
                    "attention_mask": attention_mask,
                }

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            @staticmethod
            def apply_chat_template(message, **_kwargs):
                return message[0]["content"]

        class FakeBaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.call_kwargs = []

            def forward(self, **kwargs):
                self.call_kwargs.append(kwargs)
                hidden = kwargs["input_ids"].unsqueeze(-1).expand(-1, -1, 2)
                return SimpleNamespace(hidden_states=(hidden,))

        base_model = FakeBaseModel()
        encoder = Flux3TextEncoder.__new__(Flux3TextEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.embed_dtype = torch.bfloat16
        encoder.output_layers = (0,)
        encoder.max_length = 16
        encoder.pad_multiple = 4
        encoder.processor = FakeProcessor()
        encoder.model = base_model

        result = encoder.encode_batch(["short", "long"])

        self.assertEqual(
            [tuple(call["input_ids"].shape) for call in base_model.call_kwargs],
            [(1, 4), (1, 8)],
        )
        positions = base_model.call_kwargs[0]["position_ids"]
        self.assertEqual(tuple(positions.shape), (4, 1, 4))
        self.assertEqual(positions[:, 0].tolist(), [[0, 1, 2, 1]] * 4)
        self.assertEqual(
            base_model.call_kwargs[1]["position_ids"][0, 0].tolist(),
            [0, 1, 2, 3, 4, 5, 1, 1],
        )
        self.assertEqual(
            [tuple(tokens.shape) for tokens, _ in result], [(4, 2), (8, 2)]
        )
        self.assertEqual([tuple(ids.shape) for _, ids in result], [(4, 4), (8, 4)])
        self.assertEqual(result[0][1][-1, -1].item(), 3)
        self.assertEqual(result[1][1][-1, -1].item(), 7)

    def test_constructor_truncates_and_compiles_language_model(self):
        from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import (
            Flux3TextEncoder,
        )

        class FakeLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(36)]
                )
                self.norm = torch.nn.Linear(1, 1)

            def forward(self, value):
                return value

        class FakeConditionalGeneration(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(
                    language_model=FakeLanguageModel(), visual=object()
                )

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        tokenizer = SimpleNamespace(padding_side="right")
        processor_cls = mock.Mock()
        processor_cls.from_pretrained.return_value = SimpleNamespace(
            tokenizer=tokenizer
        )
        transformers = SimpleNamespace(
            AutoProcessor=processor_cls,
            Qwen3VLForConditionalGeneration=FakeConditionalGeneration,
        )

        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch(
                "torch.compile", side_effect=lambda function, **_kwargs: function
            ) as compile_mock,
        ):
            encoder = Flux3TextEncoder(
                "model",
                output_layers=(4, 8),
                tokenizer_spec="tokenizer",
                compile_language_model=True,
            )

        language_model = encoder.model.language_model
        self.assertEqual(len(language_model.layers), 8)
        self.assertIsInstance(language_model.norm, torch.nn.Identity)
        self.assertFalse(hasattr(encoder.model, "visual"))
        processor_cls.from_pretrained.assert_called_once_with("tokenizer")
        self.assertTrue(compile_mock.call_args.kwargs["dynamic"])


class TestFlux3PackedCFGParallel(unittest.TestCase):
    def test_one_sample_per_rank_is_gathered_in_cfg_order(self):
        from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT, pack
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
            flux3_image as stages,
        )

        class Model:
            def __call__(self, *, packed):
                return {"image": packed.data["image"]}

        image_ids = torch.zeros(2, 4)
        text = (torch.zeros(1, 3), torch.zeros(1, 4))
        packed = pack(
            [
                {
                    "image": (torch.zeros(2, 2), image_ids),
                    "image_cond": None,
                    TEXT: text,
                },
                {
                    "image": (torch.ones(2, 2), image_ids),
                    "image_cond": None,
                    TEXT: text,
                },
            ],
            {"image": 2, "image_cond": 2, TEXT: 3},
        )
        with (
            mock.patch.object(
                stages,
                "get_classifier_free_guidance_world_size",
                return_value=2,
            ),
            mock.patch.object(
                stages,
                "get_classifier_free_guidance_rank",
                return_value=0,
            ),
            mock.patch.object(
                stages,
                "cfg_model_parallel_all_gather",
                side_effect=lambda value, dim: torch.cat((value, value + 1), dim=dim),
            ),
        ):
            output = stages._CFGParallelModel(Model())(packed=packed)
        torch.testing.assert_close(
            output["image"],
            torch.cat((torch.zeros(2, 2), torch.ones(2, 2))),
        )


class TestFlux3StrictStateDict(unittest.TestCase):
    def test_full_state_dict_loads_and_partial_fails(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image import (
            require_full_state_dict,
        )

        model = torch.nn.Linear(2, 3)
        state = dict(model.state_dict())
        require_full_state_dict(model, state, name="stub")
        state.pop("bias")
        with self.assertRaisesRegex(RuntimeError, "missing key"):
            require_full_state_dict(model, state, name="stub")


if __name__ == "__main__":
    unittest.main()
