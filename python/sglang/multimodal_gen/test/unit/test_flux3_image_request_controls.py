# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
    Flux3ImagePipelineConfig,
    Flux3ImageSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType


class TestFlux3ImageRequestControls(unittest.TestCase):
    def test_only_image_modes_are_served(self):
        config = Flux3ImagePipelineConfig()
        for mode in ("t2i", "i2i"):
            config.flux3_mode = mode
            config.check_pipeline_config()
            self.assertEqual(config.served_modes(), frozenset({"t2i", "i2i"}))
        config.flux3_mode = "unsupported"
        with self.assertRaisesRegex(ValueError, "flux3_mode"):
            config.check_pipeline_config()

    def test_sampling_params_reject_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "flux3_mode"):
            Flux3ImageSamplingParams(flux3_mode="unsupported")

    def test_flow_map_rejects_any_requested_cfg_scale(self):
        """The distilled flow map has no CFG; like the reference, any scale
        other than 1 is an error, not just scales above 1."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            _cfg_scale,
        )

        flow_map = SimpleNamespace(
            dit=SimpleNamespace(flow_map=True), sampler=SimpleNamespace(guidance=None)
        )
        for scale in (None, 1.0):
            batch = SimpleNamespace(true_cfg_scale=None, guidance_scale=scale)
            self.assertIsNone(_cfg_scale(batch, SimpleNamespace(), flow_map))
        for scale in (0.5, 3.5):
            batch = SimpleNamespace(true_cfg_scale=None, guidance_scale=scale)
            with self.assertRaisesRegex(ValueError, "without CFG"):
                _cfg_scale(batch, SimpleNamespace(), flow_map)

    def test_default_negative_prompt_is_empty(self):
        """The base model's CFG negative is the empty prompt; the generic default
        is a long Wan negative prompt that changed every served image."""
        self.assertEqual(Flux3ImageSamplingParams(prompt="a fox").negative_prompt, "")

    def test_sampling_params_are_image_requests(self):
        self.assertEqual(Flux3ImageSamplingParams().data_type, DataType.IMAGE)

    def test_t2i_warmup_drops_synthetic_reference(self):
        params = Flux3ImageSamplingParams(flux3_mode="t2i")
        req = SimpleNamespace(image_path="warmup.png", extra={})
        server_args = SimpleNamespace(pipeline_config=SimpleNamespace(flux3_mode="i2i"))
        params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        self.assertIsNone(req.image_path)

    def test_i2i_warmup_keeps_reference(self):
        params = Flux3ImageSamplingParams(flux3_mode="i2i")
        req = SimpleNamespace(image_path="warmup.png", extra={})
        server_args = SimpleNamespace(pipeline_config=SimpleNamespace(flux3_mode="t2i"))
        params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        self.assertEqual(req.image_path, "warmup.png")

    def test_i2i_server_warmup_without_reference_runs_t2i(self):
        """Under breakable CUDA graphs no warmup reference is synthesized; an
        i2i-default server must still warm up instead of failing on the
        missing image."""
        params = Flux3ImageSamplingParams()
        req = SimpleNamespace(image_path=None, extra={})
        server_args = SimpleNamespace(pipeline_config=SimpleNamespace(flux3_mode="i2i"))
        params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        self.assertIsNone(req.image_path)
        self.assertEqual(params.flux3_mode, "t2i")

    def test_warmup_resolves_the_default_step_target(self):
        """A request without steps means the server default. Server warmup
        (e.g. for breakable CUDA graphs) copies the request's steps as the
        full-workload target, and ``int(None)`` there failed the warmup."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        params = Flux3ImageSamplingParams(flux3_mode="t2i")
        req = Req(sampling_params=params)
        req.set_as_warmup(1)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(flux3_mode="t2i", flux3_steps=28)
        )
        params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        self.assertEqual(req.num_inference_steps, 1)
        self.assertEqual(req.extra["warmup_target_num_inference_steps"], 28)
        self.assertEqual(req.extra["cache_dit_num_inference_steps"], 28)

    def test_image_api_accepts_the_request_mode(self):
        """The image endpoints only forward fields the model declares, so an
        undeclared ``flux3_mode`` is silently dropped and the server default
        runs instead."""
        import dataclasses

        from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
            get_sampling_request_extra_fields,
        )

        fields = {
            f.name for f in dataclasses.fields(Flux3ImageSamplingParams) if f.init
        }
        self.assertLessEqual(
            Flux3ImageSamplingParams.image_request_extra_fields(), fields
        )
        self.assertIn(
            "flux3_mode",
            get_sampling_request_extra_fields(Flux3ImageSamplingParams, "image"),
        )

    def test_reference_image_without_mode_selects_i2i(self):
        """The image-edit endpoint uploads a reference but cannot name a mode;
        on a t2i-default server that used to fail as conditioning on t2i."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            _mode,
        )

        config = Flux3ImagePipelineConfig(flux3_mode="t2i")
        edit = SimpleNamespace(flux3_mode=None, image_path=["reference.png"])
        self.assertEqual(_mode(edit, config), "i2i")
        plain = SimpleNamespace(flux3_mode=None, image_path=None)
        self.assertEqual(_mode(plain, config), "t2i")
        explicit = SimpleNamespace(flux3_mode="t2i", image_path=["reference.png"])
        self.assertEqual(_mode(explicit, config), "t2i")


if __name__ == "__main__":
    unittest.main()
