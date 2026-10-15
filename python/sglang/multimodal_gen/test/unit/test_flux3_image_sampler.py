# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT, pack
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
    build_timesteps,
    denoise_flow_map,
    denoise_packed,
    resolve_sequential_cfg,
    timeshift,
)


class _VelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.timesteps = []

    def forward(self, packed):
        self.timesteps.append(packed.timesteps["image"].clone())
        outputs = []
        for sample in range(packed.num_samples):
            image_start, image_end = packed.mode_indices["image"][sample]
            text_start, text_end = packed.mode_indices[TEXT][sample]
            value = packed.data[TEXT][text_start:text_end].mean()
            outputs.append(
                torch.full(
                    (image_end - image_start, 2),
                    value,
                )
            )
        return {"image": torch.cat(outputs)}


class _FlowMapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, packed):
        self.calls.append(
            (packed.timesteps["image"][0].item(), packed.timesteps_ref.tolist())
        )
        return {"image": torch.ones_like(packed.data["image"])}


class TestFlux3PackedSampler(unittest.TestCase):
    def test_flow_map_jumps_to_the_next_time_without_cfg(self):
        """The reference flow map embeds the next schedule time as the jump
        target and steps ``x += (r - t) * u`` on the conditional sample alone."""
        schedule = build_timesteps(3, alpha=6.93)
        model = _FlowMapModel()
        packed = pack(
            [
                {
                    TEXT: (torch.zeros(2, 3), torch.zeros(2, 4)),
                    "image": (torch.zeros(4, 2), torch.zeros(4, 4)),
                }
            ],
            {"image": 2, TEXT: 3},
        )
        out = denoise_flow_map(model, packed, schedule)
        self.assertEqual(
            [ref for _, ref in model.calls], [[r] for r in schedule[1:].tolist()]
        )
        for (timestep, _), expected in zip(model.calls, schedule[:-1].tolist()):
            self.assertAlmostEqual(timestep, expected, places=2)
        # Constant velocity 1 from t=1 to 0 lands at -1.
        torch.testing.assert_close(out, torch.full((4, 2), -1.0), atol=1e-2, rtol=0)

        cfg_pair = pack(
            [
                {
                    TEXT: (torch.zeros(1, 3), torch.zeros(1, 4)),
                    "image": (torch.zeros(4, 2), torch.zeros(4, 4)),
                }
            ]
            * 2,
            {"image": 2, TEXT: 3},
        )
        with self.assertRaisesRegex(ValueError, "one sample"):
            denoise_flow_map(model, cfg_pair, schedule)

    def test_noise_keeps_the_sampling_state_in_fp32(self):
        """The reference rounds the noise to bf16 but steps in fp32; a bf16
        state drops the small early updates (dt ~ 1e-2 near t = 1)."""
        from types import SimpleNamespace

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            noise_tokens,
        )

        config = SimpleNamespace(
            dit=SimpleNamespace(in_channels={"image": 128}, image_rope_scale=0.5)
        )
        (noise, _), _ = noise_tokens(64, 64, 7, config, torch.device("cpu"))
        self.assertEqual(noise.dtype, torch.float32)
        torch.testing.assert_close(noise, noise.bfloat16().float(), atol=0, rtol=0)

    def test_shifted_schedule_has_terminal_zero(self):
        schedule = build_timesteps(4, alpha=6.93)
        self.assertEqual(schedule.shape, (5,))
        self.assertEqual(schedule[-1].item(), 0.0)
        torch.testing.assert_close(
            schedule,
            timeshift(6.93, torch.linspace(1.0, 0.0, 5)),
        )

    def test_batched_and_sequential_denoise_match(self):
        image = (torch.full((2, 2), 2.0), torch.zeros(2, 4))
        texts = [
            (torch.zeros(1, 3), torch.zeros(1, 4)),
            (torch.ones(1, 3), torch.zeros(1, 4)),
        ]
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                packed = pack(
                    [
                        {"image": image, "image_cond": None, TEXT: text}
                        for text in texts
                    ],
                    {"image": 2, "image_cond": 2, TEXT: 3},
                )
                original_text = packed.data[TEXT].clone()
                model = _VelocityModel()
                result = denoise_packed(
                    model,
                    packed,
                    torch.tensor([1.0, 0.0]),
                    guidance=3.5,
                    sequential_cfg=sequential,
                )
                torch.testing.assert_close(result, torch.full((2, 2), -1.5))
                self.assertTrue(torch.equal(packed.data[TEXT], original_text))
                self.assertTrue(torch.equal(packed.timesteps[TEXT], torch.zeros(2)))
                self.assertEqual(len(model.timesteps), 2 if sequential else 1)

    def test_cfg_layout_environment_override(self):
        image = (torch.zeros(1, 2), torch.zeros(1, 4))
        text = (torch.zeros(1, 3), torch.zeros(1, 4))
        packed = pack(
            [
                {"image": image, "image_cond": None, TEXT: text},
                {"image": image, "image_cond": None, TEXT: text},
            ],
            {"image": 2, "image_cond": 2, TEXT: 3},
        )
        with mock.patch.dict(os.environ, {"FLUX3_CFG_MODE": "sequential"}):
            self.assertTrue(resolve_sequential_cfg(packed, guidance=3.5))
        with mock.patch.dict(os.environ, {"FLUX3_CFG_MODE": "batched"}):
            self.assertFalse(resolve_sequential_cfg(packed, guidance=3.5))


if __name__ == "__main__":
    unittest.main()
