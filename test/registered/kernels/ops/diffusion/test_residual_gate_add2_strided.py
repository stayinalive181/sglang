# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.kernels.ops.diffusion import (
    can_use_residual_gate_add2_strided,
    residual_gate_add2,
    residual_gate_add2_strided,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestResidualGateAdd2Strided(CustomTestCase):
    @torch.inference_mode()
    def test_slice_broadcast_and_rounding(self):
        torch.manual_seed(27)
        for batch, length, dim in [
            (1, 1024, 1536),
            (2, 320, 3072),
            (2, 193, 3073),
            (2, 4096, 3072),
        ]:
            for gate_batch, gate_length in [(batch, length), (batch, 1), (1, 1)]:
                with self.subTest(
                    shape=(batch, length, dim), gate=(gate_batch, gate_length)
                ):

                    def rand(shape):
                        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

                    # Different offsets/row strides exercise split streams, padded
                    # residuals, and the gate view of a packed modulation tensor.
                    residual = rand((batch, length + 5, dim + 1))[
                        :, 1 : length + 1, :dim
                    ]
                    a = rand((batch, length + 320, dim))[:, 320:]
                    b = rand((batch, length + 7, dim))[:, :length]
                    gate = rand((gate_batch, gate_length, 3 * dim))[..., 2 * dim :]
                    tensors = (residual, a, b, gate)
                    saved = [t.clone() for t in tensors]
                    expected = torch.addcmul(residual, gate, a + b)
                    self.assertTrue(can_use_residual_gate_add2_strided(*tensors))
                    actual = residual_gate_add2_strided(*tensors)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(residual_gate_add2(*tensors), expected))
                    for original, snapshot in zip(tensors, saved):
                        self.assertTrue(torch.equal(original, snapshot))

    @torch.inference_mode()
    def test_unsupported_falls_back(self):
        for dtype in (torch.float32, torch.float16):
            x = torch.randn((2, 320, 3072), device="cuda", dtype=dtype)
            self.assertFalse(can_use_residual_gate_add2_strided(x, x, x, x))
            self.assertTrue(
                torch.equal(residual_gate_add2(x, x, x, x), torch.addcmul(x, x, x + x))
            )

    @torch.inference_mode()
    def test_compile(self):
        x = torch.randn((2, 320, 3072), device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(
            residual_gate_add2_strided, fullgraph=True, backend="eager"
        )
        self.assertTrue(torch.equal(compiled(x, x, x, x), torch.addcmul(x, x, x + x)))

    def test_large_storage_guard(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.empty((2, 350000, 3072), device="cuda", dtype=torch.bfloat16)
            self.assertFalse(can_use_residual_gate_add2_strided(x, x, x, x))
            # Expanded inputs can have small storage but an oversized output.
            x = torch.empty((1, 1, 3072), device="cuda", dtype=torch.bfloat16)
            x = x.expand(2, 350000, 3072)
            self.assertFalse(can_use_residual_gate_add2_strided(x, x, x, x))

    @torch.inference_mode()
    def test_modulation_matches_reference_without_stale_outputs(self):
        from sglang.multimodal_gen.runtime.models.dits.flux3 import Flux3Modulation
        from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
            Flux3ImageModulation,
        )

        reference = Flux3Modulation(32).cuda().bfloat16()
        candidate = Flux3ImageModulation(32, prefix="mod", quant_config=None)
        candidate = candidate.cuda().bfloat16()
        candidate.lin.weight.copy_(reference.lin.weight)
        for _ in range(2):
            x = torch.randn((2, 17, 32), device="cuda", dtype=torch.bfloat16)
            expected, actual = reference(x), candidate(x)
            self.assertEqual(len(expected), len(actual))
            # shift and scale feed fused kernels that need contiguous inputs
            self.assertTrue(actual[0].is_contiguous())
            self.assertTrue(actual[1].is_contiguous())
            for want, got in zip(expected, actual):
                torch.testing.assert_close(got, want)


if __name__ == "__main__":
    unittest.main()
