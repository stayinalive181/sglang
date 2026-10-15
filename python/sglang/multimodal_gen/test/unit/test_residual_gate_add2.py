"""The two-branch gated residual must equal addcmul(residual, gate, a + b) exactly."""

import unittest

import torch

from sglang.kernels.ops.diffusion import (
    can_use_residual_gate_add2_cuda,
    residual_gate_add2,
)

HAS_CUDA = torch.cuda.is_available()


def _reference(residual, a, b, gate):
    return torch.addcmul(residual, gate, a + b)


class TestResidualGateAdd2Eager(unittest.TestCase):
    """CPU inputs take the eager path, which is the contract the kernel copies."""

    def test_matches_addcmul_of_the_branch_sum(self):
        torch.manual_seed(0)
        residual = torch.randn(2, 5, 8)
        a, b = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
        for gate in (torch.randn(2, 5, 8), torch.randn(1, 1, 8)):
            got = residual_gate_add2(residual, a, b, gate)
            torch.testing.assert_close(
                got, _reference(residual, a, b, gate), rtol=0, atol=0
            )

    def test_cpu_inputs_are_rejected_by_the_cuda_gate(self):
        t = torch.zeros(2, 4)
        self.assertFalse(can_use_residual_gate_add2_cuda(t, t, t, t))


@unittest.skipUnless(HAS_CUDA, "requires CUDA")
class TestResidualGateAdd2Cuda(unittest.TestCase):
    def test_bit_identical_to_addcmul(self):
        torch.manual_seed(0)
        # Cover the full stream, a shorter batched stream, and the small-input
        # fallback. BLC slices may use Triton below the older CUDA threshold.
        for dtype in (torch.bfloat16, torch.float16):
            for shape in ((1, 27_280, 3072), (2, 331, 3072), (1, 17, 256)):
                residual = torch.randn(shape, device="cuda", dtype=dtype)
                a = torch.randn(shape, device="cuda", dtype=dtype)
                b = torch.randn(shape, device="cuda", dtype=dtype)
                for gate in (
                    torch.randn(shape, device="cuda", dtype=dtype),
                    torch.randn(
                        (1,) * (len(shape) - 1) + (shape[-1],),
                        device="cuda",
                        dtype=dtype,
                    ),
                ):
                    got = residual_gate_add2(residual, a, b, gate)
                    want = _reference(residual, a, b, gate)
                    self.assertTrue(
                        torch.equal(got, want),
                        f"{dtype} {shape} gate={tuple(gate.shape)}: "
                        f"max diff {(got.float() - want.float()).abs().max().item()}",
                    )

    def test_the_real_block_shape_takes_the_kernel(self):
        shape = (1, 27_280, 3072)
        residual, a, b = (
            torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )
        gate = torch.zeros(1, 1, 3072, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(can_use_residual_gate_add2_cuda(residual, a, b, gate))

    def test_unaligned_hidden_size_still_matches(self):
        torch.manual_seed(0)
        shape = (1, 17_000, 250)  # hidden size is not a multiple of the vector width
        residual, a, b, gate = (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
        )
        self.assertTrue(
            torch.equal(
                residual_gate_add2(residual, a, b, gate),
                _reference(residual, a, b, gate),
            )
        )

    def test_falls_back_when_an_input_is_strided(self):
        torch.manual_seed(0)
        base = torch.randn(1, 27_280, 6144, device="cuda", dtype=torch.bfloat16)
        strided = base[:, :, ::2]
        residual = torch.randn_like(strided)
        self.assertFalse(
            can_use_residual_gate_add2_cuda(residual, strided, strided, residual)
        )
        torch.testing.assert_close(
            residual_gate_add2(residual, strided, strided, residual),
            _reference(residual, strided, strided, residual),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
