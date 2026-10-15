# SPDX-License-Identifier: Apache-2.0
"""Two-branch residual update for BLC slices and packed modulation views."""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _residual_gate_add2_strided_kernel(
    R,
    A,
    B,
    G,
    O,
    N: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    batch, token = row // L, row % L
    mask = i < N
    r = tl.load(R + batch * R0 + token * R1 + col, mask, 0).to(tl.float32)
    a = tl.load(A + batch * A0 + token * A1 + col, mask, 0).to(tl.float32)
    b = tl.load(B + batch * B0 + token * B1 + col, mask, 0).to(tl.float32)
    g = tl.load(G + batch * G0 + token * G1 + col, mask, 0).to(tl.float32)
    # aten first stores a+b in BF16; addcmul then uses an FP32 fused multiply-add.
    branch_sum = round_bf16_to_fp32(a + b)
    tl.store(O + i, tl.fma(g, branch_sum, r), mask)


def can_use_residual_gate_add2_strided(residual, update_a, update_b, gate) -> bool:
    if residual.ndim != 3 or residual.dtype != torch.bfloat16:
        return False
    if not residual.is_cuda or not (1 << 20 <= residual.numel() < 2**31 - 4096):
        return False
    if update_a.shape != residual.shape or update_b.shape != residual.shape:
        return False
    if gate.ndim != 3 or gate.shape[-1] != residual.shape[-1]:
        return False
    if any(g not in (1, r) for g, r in zip(gate.shape[:2], residual.shape[:2])):
        return False
    return all(
        t.device == residual.device and t.dtype == residual.dtype and t.stride(-1) == 1
        # Include padding in packed/sliced storage when bounding 32-bit indices.
        and sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        < 2**31 - 4096
        and not (torch.is_grad_enabled() and t.requires_grad)
        for t in (residual, update_a, update_b, gate)
    )


def _fake(residual, update_a, update_b, gate):
    return torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)


@register_custom_op(
    op_name="diffusion_residual_gate_add2_strided", mutates_args=[], fake_impl=_fake
)
def residual_gate_add2_strided(
    residual: torch.Tensor,
    update_a: torch.Tensor,
    update_b: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    if not can_use_residual_gate_add2_strided(residual, update_a, update_b, gate):
        raise ValueError("unsupported residual-gate2 strided inputs")
    out = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    with torch.cuda.device(residual.device):
        _residual_gate_add2_strided_kernel[(triton.cdiv(out.numel(), 1024),)](
            residual,
            update_a,
            update_b,
            gate,
            out,
            out.numel(),
            out.shape[1],
            out.shape[2],
            residual.stride(0),
            residual.stride(1),
            update_a.stride(0),
            update_a.stride(1),
            update_b.stride(0),
            update_b.stride(1),
            gate.stride(0) if gate.shape[0] != 1 else 0,
            gate.stride(1) if gate.shape[1] != 1 else 0,
            BLOCK=1024,
            num_warps=4,
        )
    return out
