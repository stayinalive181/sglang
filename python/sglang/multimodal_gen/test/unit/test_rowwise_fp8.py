# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model-agnostic rowwise W8A8 FP8 quantization."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization import fp8
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
    RowwiseFp8Config,
)
from sglang.multimodal_gen.runtime.models import parameter

FP8_E4M3_MAX = 448.0


def _config(fp8_layers=("proj",), input_scale_layers=()):
    return RowwiseFp8Config(
        format_version="1.1",
        fp8_layers=frozenset(fp8_layers),
        input_scale_layers=frozenset(input_scale_layers),
    )


@contextmanager
def _mock_tp(tp_size=1, tp_rank=0):
    group = object()
    with (
        mock.patch.object(linear, "get_tp_group", return_value=group),
        mock.patch.object(linear, "get_group_size", return_value=tp_size),
        mock.patch.object(linear, "get_group_rank", return_value=tp_rank),
        mock.patch.object(parameter, "get_tp_rank", return_value=tp_rank),
    ):
        yield


class TestRowwiseFp8Config(unittest.TestCase):
    def test_selects_quant_method_by_prefix(self):
        config = _config()
        with _mock_tp():
            quantized = ColumnParallelLinear(
                8, 8, bias=False, prefix="proj", quant_config=config
            )
            unquantized = ColumnParallelLinear(
                8, 8, bias=False, prefix="other", quant_config=config
            )

        self.assertIsInstance(quantized.quant_method, Fp8LinearMethod)
        self.assertIsInstance(unquantized.quant_method, UnquantizedLinearMethod)

    def test_creates_parallel_rowwise_parameters(self):
        config = _config(
            fp8_layers=("column", "merged", "row"),
            input_scale_layers=("column",),
        )
        with _mock_tp(tp_size=2):
            column = ColumnParallelLinear(
                8, 12, bias=False, prefix="column", quant_config=config
            )
            merged = MergedColumnParallelLinear(
                8, [8, 12], bias=False, prefix="merged", quant_config=config
            )
            row = RowParallelLinear(
                12, 8, bias=False, prefix="row", quant_config=config
            )

        self.assertEqual(tuple(column.weight.shape), (6, 8))
        self.assertEqual(tuple(column.weight_scale.shape), (6,))
        self.assertEqual(tuple(column.input_scale.shape), (1,))
        self.assertEqual(tuple(merged.weight.shape), (10, 8))
        self.assertEqual(tuple(merged.weight_scale.shape), (10,))
        self.assertIsNone(merged.input_scale)
        self.assertEqual(tuple(row.weight.shape), (8, 6))
        self.assertEqual(tuple(row.weight_scale.shape), (8,))

    def test_v2_loaders_shard_channel_scales(self):
        config = _config(
            fp8_layers=("column", "merged", "row"),
            input_scale_layers=("column",),
        )
        with _mock_tp(tp_size=2, tp_rank=1):
            column = ColumnParallelLinear(
                8, 12, bias=False, prefix="column", quant_config=config
            )
            merged = MergedColumnParallelLinear(
                8, [8, 12], bias=False, prefix="merged", quant_config=config
            )
            row = RowParallelLinear(
                12, 8, bias=False, prefix="row", quant_config=config
            )

            column.weight_scale.weight_loader(
                column.weight_scale, torch.arange(12, dtype=torch.float32)
            )
            column.input_scale.weight_loader(column.input_scale, torch.tensor([0.125]))
            merged.weight_scale.weight_loader(
                merged.weight_scale, torch.arange(20, dtype=torch.float32)
            )
            row.weight_scale.weight_loader(
                row.weight_scale, torch.arange(8, dtype=torch.float32)
            )

        torch.testing.assert_close(
            column.weight_scale, torch.arange(6, 12, dtype=torch.float32)
        )
        torch.testing.assert_close(column.input_scale, torch.tensor([0.125]))
        torch.testing.assert_close(
            merged.weight_scale,
            torch.cat(
                (
                    torch.arange(4, 8, dtype=torch.float32),
                    torch.arange(14, 20, dtype=torch.float32),
                )
            ),
        )
        torch.testing.assert_close(
            row.weight_scale, torch.arange(8, dtype=torch.float32)
        )


class TestFp8LinearMethodExistingModes(unittest.TestCase):
    def test_parameter_layouts_remain_distinct(self):
        with _mock_tp():
            online = ColumnParallelLinear(8, 8, bias=False, quant_config=Fp8Config())
            serialized = ColumnParallelLinear(
                8,
                8,
                bias=False,
                quant_config=Fp8Config(is_checkpoint_fp8_serialized=True),
            )
            with mock.patch.object(
                fp8, "get_tensor_model_parallel_world_size", return_value=1
            ):
                block = ColumnParallelLinear(
                    8,
                    8,
                    bias=False,
                    quant_config=Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        weight_block_size=[4, 4],
                    ),
                )

        self.assertEqual(online.weight.dtype, torch.float32)
        self.assertFalse(hasattr(online, "weight_scale"))
        self.assertEqual(serialized.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(serialized.weight_scale.shape), (1,))
        self.assertFalse(serialized.quant_method.per_channel_serialized)
        self.assertEqual(tuple(block.weight_scale_inv.shape), (2, 2))
        self.assertTrue(block.quant_method.block_quant)


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9 or (major == 8 and minor >= 9)


@unittest.skipUnless(_fp8_capable(), "requires FP8-capable CUDA GPU")
class TestFp8LinearMethodRowwise(unittest.TestCase):
    @staticmethod
    def _layer(n, k, input_scale=None):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        weight_scale = (weight.float().abs().amax(dim=1) / FP8_E4M3_MAX).clamp(min=1e-8)
        quantized_weight = (
            (weight.float() / weight_scale[:, None])
            .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        return SimpleNamespace(
            # Matches the layout materialized by
            # Fp8LinearMethod.process_weights_after_loading.
            weight=quantized_weight.t(),
            weight_scale=weight_scale,
            input_scale=input_scale,
        )

    def _check(self, input_scale):
        torch.manual_seed(0)
        m, k, n = 8, 128, 64
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        layer = self._layer(n, k, input_scale)
        method = Fp8LinearMethod(_config())
        try:
            output = method.apply(layer, x)
        except (NotImplementedError, RuntimeError) as exc:
            self.skipTest(f"FP8 GEMM unavailable: {exc}")

        if input_scale is None:
            scale = (x.float().abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX).clamp(
                min=1e-8
            )
        else:
            scale = input_scale
        quantized_input = (
            (x.float() / scale)
            .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        reference = torch._scaled_mm(
            quantized_input,
            layer.weight,
            scale_a=scale.expand(m, 1).contiguous(),
            scale_b=layer.weight_scale.reshape(1, -1),
            out_dtype=torch.bfloat16,
        )
        cosine = F.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
        self.assertGreater(cosine.item(), 0.999)

    def test_dynamic_input_scale_matches_scaled_mm_reference(self):
        self._check(input_scale=None)

    def test_static_input_scale_matches_scaled_mm_reference(self):
        self._check(
            input_scale=torch.tensor([0.01], device="cuda", dtype=torch.float32)
        )


if __name__ == "__main__":
    unittest.main()
