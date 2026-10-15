# SPDX-License-Identifier: Apache-2.0
"""CPU-safe tests for the FLUX.3 PrimsTS tactic table and its TMEM overlap upgrade."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from flashinfer.prims_ts import gemm as prims_ts_gemm

from sglang.kernels.ops.diffusion.gemm import flux3_prims_ts_config as config_module
from sglang.multimodal_gen.runtime.layers.quantization import (
    flux3_prims_ts as quant_module,
)
from sglang.srt.environ import envs

_OVERLAP_EPILOGUES = frozenset({"swiglu", "qkv_qknorm_rope"})
_BUCKETS = (1024, 8192, 10240, 32768, 40960)


def _config(*, m, k, n, epilogue, operand_format="nvfp4_e2m1", **kwargs):
    x = SimpleNamespace(
        device=torch.device("cuda", 0), shape=(m, k), numel=lambda: m * k
    )
    # Configs are cached per shape; tests change the overlap support between calls.
    config_module._tuned_config.cache_clear()
    config_module._device_arch.cache_clear()
    with mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
        return config_module.flux3_prims_ts_config(
            x, operand_format=operand_format, epilogue=epilogue, k=k, n=n, **kwargs
        )


def _qkv(m):
    return _config(m=m, k=3072, n=9216, epilogue="qkv_qknorm_rope", head_dim=128)


def _swiglu(m, **kwargs):
    return _config(m=m, k=3072, n=18432, epilogue="swiglu", **kwargs)


class TestFlux3PrimsTsTmemOverlap(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SGLANG_ENABLE_FLUX3_PRIMS_TS_AUTOTUNE", None)
        config_module._tmem_overlap_epilogues.cache_clear()
        self.addCleanup(config_module._tmem_overlap_epilogues.cache_clear)

    def _overlap_support(self, epilogues):
        return mock.patch.object(
            config_module, "_tmem_overlap_epilogues", return_value=epilogues
        )

    def test_every_tuned_swiglu_and_qkv_row_takes_overlap_when_supported(self):
        # A re-tuned row outside the kernel's overlap constraints would silently
        # lose the FlashInfer PR A/B speed-up.
        with self._overlap_support(_OVERLAP_EPILOGUES):
            for bucket in _BUCKETS:
                configs = (
                    _swiglu(bucket),
                    _swiglu(bucket, output_format="nvfp4_e2m1"),
                    _qkv(bucket),
                )
                for config in configs:
                    self.assertTrue(config.tmem_overlap, (bucket, config))
                    self.assertFalse(config.use_tma_store, (bucket, config))

    def test_rows_are_unchanged_without_flashinfer_support(self):
        # FlashInfer main rejects SwiGLU/QKV overlap, so the table must pass through.
        with self._overlap_support(frozenset()):
            for bucket in _BUCKETS:
                self.assertFalse(_swiglu(bucket).tmem_overlap)
                qkv = _qkv(bucket)
                self.assertFalse(qkv.tmem_overlap)
                self.assertTrue(qkv.use_tma_store)

    def test_upgrade_leaves_other_configs_alone(self):
        cases = {
            "linear text bucket": lambda: _config(
                m=160, k=3072, n=3072, epilogue="linear"
            ),
            "fp8 swiglu": lambda: _config(
                m=8192, k=3072, n=18432, epilogue="swiglu", operand_format="fp8_e4m3"
            ),
            "untuned shape fallback": lambda: _config(
                m=8192, k=4096, n=4096, epilogue="swiglu"
            ),
        }
        for name, build in cases.items():
            with self._overlap_support(frozenset()):
                expected = build()
            with self._overlap_support(_OVERLAP_EPILOGUES):
                self.assertEqual(build(), expected, name)

    def test_repeated_lookups_return_the_same_config_object(self):
        # The prepared-GEMM cache keys on the config, so a fresh object per call
        # would rebuild and rehash it on every launch.
        with self._overlap_support(_OVERLAP_EPILOGUES):
            first = _swiglu(8192)
            with mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 3)
            ):
                x = SimpleNamespace(
                    device=torch.device("cuda", 0),
                    shape=(8192, 3072),
                    numel=lambda: 8192 * 3072,
                )
                again = config_module.flux3_prims_ts_config(
                    x, operand_format="nvfp4_e2m1", epilogue="swiglu", k=3072, n=18432
                )
        self.assertIs(again, first)

    def test_autotune_mode_returns_no_config(self):
        os.environ["SGLANG_ENABLE_FLUX3_PRIMS_TS_AUTOTUNE"] = "1"
        with self._overlap_support(_OVERLAP_EPILOGUES):
            self.assertIsNone(_swiglu(8192))

    def test_support_is_read_from_the_prepare_function_signatures(self):
        def swiglu_with_overlap(weight, *, config=None, tmem_overlap=False):
            pass

        def qkv_without_overlap(weight, *, config=None):
            pass

        with (
            mock.patch.object(
                prims_ts_gemm, "prepare_fp4_linear_swiglu", swiglu_with_overlap
            ),
            mock.patch.object(
                prims_ts_gemm, "prepare_fp4_qkv_qknorm_rope", qkv_without_overlap
            ),
        ):
            self.assertEqual(
                config_module._tmem_overlap_epilogues(), frozenset({"swiglu"})
            )

    def test_separate_qkv_output_is_sage_specific(self):
        with envs.SGLANG_USE_SAGE_ATTENTION_IMPL.override(""):
            self.assertFalse(quant_module._use_separate_qkv_output())
        for value in ("trtllm_gen", "trtllm-gen"):
            with envs.SGLANG_USE_SAGE_ATTENTION_IMPL.override(value):
                self.assertTrue(quant_module._use_separate_qkv_output())


if __name__ == "__main__":
    unittest.main()
