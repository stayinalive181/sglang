# SPDX-License-Identifier: Apache-2.0
"""CPU-safe FLUX.3 integration tests for the ModelOpt NVFP4 path."""

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization import flux3_prims_ts
from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
    Flux3ModelOptFp4Config,
    Flux3Nvfp4LinearMethod,
    Flux3PrimsTsRole,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    _linear_nvfp4_scales_to_tma,
    _swizzled_nvfp4_scales_to_linear,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability

# These tests only construct layers; NVFP4 linear methods refuse pre-Blackwell
# devices at construction, so pin a Blackwell capability for the whole module.
_BLACKWELL_CAPABILITY = mock.patch.object(
    current_platform, "get_device_capability", return_value=DeviceCapability(10, 0)
)


def setUpModule():
    _BLACKWELL_CAPABILITY.start()


def tearDownModule():
    _BLACKWELL_CAPABILITY.stop()


_LEAVES = ("to_qkv", "mlp_in", "attn_out", "mlp_out")
_ALL_ROLES = {
    "to_qkv": Flux3PrimsTsRole.QKV_QKNORM_ROPE,
    "mlp_in": Flux3PrimsTsRole.SWIGLU,
    "attn_out": Flux3PrimsTsRole.LINEAR,
    "mlp_out": Flux3PrimsTsRole.LINEAR,
}


def _write_synthetic_safetensors(path, tensors, metadata):
    sizes = {"U8": 1, "F8_E4M3": 1, "F32": 4, "BF16": 2}
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = sizes[dtype] * math.prod(shape or (1,))
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\x00" * offset)


class TestBflNvfp4Contract(unittest.TestCase):
    def _checkpoint(self, directory, *, group_size=16):
        qkv = [f"single_blocks.0.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj")]
        quantized = qkv + [f"blocks.{index}.linear" for index in range(231)]
        excluded = [f"excluded.{index}" for index in range(37)]
        tensors = {}
        for prefix in quantized:
            tensors.update(
                {
                    f"{prefix}.weight": ("U8", [4, 8]),
                    f"{prefix}.weight_scale": ("F8_E4M3", [4, 1]),
                    f"{prefix}.weight_scale_2": ("F32", []),
                    f"{prefix}.input_scale": ("F32", []),
                }
            )
        tensors.update({f"{prefix}.weight": ("BF16", [4, 16]) for prefix in excluded})
        tensors.update({f"other.{index}.scale": ("BF16", [4]) for index in range(86)})
        weights = Path(directory) / "model.safetensors"
        _write_synthetic_safetensors(
            weights,
            tensors,
            metadata={"quantization": "nvfp4", "format": "pt"},
        )
        sidecar = Path(directory) / "hf_quant_config.json"
        sidecar.write_text(
            json.dumps(
                {
                    "producer": {"name": "bfl"},
                    "quantization": {
                        "quant_algo": "NVFP4",
                        "group_size": group_size,
                        "exclude_modules": excluded,
                    },
                }
            )
        )
        return weights, sidecar, excluded

    def test_builds_linear_static_modelopt_config(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            build_flux3_image_nvfp4_quant_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            weights, sidecar, excluded = self._checkpoint(directory)
            config = build_flux3_image_nvfp4_quant_config(weights, sidecar)

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertEqual(config.group_size, 16)
        self.assertEqual(config.checkpoint_weight_scale_layout, "linear")
        self.assertFalse(config.swap_weight_nibbles)
        self.assertFalse(config.dynamic_input_scale)
        self.assertTrue(config.is_checkpoint_nvfp4_serialized)
        # Three split BF16 Q/K/V paths collapse to one native to_qkv exclusion
        # only when they are excluded; this fixture quantizes them.
        self.assertEqual(len(excluded), 37)

    def test_rejects_wrong_sidecar_group_size(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            build_flux3_image_nvfp4_quant_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            weights, sidecar, _ = self._checkpoint(directory, group_size=32)
            with self.assertRaisesRegex(ValueError, "group_size=16"):
                build_flux3_image_nvfp4_quant_config(weights, sidecar)


def _tiny_quant_config(exclude_modules=(), *, dynamic_input_scale=False):
    return ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        exclude_modules=list(exclude_modules),
        checkpoint_weight_scale_layout="swizzled",
        dynamic_input_scale=dynamic_input_scale,
    )


def _flux3_config(*args, head_dim=128, **kwargs):
    return Flux3ModelOptFp4Config.from_base_config(
        _tiny_quant_config(*args, **kwargs), head_dim=head_dim
    )


def _prims_ts_device(supported=True):
    return mock.patch.object(
        flux3_prims_ts, "flux3_prims_ts_available", return_value=supported
    )


def _tiny_native_model(
    quant_config,
    *,
    tp_size=1,
    tp_rank=0,
    hidden_size=256,
    num_heads=2,
    depth=0,
    prims_ts_supported=True,
):
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        flux3_image_dit_config,
    )
    from sglang.multimodal_gen.runtime.models.dits import flux3_image
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import Flux3ImageDiT

    head_dim = hidden_size // num_heads
    config = flux3_image_dit_config(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        depth=depth,
        depth_single_blocks=1,
        context_in_dim=64,
        axes_dim=(head_dim // 4,) * 4,
    )
    group = object()
    with (
        mock.patch.object(linear, "get_tp_group", return_value=group),
        mock.patch.object(linear, "get_group_size", return_value=tp_size),
        mock.patch.object(linear, "get_group_rank", return_value=tp_rank),
        mock.patch.object(flux3_image, "get_tp_world_size", return_value=tp_size),
        _prims_ts_device(prims_ts_supported),
        mock.patch.object(torch, "get_default_dtype", return_value=torch.bfloat16),
    ):
        return Flux3ImageDiT(config, quant_config=quant_config)


def _roles(block):
    return {
        name: getattr(getattr(block, name).quant_method, "prims_ts_role", None)
        for name in _LEAVES
    }


class TestFlux3Nvfp4Construction(unittest.TestCase):
    def test_quantizes_native_modulation_linears(self):
        path = "early_stream_modulations.image.lin"
        model = _tiny_native_model(_flux3_config())
        layer = dict(model.named_modules())[path]

        self.assertIsInstance(layer.quant_method, Flux3Nvfp4LinearMethod)
        self.assertIsNone(layer.quant_method.prims_ts_role)

    def test_dynamic_activations_use_generic_nvfp4_backend(self):
        model = _tiny_native_model(
            _flux3_config(dynamic_input_scale=True),
            hidden_size=512,
            num_heads=4,
        )
        self.assertEqual(set(_roles(model.single_blocks[0]).values()), {None})

    def test_prims_ts_falls_back_on_unsupported_gpu(self):
        model = _tiny_native_model(
            _flux3_config(),
            hidden_size=512,
            num_heads=4,
            prims_ts_supported=False,
        )
        self.assertEqual(set(_roles(model.single_blocks[0]).values()), {None})

    def test_fused_qkv_requires_head_dim_128(self):
        model = _tiny_native_model(
            _flux3_config(head_dim=64), hidden_size=512, num_heads=4
        )
        self.assertEqual(
            _roles(model.single_blocks[0]),
            {**_ALL_ROLES, "to_qkv": Flux3PrimsTsRole.LINEAR},
        )

    def test_tp_local_shard_shapes_gate_prims_ts(self):
        """At TP=2 the QKV shard's N=768 no longer fits the 512-wide tiles."""
        model = _tiny_native_model(
            _flux3_config(), hidden_size=512, num_heads=4, tp_size=2
        )
        block = model.single_blocks[0]
        self.assertEqual(block.to_qkv.output_size_per_partition, 768)
        self.assertEqual(_roles(block), {**_ALL_ROLES, "to_qkv": None})

    def test_prims_ts_coverage_exempts_unsupported_hardware(self):
        """Static NVFP4 loads on the generic backend where PrimsTS is absent."""
        config = _flux3_config()
        model = _tiny_native_model(
            config, hidden_size=512, num_heads=4, depth=1, prims_ts_supported=False
        )
        self.assertEqual(set(_roles(model.single_blocks[0]).values()), {None})
        with _prims_ts_device(False):
            config.verify_prims_ts_coverage(model)

    def test_prims_ts_coverage_fails_closed_on_a_generic_projection(self):
        config = _flux3_config()
        model = _tiny_native_model(config, hidden_size=512, num_heads=4, depth=1)
        model.single_blocks[0].attn_out.quant_method.prims_ts_role = None
        model.single_blocks[
            0
        ].to_qkv.quant_method.prims_ts_role = Flux3PrimsTsRole.LINEAR
        with _prims_ts_device(), self.assertRaises(RuntimeError) as ctx:
            config.verify_prims_ts_coverage(model)
        message = str(ctx.exception)
        self.assertIn("single_blocks.0.attn_out", message)
        self.assertIn("single_blocks.0.to_qkv (QKV/QKNorm/RoPE epilogue)", message)
        self.assertNotIn("single_blocks.0.to_qkv,", message)
        self.assertNotIn("content_mode_blocks", message)

    def test_fused_mlp_up_passes_packed_fp4_directly_to_mlp_down(self):
        """The fused SwiGLU FP4 epilogue eliminates MLP-down requantization."""
        from sglang.kernels.ops.diffusion.gemm import flux3_nvfp4_prims_ts as nvfp4

        x = torch.zeros((2, 3, 512), dtype=torch.bfloat16)
        a_packed = torch.empty((6, 256), dtype=torch.uint8)
        a_scale = torch.empty((1536,), dtype=torch.uint8)
        mlp_up_payload = torch.empty((6, 512), dtype=torch.uint8)
        mlp_up_scale = torch.empty((3072,), dtype=torch.uint8)
        mlp_down_output = torch.empty((6, 512), dtype=torch.bfloat16)
        with (
            mock.patch.object(
                nvfp4,
                "_quantized_activation",
                return_value=(a_packed, a_scale, 0.25),
            ) as quantize,
            mock.patch.object(
                nvfp4.prims_ts_gemm,
                "fp4_linear_swiglu",
                return_value=(mlp_up_payload, mlp_up_scale),
            ) as mlp_up,
            mock.patch.object(
                nvfp4.prims_ts_gemm,
                "fp4_linear",
                return_value=mlp_down_output,
            ) as mlp_down,
        ):
            packed_activation = nvfp4.flux3_nvfp4_linear_swiglu_fp4(
                x,
                torch.empty((1024, 256), dtype=torch.uint8),
                torch.empty((1,), dtype=torch.uint8),
                0.25,
                0.5,
            )
            result = nvfp4.flux3_nvfp4_linear_from_packed_activation(
                packed_activation,
                torch.empty((512, 512), dtype=torch.uint8),
                torch.empty((1,), dtype=torch.uint8),
                0.5,
            )

        self.assertEqual(tuple(result.shape), (2, 3, 512))
        self.assertEqual(quantize.call_count, 1)
        self.assertEqual(mlp_up.call_args.kwargs["out_dtype"], torch.uint8)
        self.assertEqual(
            mlp_down.call_args.args[:3], (mlp_up_payload, mlp_up_scale, 1.0)
        )

    def test_prims_ts_plans_all_image_and_joint_block_gemms(self):
        """The NVFP4 fast path covers QKV, attention-out, MLP-up, and MLP-down."""
        config = _flux3_config()
        model = _tiny_native_model(config, hidden_size=512, num_heads=4, depth=1)
        for block in (model.content_mode_blocks["image"][0], model.single_blocks[0]):
            self.assertEqual(_roles(block), _ALL_ROLES)
        with _prims_ts_device():
            config.verify_prims_ts_coverage(model)

    def test_prims_ts_loader_preserves_128x4_scales_and_qkv_factors(self):
        model = _tiny_native_model(_flux3_config(), hidden_size=512, num_heads=4)
        block = model.single_blocks[0]
        for layer in (block.to_qkv, block.mlp_in, block.attn_out, block.mlp_out):
            layer.weight_scale.data.fill_(1.0)
            layer.input_scale.data.fill_(1.0)
            layer.weight_scale_2.data.fill_(1.0)
            if layer is block.to_qkv:
                layer.input_scale.data.copy_(torch.tensor([0.5, 1.0, 2.0]))
                layer.weight_scale_2.data.copy_(torch.tensor([1.0, 2.0, 4.0]))
            layer.quant_method.process_weights_after_loading(layer)
            rows, packed_k = layer.weight.shape
            scale_k = ((packed_k * 2 // 16 + 3) // 4) * 4
            expected_numel = ((rows + 127) // 128) * 128 * scale_k
            self.assertEqual(layer.weight_scale_prims_ts_128x4.dtype, torch.uint8)
            self.assertEqual(layer.weight_scale_prims_ts_128x4.ndim, 1)
            self.assertEqual(layer.weight_scale_prims_ts_128x4.numel(), expected_numel)
            if layer is block.to_qkv:
                # Exact per-partition factors bypass the generic rebasing.
                self.assertTrue(
                    torch.equal(
                        layer.weight_scale_prims_ts_128x4,
                        _linear_nvfp4_scales_to_tma(layer.weight_scale)
                        .view(torch.uint8)
                        .reshape(-1),
                    )
                )
                self.assertTrue(
                    torch.equal(layer.prims_ts_qkv_scale, torch.tensor([2.0, 4.0, 8.0]))
                )
                self.assertEqual(layer.prims_ts_input_global_scale, 2.0)
                self.assertEqual(layer.prims_ts_weight_global_scale, 4.0)
            else:
                self.assertFalse(hasattr(layer, "prims_ts_qkv_scale"))

    def test_plain_prims_ts_qkv_rebases_partition_scales(self):
        model = _tiny_native_model(
            _flux3_config(head_dim=64), hidden_size=512, num_heads=4
        )
        layer = model.single_blocks[0].to_qkv
        self.assertIs(layer.quant_method.prims_ts_role, Flux3PrimsTsRole.LINEAR)
        layer.weight_scale.data.fill_(1.0)
        layer.input_scale.data.fill_(1.0)
        layer.weight_scale_2.data.copy_(torch.tensor([1.0, 2.0, 4.0]))
        layer.quant_method.process_weights_after_loading(layer)

        width = layer.logical_widths[0]
        expected = torch.cat(
            [torch.full((width, 32), value) for value in (0.25, 0.5, 1.0)]
        ).to(torch.float8_e4m3fn)
        self.assertFalse(hasattr(layer, "prims_ts_qkv_scale"))
        self.assertTrue(
            torch.equal(
                layer.weight_scale_prims_ts_128x4,
                _linear_nvfp4_scales_to_tma(expected).view(torch.uint8).reshape(-1),
            )
        )

    def test_prims_ts_loader_interleaves_gated_mlp_rows_and_scales(self):
        """SwiGLU PrimsTS weights must be [value_0, gate_0, ...], not split."""
        model = _tiny_native_model(_flux3_config(), hidden_size=512, num_heads=4)
        layer = model.single_blocks[0].mlp_in
        self.assertIs(layer.quant_method.prims_ts_role, Flux3PrimsTsRole.SWIGLU)
        n, packed_k = layer.weight.shape
        source_weight = (
            torch.arange(n * packed_k, dtype=torch.int64)
            .to(torch.uint8)
            .view(n, packed_k)
        )
        source_scales = (
            torch.arange(n * layer.weight_scale.shape[1], dtype=torch.float32)
            .view_as(layer.weight_scale)
            .remainder(16)
            .to(torch.float8_e4m3fn)
        )
        layer.weight.data.copy_(source_weight)
        layer.weight_scale.data.copy_(source_scales)
        layer.weight_scale.nvfp4_scale_layout = "linear"
        layer.input_scale.data.fill_(1.0)
        layer.weight_scale_2.data.fill_(1.0)
        layer.quant_method.process_weights_after_loading(layer)

        half = n // 2
        expected_weight = torch.stack(
            (source_weight[half:], source_weight[:half]), dim=1
        ).reshape_as(source_weight)
        expected_scales = torch.stack(
            (source_scales[half:], source_scales[:half]), dim=1
        ).reshape_as(source_scales)
        self.assertTrue(torch.equal(layer.weight, expected_weight))
        self.assertTrue(
            torch.equal(
                layer.weight_scale_prims_ts_128x4,
                _linear_nvfp4_scales_to_tma(expected_scales)
                .view(torch.uint8)
                .reshape(-1),
            )
        )

    def test_prims_ts_prepared_layer_rejects_generic_apply(self):
        model = _tiny_native_model(_flux3_config(), hidden_size=512, num_heads=4)
        layer = model.single_blocks[0].attn_out
        with self.assertRaisesRegex(RuntimeError, "prepared for PrimsTS"):
            layer.quant_method.apply(layer, torch.zeros(1, 512, dtype=torch.bfloat16))

    def test_packed_qkv_maps_tensorwise_scales(self):
        from sglang.multimodal_gen.runtime.loader.utils import (
            get_param_names_mapping,
        )
        from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
            _QKV_PARAM_NAMES_MAPPING,
        )

        map_name = get_param_names_mapping(_QKV_PARAM_NAMES_MAPPING)
        for suffix in ("weight_scale_2", "input_scale"):
            for projection, shard in (("q_proj", 0), ("k_proj", 1), ("v_proj", 2)):
                target, mapped_shard, count = map_name(
                    f"content_mode_blocks.image.0.{projection}.{suffix}"
                )
                self.assertEqual(target, f"content_mode_blocks.image.0.to_qkv.{suffix}")
                self.assertEqual((mapped_shard, count), (shard, 3))

    def test_preprocess_promotes_split_qkv_scalars_for_concatenation(self):
        model = _tiny_native_model(_flux3_config())
        weights = [
            (f"single_blocks.0.{projection}.{suffix}", torch.tensor(0.5))
            for projection in ("q_proj", "k_proj", "v_proj")
            for suffix in ("weight_scale_2", "input_scale")
        ]
        processed = list(model.preprocess_loaded_state_dict(weights))

        self.assertTrue(processed)
        self.assertTrue(all(tuple(value.shape) == (1,) for _, value in processed))

    def test_single_block_uses_modelopt_linears(self):
        model = _tiny_native_model(_flux3_config())
        block = model.single_blocks[0]

        for module in (
            block.to_qkv,
            block.mlp_in,
            block.attn_out,
            block.mlp_out,
        ):
            self.assertIsInstance(module.quant_method, Flux3Nvfp4LinearMethod)
            self.assertIsInstance(module.quant_method, ModelOptFp4LinearMethod)
            self.assertEqual(module.weight.dtype, torch.uint8)
            self.assertEqual(module.weight_scale.dtype, torch.float8_e4m3fn)

    def test_flux3_method_keeps_the_v2_weight_loader(self):
        """Merged scalars such as ``mlp_in.weight_scale_2`` need the v2 loader."""
        block = _tiny_native_model(_flux3_config()).single_blocks[0]
        for name in _LEAVES:
            layer = getattr(block, name)
            for param in (layer.weight, layer.input_scale, layer.weight_scale_2):
                self.assertEqual(param.weight_loader, layer.weight_loader_v2)

    def test_merged_gated_mlp_preserves_logical_partitions(self):
        model = _tiny_native_model(_flux3_config())
        mlp = model.single_blocks[0].mlp_in

        self.assertEqual(mlp.output_partition_sizes, [768, 768])
        self.assertEqual(tuple(mlp.weight.shape), (1536, 128))
        self.assertEqual(tuple(mlp.weight_scale.shape), (1536, 16))
        self.assertEqual(tuple(mlp.input_scale.shape), (2,))
        self.assertEqual(tuple(mlp.weight_scale_2.shape), (2,))

    def test_excluded_prefix_stays_bf16(self):
        model = _tiny_native_model(
            _flux3_config(exclude_modules=("single_blocks.0.to_qkv",))
        )
        block = model.single_blocks[0]

        self.assertIsInstance(block.to_qkv.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.mlp_in.quant_method, Flux3Nvfp4LinearMethod)

    def test_dynamic_input_scales_declare_unit_initialization(self):
        model = _tiny_native_model(_flux3_config(dynamic_input_scale=True))
        input_scale_names = {
            name
            for name, _ in model.named_parameters()
            if name.endswith(".input_scale")
        }
        for name, parameter in model.named_parameters():
            if name in input_scale_names:
                self.assertEqual(parameter.missing_param_init, "ones")

    def test_tp_shards_packed_gated_and_row_parallel_weights(self):
        from sglang.multimodal_gen.runtime.models import parameter as parameter_module

        model = _tiny_native_model(_flux3_config(), tp_size=2, tp_rank=1)
        block = model.single_blocks[0]

        full_mlp_weight = (
            torch.arange(1536 * 128, dtype=torch.int64).to(torch.uint8).view(1536, 128)
        )
        with mock.patch.object(parameter_module, "get_tp_rank", return_value=1):
            block.mlp_in.weight.weight_loader(block.mlp_in.weight, full_mlp_weight)
        expected_mlp = torch.cat((full_mlp_weight[384:768], full_mlp_weight[1152:1536]))
        self.assertTrue(torch.equal(block.mlp_in.weight, expected_mlp))

        full_out_weight = (
            torch.arange(256 * 128, dtype=torch.int64).to(torch.uint8).view(256, 128)
        )
        full_out_scale = (
            (torch.arange(256 * 16, dtype=torch.int64) % 16)
            .to(torch.float8_e4m3fn)
            .view(256, 16)
        )
        with mock.patch.object(parameter_module, "get_tp_rank", return_value=1):
            block.attn_out.weight.weight_loader(block.attn_out.weight, full_out_weight)
            block.attn_out.weight_scale.weight_loader(
                block.attn_out.weight_scale, full_out_scale
            )
        self.assertTrue(torch.equal(block.attn_out.weight, full_out_weight[:, 64:]))
        # The config declares swizzled scales, so the loader normalizes them to
        # row-major before narrowing the input axis.
        self.assertTrue(
            torch.equal(
                block.attn_out.weight_scale,
                _swizzled_nvfp4_scales_to_linear(full_out_scale)[:, 8:],
            )
        )

    def test_row_parallel_weight_scale_shard_is_the_logical_shard(self):
        """A rank's row-parallel scales must be its own K range's scales.

        ``weight_scale`` is serialized in the 128x4 swizzled layout, where a
        stored row mixes logical ``m`` and logical ``k``. Narrowing dim 1 of
        those bytes as if it were the logical ``K/group_size`` axis hands each
        rank scales that belong to other blocks, which silently degrades every
        quantized row-parallel projection under TP>1.
        """
        from sglang.multimodal_gen.runtime.models import parameter as parameter_module

        tp_size, tp_rank = 2, 1
        config = _flux3_config()
        model = _tiny_native_model(config, tp_size=tp_size, tp_rank=tp_rank)
        attn_out = model.single_blocks[0].attn_out

        rows = attn_out.weight_scale.shape[0]
        k_blocks = attn_out.weight_scale.shape[1] * tp_size
        # A checkpoint whose LOGICAL entry (row, k_block) is a distinct value,
        # stored the way a swizzled delivery stores it.
        logical_full = (
            (torch.arange(rows * k_blocks, dtype=torch.int64) % 240 + 1)
            .to(torch.float8_e4m3fn)
            .view(rows, k_blocks)
        )
        serialized = _linear_nvfp4_scales_to_tma(logical_full)

        with mock.patch.object(parameter_module, "get_tp_rank", return_value=tp_rank):
            attn_out.weight_scale.weight_loader(attn_out.weight_scale, serialized)

        # Recover the logical scales the way process_weights_after_loading does.
        stored = attn_out.weight_scale
        held = getattr(
            stored, "nvfp4_scale_layout", config.checkpoint_weight_scale_layout
        )
        recovered = (
            _swizzled_nvfp4_scales_to_linear(stored) if held == "swizzled" else stored
        )
        shard = k_blocks // tp_size
        expected = logical_full[:, tp_rank * shard : (tp_rank + 1) * shard]
        self.assertEqual(tuple(recovered.shape), tuple(expected.shape))
        self.assertTrue(
            torch.equal(recovered.view(torch.uint8), expected.view(torch.uint8))
        )

    def test_loader_selects_flux3_modelopt_fp4_config(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            select_flux3_image_quant_config,
        )

        base = _tiny_quant_config()
        # quantization_utils adjusts this after construction; it must survive.
        base.swap_weight_nibbles = True
        dit_config = SimpleNamespace(
            arch_config=SimpleNamespace(hidden_size=3072, num_attention_heads=24)
        )
        selected = select_flux3_image_quant_config(base, dit_config)

        self.assertIsInstance(selected, Flux3ModelOptFp4Config)
        self.assertEqual(selected.head_dim, 128)
        fallback = select_flux3_image_quant_config(
            base, dit_config, enable_prims_ts=False
        )
        self.assertIsInstance(fallback, Flux3ModelOptFp4Config)
        self.assertFalse(fallback.enable_prims_ts)
        self.assertTrue(selected.swap_weight_nibbles)
        self.assertEqual(selected.group_size, 16)
        self.assertEqual(selected.get_name(), "modelopt_fp4")
        self.assertNotIsInstance(base, Flux3ModelOptFp4Config)


class TestGenericModelOptHasNoFlux3Coupling(unittest.TestCase):
    def test_generic_modelopt_module_does_not_reference_flux3(self):
        import inspect

        from sglang.multimodal_gen.runtime.layers.quantization import modelopt_quant

        self.assertNotIn("flux3", inspect.getsource(modelopt_quant).lower())

    def test_generic_method_rebases_packed_partition_scales(self):
        """Dynamic NVFP4 keeps one global decode scale on the generic backend."""
        from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
            _rebase_partition_weight_scales,
        )

        scales = torch.ones(6, 2).to(torch.float8_e4m3fn)
        rebased = _rebase_partition_weight_scales(
            scales, [2, 4], torch.tensor([1.0, 4.0]), torch.tensor(4.0)
        )
        self.assertTrue(
            torch.equal(
                rebased.float(),
                torch.tensor([[0.25] * 2] * 2 + [[1.0] * 2] * 4),
            )
        )


if __name__ == "__main__":
    unittest.main()
