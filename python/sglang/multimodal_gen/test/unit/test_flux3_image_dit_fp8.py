# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FLUX.3 rowwise-FP8 configuration and construction.

These are CPU-safe: the contract parser only reads safetensors headers, so the
tests craft *tiny synthetic* header-only safetensors files (correct
dtype/shape/offsets, zero-filled payload) — no confidential reference code and
no 13 GB checkpoint. Generic layer and kernel behavior is covered separately in
``test_rowwise_fp8.py``.
"""

import json
import math
import struct
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization import flux3_prims_ts
from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
    Flux3Fp8LinearMethod,
    Flux3PrimsTsRole,
    Flux3RowwiseFp8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
    RowwiseFp8Config,
    RowwiseFp8Error,
)
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    get_quant_config_from_safetensors_metadata,
)

# safetensors header dtype -> element byte size
_DTYPE_BYTES = {"F8_E4M3": 1, "F32": 4, "BF16": 2, "F16": 2}


def _write_synthetic_safetensors(path, tensors, metadata):
    """Write a header-only safetensors file.

    ``tensors`` maps name -> ``(dtype_str, shape)``. The payload is zero-filled
    (the parser never reads it). ``metadata`` is a str->str dict.
    """
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = (
            _DTYPE_BYTES[dtype] * math.prod(shape) if shape else _DTYPE_BYTES[dtype]
        )
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\x00" * offset)


def _fp8_layer_tensors(paths, out=8, in_=8, mlp_out=16, with_input_scale=()):
    """Build the weight/weight_scale[/input_scale] tensor headers for FP8 layers."""
    tensors = {}
    for p in paths:
        o = mlp_out if p.endswith("mlp_in") else out
        i = mlp_out if p.endswith("mlp_out") else in_
        tensors[f"{p}.weight"] = ("F8_E4M3", [o, i])
        tensors[f"{p}.weight_scale"] = ("F32", [o])
        if p in with_input_scale:
            tensors[f"{p}.input_scale"] = ("F32", [1])
    return tensors


def _fp8_meta(
    fp8_paths, bf16_paths=(), version="1.1", recipe="rowwise", extra_cfg=None
):
    layers = {p: {"format": "rowwise-float8_e4m3fn"} for p in fp8_paths}
    layers.update({p: {"format": "bf16"} for p in bf16_paths})
    quant_configs = {"rowwise-float8_e4m3fn": {"scale_recipe": recipe}}
    if extra_cfg:
        quant_configs.update(extra_cfg)
    meta = {
        "format_version": version,
        "layers": layers,
        "quant_configs": quant_configs,
    }
    return {"_quantization_metadata": json.dumps(meta, sort_keys=True)}


_FP8 = [
    f"single_blocks.0.{leaf}"
    for leaf in ("q_proj", "k_proj", "v_proj", "mlp_in", "attn_out", "mlp_out")
]
_BF16 = ["txt_in", "emb_in.image", "content_mode_blocks.image.0.q_proj"]


class TestContractParse(unittest.TestCase):
    def _ckpt(self, tmp, fp8=_FP8, bf16=_BF16, tensors=None, meta=None, with_is=()):
        import os

        path = os.path.join(tmp, "dit.safetensors")
        t = (
            tensors
            if tensors is not None
            else _fp8_layer_tensors(fp8, with_input_scale=with_is)
        )
        # add zero-size stand-ins for bf16 layer weights so counts are realistic
        for p in bf16:
            t.setdefault(f"{p}.weight", ("BF16", [4, 4]))
        m = meta if meta is not None else _fp8_meta(fp8, bf16, extra_cfg=None)
        if with_is:
            # rebuild tensors including input_scale
            t = _fp8_layer_tensors(fp8, with_input_scale=with_is)
            for p in bf16:
                t.setdefault(f"{p}.weight", ("BF16", [4, 4]))
        _write_synthetic_safetensors(path, t, m)
        return path

    def test_bf16_checkpoint_returns_none(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dit.safetensors")
            _write_synthetic_safetensors(
                path, {"txt_in.weight": ("BF16", [4, 4])}, metadata={}
            )
            self.assertIsNone(get_quant_config_from_safetensors_metadata(path))

    def test_mixed_layer_map(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            spec = get_quant_config_from_safetensors_metadata(self._ckpt(tmp))
            self.assertIsInstance(spec, RowwiseFp8Config)
            self.assertEqual(spec.num_fp8_layers, 6)
            self.assertEqual(spec.num_bf16_layers, 3)
            self.assertEqual(spec.num_linear_layers, 9)
            self.assertEqual(set(spec.fp8_layers), set(_FP8))
            self.assertEqual(spec.format_version, "1.1")
            self.assertEqual(len(spec.input_scale_layers), 0)

    def test_static_input_scale_detected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._ckpt(tmp, with_is=(_FP8[0],))
            spec = get_quant_config_from_safetensors_metadata(path)
            self.assertTrue(spec.has_input_scale(_FP8[0]))
            self.assertFalse(spec.has_input_scale(_FP8[1]))

    def _expect_error(self, tmp, *, tensors=None, meta=None, match=None):
        import os

        path = os.path.join(tmp, "dit.safetensors")
        t = tensors if tensors is not None else _fp8_layer_tensors(_FP8)
        m = meta if meta is not None else _fp8_meta(_FP8, _BF16)
        _write_synthetic_safetensors(path, t, m)
        with self.assertRaises(RowwiseFp8Error) as ctx:
            get_quant_config_from_safetensors_metadata(path)
        if match:
            self.assertIn(match, str(ctx.exception))

    def test_reject_unknown_version(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._expect_error(
                tmp, meta=_fp8_meta(_FP8, _BF16, version="2.0"), match="format_version"
            )

    def test_reject_unknown_format(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            meta = {
                "_quantization_metadata": json.dumps(
                    {
                        "format_version": "1.1",
                        "layers": {_FP8[0]: {"format": "int4"}},
                        "quant_configs": {
                            "rowwise-float8_e4m3fn": {"scale_recipe": "rowwise"}
                        },
                    }
                )
            }
            self._expect_error(tmp, meta=meta, match="unknown quant format")

    def test_reject_wrong_scale_recipe(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._expect_error(
                tmp,
                meta=_fp8_meta(_FP8, _BF16, recipe="tensorwise"),
                match="scale_recipe",
            )

    def test_reject_missing_weight_scale(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            t = _fp8_layer_tensors(_FP8)
            del t[f"{_FP8[0]}.weight_scale"]
            self._expect_error(tmp, tensors=t, match="weight_scale")

    def test_reject_wrong_scale_dtype(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            t = _fp8_layer_tensors(_FP8)
            t[f"{_FP8[0]}.weight_scale"] = ("BF16", [8])
            self._expect_error(tmp, tensors=t, match="dtype")

    def test_reject_wrong_scale_shape(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            t = _fp8_layer_tensors(_FP8)
            t[f"{_FP8[0]}.weight_scale"] = ("F32", [9])
            self._expect_error(tmp, tensors=t, match="shape")

    def test_reject_orphan_weight_scale(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            t = _fp8_layer_tensors(_FP8)
            t["txt_in.weight_scale"] = ("F32", [4])
            self._expect_error(tmp, tensors=t, match="unexpected")


class TestBflFp8rContract(unittest.TestCase):
    def _checkpoint(self, directory, *, excluded_override=None):
        import os

        fp8 = [f"blocks.{index}.linear" for index in range(234)]
        bf16 = [f"excluded.{index}" for index in range(37)]
        tensors = _fp8_layer_tensors(fp8, out=4, in_=4)
        tensors.update({f"{prefix}.weight": ("BF16", [4, 4]) for prefix in bf16})
        tensors.update({f"other.{index}.scale": ("BF16", [4]) for index in range(86)})
        weights = os.path.join(directory, "model.safetensors")
        _write_synthetic_safetensors(
            weights,
            tensors,
            metadata={"quantization": "fp8r", "scale": "per_row_fp32"},
        )
        sidecar = os.path.join(directory, "hf_quant_config.json")
        with open(sidecar, "w") as handle:
            json.dump(
                {
                    "producer": {"name": "bfl"},
                    "quantization": {
                        "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
                        "exclude_modules": (
                            bf16 if excluded_override is None else excluded_override
                        ),
                    },
                },
                handle,
            )
        return weights, sidecar, fp8, bf16

    def test_builds_rowwise_config_from_sidecar_and_tensor_headers(self):
        import tempfile

        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            build_flux3_image_fp8r_quant_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            weights, sidecar, fp8, bf16 = self._checkpoint(directory)
            config = build_flux3_image_fp8r_quant_config(weights, sidecar)

        self.assertIsInstance(config, RowwiseFp8Config)
        self.assertEqual(config.fp8_layers, frozenset(fp8))
        self.assertEqual(config.num_fp8_layers, 234)
        self.assertEqual(config.num_bf16_layers, 37)
        self.assertEqual(config.num_linear_layers, 271)
        self.assertEqual(config.num_tensors, 591)
        self.assertFalse(config.input_scale_layers)
        self.assertEqual(len(bf16), 37)

    def test_rejects_sidecar_exclusions_that_disagree_with_weights(self):
        import tempfile

        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            build_flux3_image_fp8r_quant_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            weights, sidecar, _, _ = self._checkpoint(directory, excluded_override=[])
            with self.assertRaisesRegex(ValueError, "exclude_modules disagrees"):
                build_flux3_image_fp8r_quant_config(weights, sidecar)


_LEAVES = ("to_qkv", "mlp_in", "attn_out", "mlp_out")


def _tiny_quant_config(input_scale_layers=(), fp8_layers=None):
    fp8 = (
        frozenset(f"single_blocks.0.{leaf}" for leaf in _LEAVES)
        if fp8_layers is None
        else frozenset(fp8_layers)
    )
    return RowwiseFp8Config(
        format_version="1.1",
        fp8_layers=fp8,
        input_scale_layers=frozenset(input_scale_layers),
        num_linear_layers=len(fp8),
        num_bf16_layers=0,
        num_tensors=len(fp8) * 2 + len(input_scale_layers),
    )


def _flux3_config(**kwargs):
    return Flux3RowwiseFp8Config.from_base_config(
        _tiny_quant_config(**kwargs), head_dim=128
    )


def _prims_ts_device(supported=True):
    return mock.patch.object(
        flux3_prims_ts, "flux3_prims_ts_available", return_value=supported
    )


def _tiny_native_model(quant_config, *, prims_ts_supported=True, tp_size=1):
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        flux3_image_dit_config,
    )
    from sglang.multimodal_gen.runtime.models.dits import flux3_image
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import Flux3ImageDiT

    # These dimensions satisfy the production PrimsTS kernel's K/N tiling
    # requirements at TP=1; TP=2 shards some projections below the tiles.
    config = flux3_image_dit_config(
        hidden_size=512,
        num_attention_heads=4,
        depth=0,
        depth_single_blocks=1,
        context_in_dim=64,
        axes_dim=(32, 32, 32, 32),
    )
    group = object()
    with (
        mock.patch.object(linear, "get_tp_group", return_value=group),
        mock.patch.object(linear, "get_group_size", return_value=tp_size),
        mock.patch.object(linear, "get_group_rank", return_value=0),
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


_ALL_ROLES = {
    "to_qkv": Flux3PrimsTsRole.QKV_QKNORM_ROPE,
    "mlp_in": Flux3PrimsTsRole.SWIGLU,
    "attn_out": Flux3PrimsTsRole.LINEAR,
    "mlp_out": Flux3PrimsTsRole.LINEAR,
}


class TestNativeQuantConstruction(unittest.TestCase):
    def test_plans_every_block_projection_on_supported_gpu(self):
        block = _tiny_native_model(_flux3_config()).single_blocks[0]
        self.assertEqual(_roles(block), _ALL_ROLES)

    def test_prims_ts_falls_back_on_unsupported_gpu(self):
        block = _tiny_native_model(
            _flux3_config(), prims_ts_supported=False
        ).single_blocks[0]

        self.assertEqual(set(_roles(block).values()), {None})
        for name in _LEAVES:
            layer = getattr(block, name)
            self.assertIsInstance(layer.quant_method, Flux3Fp8LinearMethod)
            storage = layer.weight.data_ptr()
            layer.quant_method.process_weights_after_loading(layer)
            # Generic [K, N] view of the untouched checkpoint storage.
            self.assertEqual(layer.weight.data_ptr(), storage)
            self.assertTrue(layer.weight.t().is_contiguous())

    def test_tp_local_shard_shapes_gate_prims_ts(self):
        """At TP=2 the QKV shard's N=768 no longer fits the 512-wide tiles."""
        block = _tiny_native_model(_flux3_config(), tp_size=2).single_blocks[0]
        self.assertEqual(block.to_qkv.output_size_per_partition, 768)
        self.assertEqual(_roles(block), {**_ALL_ROLES, "to_qkv": None})

    def test_bias_disables_prims_ts_at_load(self):
        layer = _tiny_native_model(_flux3_config()).single_blocks[0].mlp_in
        layer.bias = torch.nn.Parameter(torch.zeros(3072), requires_grad=False)
        source = layer.weight.detach().clone()
        layer.quant_method.process_weights_after_loading(layer)
        self.assertIsNone(layer.quant_method.prims_ts_role)
        self.assertTrue(
            torch.equal(layer.weight.t().view(torch.uint8), source.view(torch.uint8))
        )

    def test_native_model_exposes_generic_loader_hooks(self):
        model = _tiny_native_model(_flux3_config())
        processed = dict(model.preprocess_loaded_state_dict(model.state_dict().items()))

        self.assertEqual(set(processed), set(model.state_dict()))
        self.assertIsNone(model.post_load_weights())

    def test_flux3_method_keeps_the_v2_weight_loader(self):
        block = _tiny_native_model(_flux3_config()).single_blocks[0]
        for name in _LEAVES:
            layer = getattr(block, name)
            self.assertEqual(layer.weight.weight_loader, layer.weight_loader_v2)
            self.assertEqual(layer.weight_scale.weight_loader, layer.weight_loader_v2)

    def test_process_weights_prepares_prims_ts_weights(self):
        from sglang.kernels.ops.diffusion import pack_flux3_mlp_weight

        block = _tiny_native_model(_flux3_config()).single_blocks[0]
        sources = {}
        for name in _LEAVES:
            layer = getattr(block, name)
            layer.weight.data.copy_(
                torch.randn(layer.weight.shape).to(layer.weight.dtype)
            )
            layer.weight_scale.data.copy_(torch.rand(layer.weight_scale.shape))
            sources[name] = (
                layer.weight.data_ptr(),
                layer.weight.detach().clone(),
                layer.weight_scale.detach().clone(),
            )
            layer.quant_method.process_weights_after_loading(layer)

        # Every role hands PrimsTS the [N, K] storage itself; no second copy.
        for name in _LEAVES:
            layer = getattr(block, name)
            storage, weight_nk, scale = sources[name]
            if name == "mlp_in":
                weight_nk, scale = pack_flux3_mlp_weight(weight_nk.t(), scale)
            self.assertEqual(layer.weight.data_ptr(), storage)
            self.assertTrue(layer.weight.t().is_contiguous())
            self.assertTrue(
                torch.equal(
                    layer.weight.t().view(torch.uint8), weight_nk.view(torch.uint8)
                )
            )
            self.assertTrue(torch.equal(layer.weight_scale, scale))

    def test_swiglu_layout_rejects_generic_apply(self):
        layer = _tiny_native_model(_flux3_config()).single_blocks[0].mlp_in
        layer.quant_method.process_weights_after_loading(layer)
        with self.assertRaisesRegex(RuntimeError, "interleaved for the PrimsTS"):
            layer.quant_method.apply(layer, torch.zeros(1, 512, dtype=torch.bfloat16))

    def test_constructs_quantized_standard_parallel_linears(self):
        quant_config = _flux3_config()
        model = _tiny_native_model(quant_config)
        modules = dict(model.named_modules())
        for path in quant_config.fp8_layers:
            leaf = path.rsplit(".", 1)[-1]
            expected = {
                "to_qkv": MergedColumnParallelLinear,
                "mlp_in": MergedColumnParallelLinear,
                "attn_out": RowParallelLinear,
                "mlp_out": RowParallelLinear,
            }[leaf]
            self.assertIsInstance(modules[path], expected)
            self.assertEqual(modules[path].prefix, path)
            self.assertIsInstance(modules[path].quant_method, Flux3Fp8LinearMethod)

    def test_quantizes_native_modulation_linears(self):
        path = "early_stream_modulations.image.lin"
        model = _tiny_native_model(_flux3_config(fp8_layers=(path,)))
        layer = dict(model.named_modules())[path]

        self.assertIsInstance(layer, ReplicatedLinear)
        self.assertIsInstance(layer.quant_method, Flux3Fp8LinearMethod)

    def test_native_modulation_factory_keeps_bf16_unquantized(self):
        path = "early_stream_modulations.image.lin"
        model = _tiny_native_model(None)
        layer = dict(model.named_modules())[path]

        self.assertIsInstance(layer, ReplicatedLinear)
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_static_input_scale_stays_generic(self):
        path = "single_blocks.0.to_qkv"
        model = _tiny_native_model(_flux3_config(input_scale_layers=(path,)))
        module = dict(model.named_modules())[path]
        self.assertEqual(tuple(module.input_scale.shape), (1,))
        self.assertIn(f"{path}.input_scale", set(model.state_dict()))
        self.assertIsNone(module.quant_method.prims_ts_role)

    def test_metadata_declared_bf16_path_stays_unquantized(self):
        quant_config = _flux3_config(fp8_layers=("single_blocks.0.to_qkv",))
        block = _tiny_native_model(quant_config).single_blocks[0]
        self.assertIsInstance(block.to_qkv.quant_method, Flux3Fp8LinearMethod)
        self.assertIsInstance(block.mlp_in.quant_method, UnquantizedLinearMethod)

    def test_fused_mlp_weight_pack_reverses_each_gate_value_pair(self):
        from sglang.kernels.ops.diffusion import pack_flux3_mlp_weight

        # Stored live weight is [K, gate|value].
        weight = torch.tensor([[1, 2, 11, 12], [3, 4, 13, 14]])
        scale = torch.tensor([0.1, 0.2, 1.1, 1.2])
        packed_weight, packed_scale = pack_flux3_mlp_weight(weight, scale)
        self.assertTrue(
            torch.equal(
                packed_weight,
                torch.tensor([[11, 13], [1, 3], [12, 14], [2, 4]]),
            )
        )
        self.assertTrue(torch.equal(packed_scale, torch.tensor([1.1, 0.1, 1.2, 0.2])))

    def test_flux3_config_remaps_split_qkv_metadata_and_input_scales(self):
        base = _tiny_quant_config(fp8_layers=tuple(_FP8), input_scale_layers=_FP8[:3])
        packed = _tiny_quant_config(input_scale_layers=("single_blocks.0.to_qkv",))

        config = Flux3RowwiseFp8Config.from_base_config(base, head_dim=128)

        self.assertEqual(config.fp8_layers, packed.fp8_layers)
        self.assertEqual(config.input_scale_layers, packed.input_scale_layers)
        self.assertEqual(base.fp8_layers, frozenset(_FP8))
        qkv = _tiny_native_model(config).single_blocks[0].to_qkv
        self.assertIsInstance(qkv.quant_method, Flux3Fp8LinearMethod)
        self.assertEqual(tuple(qkv.input_scale.shape), (1,))

    def test_loader_selects_flux3_rowwise_fp8_config(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            select_flux3_image_quant_config,
        )

        dit_config = SimpleNamespace(
            arch_config=SimpleNamespace(hidden_size=3072, num_attention_heads=24)
        )
        base = _tiny_quant_config(fp8_layers=tuple(_FP8))
        selected = select_flux3_image_quant_config(base, dit_config)

        self.assertIsInstance(selected, Flux3RowwiseFp8Config)
        self.assertEqual(selected.head_dim, 128)
        self.assertIn("single_blocks.0.to_qkv", selected.fp8_layers)
        self.assertIs(select_flux3_image_quant_config(selected, dit_config), selected)
        fallback = select_flux3_image_quant_config(
            base, dit_config, enable_prims_ts=False
        )
        self.assertIsInstance(fallback, Flux3RowwiseFp8Config)
        self.assertFalse(fallback.enable_prims_ts)
        self.assertIsNone(select_flux3_image_quant_config(None, dit_config))


if __name__ == "__main__":
    unittest.main()
