# SPDX-License-Identifier: Apache-2.0
"""Tests for the FLUX.3 image DiT (``Flux3ImageDiT``).

The native DiT is the shared FLUX 3 transformer with native blocks; parity is
checked against :class:`Flux3Transformer` at toy size.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

_TINY = dict(
    hidden_size=256,
    num_attention_heads=2,
    depth=1,
    depth_single_blocks=2,
    context_in_dim=64,
)


def _tiny_native_dit(**overrides):
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        flux3_image_dit_config,
    )
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import Flux3ImageDiT

    torch.manual_seed(0)
    model = (
        Flux3ImageDiT(flux3_image_dit_config(**_TINY, **overrides)).cuda().bfloat16()
    )
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return model.eval()


def _sample(text_len: int, image_cond: bool):
    image = (torch.randn(4, 128).bfloat16().cuda(), torch.rand(4, 4).cuda() * 8)
    condition = (
        (torch.randn(4, 128).bfloat16().cuda(), torch.rand(4, 4).cuda() * 8)
        if image_cond
        else None
    )
    text = (
        torch.randn(text_len, 64).bfloat16().cuda(),
        torch.rand(text_len, 4).cuda() * 8,
    )
    return image, condition, text


def _pack(samples):
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT, pack

    return pack(
        [
            {"image": image, "image_cond": condition, TEXT: text}
            for image, condition, text in samples
        ],
        {"image": 128, "image_cond": 128, TEXT: 64},
    )


def test_image_dit_config_extends_the_shared_flux3_config():
    """The image DiT reuses the FLUX 3 arch config: image streams, one output
    head, no pooled vector, and the latent width of the first stream."""
    from sglang.multimodal_gen.configs.models.dits.flux3 import Flux3ArchConfig
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        flux3_image_dit_config,
    )

    arch = flux3_image_dit_config().arch_config
    assert isinstance(arch, Flux3ArchConfig)
    assert arch.sequence == {"x_image": "image", "x_image_cond": "image_cond"}
    assert arch.out_modes == ["image"]
    assert arch.vec_in_dim is None
    assert arch.num_channels_latents == 128
    # FLUX.3 Action keeps its defaults: every stream gets a head.
    assert Flux3ArchConfig().out_modes is None
    assert Flux3ArchConfig().num_channels_latents == 96


def test_registry_keeps_action_and_image_dits_apart():
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    native, _ = ModelRegistry.resolve_model_cls("Flux3ImageDiT")
    action, _ = ModelRegistry.resolve_model_cls("Flux3Transformer")
    assert native.__name__ == "Flux3ImageDiT"
    assert action.__name__ == "Flux3Transformer"
    assert issubclass(native, action)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA kernels")
def test_native_module_tree_matches_the_image_checkpoint(request):
    """Image checkpoints hold split q/k/v, a separate ``mlp_in`` and one
    ``final_layer.image`` head; the native tree must load them by name."""
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
        flux3_dit_layer_names,
    )

    request.getfixturevalue("single_process_model_parallel")
    model = _tiny_native_dit()
    params = set(model.state_dict())
    assert not any(".qkv_mlp." in name for name in params)
    assert not any(name.startswith("final_layer.image_cond") for name in params)
    assert model.vector_in is None
    mapping = get_param_names_mapping(model.param_names_mapping)
    for name in (
        "single_blocks.1.q_proj.weight",
        "content_mode_blocks.image_cond.0.v_proj.weight",
        "txt_mode_blocks.0.mlp_in.weight",
        "single_blocks.0.attn_out.weight",
        "early_stream_modulations.txt.lin.weight",
        "single_stream_modulations.image.lin.weight",
        "time_in.in_layer.weight",
        "emb_in.image_cond.weight",
        "txt_in.weight",
        "final_layer.image.adaLN_modulation.1.weight",
    ):
        target, _, _ = mapping(name)
        assert model._remap(target) in params, (name, target)
    names = flux3_dit_layer_names(model)
    modules = dict(model.named_modules())
    assert set(names) == {
        "single_blocks",
        "txt_mode_blocks",
        "content_mode_blocks.image",
        "content_mode_blocks.image_cond",
    }
    assert all(isinstance(modules[name], torch.nn.ModuleList) for name in names)


def _reference_weights(native) -> dict[str, torch.Tensor]:
    """The native weights renamed to the FLUX 3 transformer's module tree."""
    state = {}
    native_state = native.state_dict()
    for name, tensor in native_state.items():
        if ".to_qkv." in name:
            prefix = name.split(".to_qkv.")[0]
            mlp_in = native_state[f"{prefix}.mlp_in.weight"]
            state[f"{prefix}.qkv_mlp.weight"] = torch.cat([tensor, mlp_in])
        elif ".mlp_in." in name:
            continue
        else:
            state[name.replace("_norm.weight", "_norm.scale")] = tensor
    return state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA kernels")
def test_packed_cfg_forward_matches_flux3_transformer_per_sample(request):
    """Packing both CFG samples into one varlen sequence must not let them see
    each other: each sample's prediction equals the shared FLUX 3 transformer
    run on that sample alone (bf16 fused-kernel tolerance)."""
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        flux3_image_dit_config,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.models.dits.flux3 import Flux3Transformer

    request.getfixturevalue("single_process_model_parallel")
    native = _tiny_native_dit()
    reference = Flux3Transformer(flux3_image_dit_config(**_TINY)).cuda().bfloat16()
    reference.load_state_dict(_reference_weights(native), strict=True)

    samples = [_sample(3, image_cond=True), _sample(5, image_cond=True)]
    packed = _pack(samples)
    packed.timesteps["image"].fill_(0.7)
    with torch.inference_mode():
        predicted = native(packed)["image"].float()
        with set_forward_context(current_timestep=0, attn_metadata=None):
            expected = []
            for image, condition, text in samples:
                zero = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
                out = reference(
                    ctx=text[0][None],
                    ctx_ids=text[1][None],
                    timesteps_ctx=zero,
                    x_image=image[0][None],
                    x_image_ids=image[1][None],
                    x_image_timesteps=zero + 0.7,
                    x_image_cond=condition[0][None],
                    x_image_cond_ids=condition[1][None],
                    x_image_cond_timesteps=zero,
                )
                expected.append(out["x_image"][0].float())
    expected = torch.cat(expected)
    cosine = torch.nn.functional.cosine_similarity(
        predicted.flatten(), expected.flatten(), dim=0
    )
    assert cosine > 0.9995, cosine
    torch.testing.assert_close(predicted, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA kernels")
def test_packed_forward_skips_absent_streams(request):
    """t2i has no image_cond tokens: its mode blocks must not run, and only the
    configured output head is returned."""
    request.getfixturevalue("single_process_model_parallel")
    model = _tiny_native_dit()
    called = []
    for name, block in (
        ("image", model.content_mode_blocks["image"][0]),
        ("image_cond", model.content_mode_blocks["image_cond"][0]),
        ("text", model.txt_mode_blocks[0]),
        ("single", model.single_blocks[0]),
    ):
        block.register_forward_pre_hook(
            lambda _module, _inputs, value=name: called.append(value)
        )
    packed = _pack([_sample(3, image_cond=False), _sample(3, image_cond=False)])
    with torch.inference_mode():
        out = model(packed)
    assert set(out) == {"image"}
    assert out["image"].shape == (8, 128)
    assert called == ["image", "text", "single"]
    assert model._layerwise_active_content_mods == frozenset({"image"})


def test_flow_map_adds_each_samples_target_to_all_its_tokens(request):
    """As in the reference flow map: every token of a sample (text, condition
    and image) gets ``ref_time_in(emb(r))`` of that sample, r rounded to the
    timestep dtype; the phantom token of an absent mode gets none."""
    from sglang.multimodal_gen.runtime.models.dits.flux3 import timestep_embedding
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT

    request.getfixturevalue("single_process_model_parallel")
    model = _tiny_native_dit(flow_map=True)
    modes = ("image", "image_cond", TEXT)

    def added(packed):
        vec = {
            mode: torch.zeros(
                max(packed.data[mode].shape[0], 1), 256, device="cuda"
            ).bfloat16()
            for mode in modes
        }
        with torch.inference_mode():
            model._add_reference_time(packed, modes, vec)
        return vec

    packed = _pack([_sample(3, image_cond=True), _sample(5, image_cond=False)])
    packed.timesteps_ref = torch.tensor([0.5, 0.3], device="cuda")
    with torch.inference_mode():
        target = model.ref_time_in(
            timestep_embedding(packed.timesteps_ref.bfloat16()).bfloat16()
        )
    vec = added(packed)
    for mode in modes:
        for sample, (start, end) in enumerate(packed.mode_indices[mode]):
            torch.testing.assert_close(
                vec[mode][start:end], target[sample].expand(end - start, -1)
            )
    assert packed.mode_indices["image_cond"][1] == (4, 4)

    t2i = _pack([_sample(3, image_cond=False)])
    t2i.timesteps_ref = torch.tensor([0.5], device="cuda")
    assert added(t2i)["image_cond"].abs().sum() == 0

    t2i.timesteps_ref = None
    with pytest.raises(ValueError, match="flow-map"):
        added(t2i)
    plain = _tiny_native_dit()
    t2i.timesteps_ref = torch.tensor([0.5], device="cuda")
    with pytest.raises(ValueError, match="flow-map"):
        plain._add_reference_time(t2i, modes, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA kernels")
def test_packed_forward_compiles_across_text_lengths(request):
    """The DiT compiles as one module (no regional compile conditions); the
    text bucket length changes between requests and must not break it."""
    request.getfixturevalue("single_process_model_parallel")
    model = _tiny_native_dit()
    compiled = torch.compile(model, backend="eager", dynamic=True)
    for text_len in (3, 5):
        torch.manual_seed(text_len)
        samples = [_sample(text_len, image_cond=False)] * 2
        with torch.inference_mode():
            expected = model(_pack(samples))["image"]
            actual = compiled(_pack(samples))["image"]
        torch.testing.assert_close(actual, expected)


def test_joint_layout_round_trips_in_sample_major_order():
    """The joint sequence is sample-major with modes in sorted order; unpacking
    must put every row back where the mode-major pack had it."""
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
        TEXT,
        _JointLayout,
        pack,
    )

    def rows(start, count):
        values = torch.arange(start, start + count, dtype=torch.float32)
        return values[:, None].expand(-1, 2).bfloat16(), torch.zeros(count, 4)

    packed = pack(
        [
            {"image": rows(0, 2), "image_cond": None, TEXT: rows(10, 1)},
            {"image": rows(2, 2), "image_cond": rows(20, 3), TEXT: rows(11, 2)},
        ],
        {"image": 2, "image_cond": 2, TEXT: 2},
    )
    layout = _JointLayout(packed, ("image", "image_cond", TEXT))
    joint = layout.pack(packed.data)
    assert joint[:, 0].tolist() == [0, 1, 10, 2, 3, 20, 21, 22, 11, 12]
    assert layout.sample_bounds == [(0, 3), (3, 10)]
    for mode, data in packed.data.items():
        assert torch.equal(layout.unpack(joint, mode, like=data), data)


def test_content_stack_prefetch_follows_active_mods():
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
        flux3_content_stack_is_active,
    )

    active = frozenset({"image"})
    assert flux3_content_stack_is_active("content_mode_blocks.image", active)
    assert not flux3_content_stack_is_active("content_mode_blocks.image_cond", active)
    assert flux3_content_stack_is_active("single_blocks", active)
    assert flux3_content_stack_is_active("txt_mode_blocks", active)
    assert flux3_content_stack_is_active("content_mode_blocks.image", None)


def test_cross_group_prefetch_target():
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
        flux3_cross_group_prefetch_target as nxt,
    )

    mods = ["image", "image_cond"]
    assert nxt("content_mode_blocks.image", 0, mods, 5, 28) == (
        "content_mode_blocks.image_cond",
        0,
    )
    assert nxt("content_mode_blocks.image_cond", 2, mods, 5, 28) == (
        "txt_mode_blocks",
        2,
    )
    assert nxt("txt_mode_blocks", 4, mods, 5, 28) == ("single_blocks", 0)
    assert nxt("single_blocks", 0, mods, 5, 28) is None


class TestFlux3ImageDiTContract(unittest.TestCase):
    def test_packed_qkv_checkpoint_mapping(self):
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
        from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
            Flux3ImageDiT,
        )

        mapping = get_param_names_mapping(Flux3ImageDiT.param_names_mapping)
        for src, shard in (("q_proj", 0), ("k_proj", 1), ("v_proj", 2)):
            name, idx, total = mapping(f"single_blocks.0.{src}.weight")
            self.assertEqual(name, "single_blocks.0.to_qkv.weight")
            self.assertEqual((idx, total), (shard, 3))
        for src, shard in (("q_proj", 0), ("k_proj", 1), ("v_proj", 2)):
            name, idx, total = mapping(f"single_blocks.0.{src}.input_scale")
            self.assertEqual(name, "single_blocks.0.to_qkv.input_scale")
            self.assertEqual((idx, total), (shard, 3))

    def test_checkpoint_preprocessor_ignores_only_known_training_extras(self):
        import torch

        from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
            Flux3ImageDiT,
        )

        class LoaderHarness:
            def __init__(self):
                self._ignored_checkpoint_prefixes = (
                    Flux3ImageDiT._ignored_checkpoint_prefixes
                )
                self._param_renames = Flux3ImageDiT._param_renames

            _remap = Flux3ImageDiT._remap
            preprocess_loaded_state_dict = Flux3ImageDiT.preprocess_loaded_state_dict

        model = LoaderHarness()
        weights = list(
            model.preprocess_loaded_state_dict(
                [
                    ("weight", torch.ones(2)),
                    ("single_blocks.0.query_norm.scale", torch.ones(2)),
                    ("repa_projectors.block.weight", torch.ones(1)),
                    ("repa_projectors_extra.block.weight", torch.ones(1)),
                    ("unsupported.weight", torch.ones(1)),
                ]
            )
        )
        self.assertEqual(
            [name for name, _ in weights],
            [
                "weight",
                "single_blocks.0.query_norm.weight",
                "repa_projectors_extra.block.weight",
                "unsupported.weight",
            ],
        )

    def test_flat_checkpoint_loader_delegates_to_generic_lifecycle(self):
        import tempfile
        from unittest import mock

        import torch

        from sglang.multimodal_gen.runtime.loader import fsdp_load
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            load_flux3_image_dit,
        )

        expected = object()
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            with mock.patch.object(
                fsdp_load, "maybe_load_fsdp_model", return_value=expected
            ) as load:
                result = load_flux3_image_dit(
                    checkpoint.name,
                    config=object(),
                    device=torch.device("cpu"),
                    quant_config=None,
                )

        self.assertIs(result, expected)
        self.assertEqual(load.call_args.kwargs["weight_dir_list"], [checkpoint.name])
        self.assertTrue(load.call_args.kwargs["strict"])
        self.assertFalse(load.call_args.kwargs["fsdp_inference"])


class TestFlux3LossyAccelPlumbing(unittest.TestCase):
    def test_kitchen_int8_is_registered(self):
        from sglang.multimodal_gen.runtime.layers.quantization import (
            QUANTIZATION_METHODS,
            get_quantization_config,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
            ConvRotInt8Config,
        )

        # ``kitchen_int8`` is the deprecated alias of ``convrot_int8``.
        self.assertIn("convrot_int8", QUANTIZATION_METHODS)
        self.assertIs(get_quantization_config("kitchen_int8"), ConvRotInt8Config)

    def test_kitchen_int8_skips_non_divisible_input(self):
        from sglang.multimodal_gen.runtime.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
            ConvRotInt8Config,
        )

        layer = object.__new__(LinearBase)
        layer.input_size = 2688  # 2688 % 256 != 0
        layer.output_size = 3072
        cfg = ConvRotInt8Config()
        method = cfg.get_quant_method(layer, "adaln.linear")
        self.assertIsInstance(method, UnquantizedLinearMethod)
        self.assertTrue(any("adaln.linear" in s for s in cfg.skipped))

    def test_sol_attn_parses_flux3_layer_prefixes(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn import (
            SolAttnImpl,
        )

        parse = SolAttnImpl._parse_layer_idx
        # Official parser is ``blocks.(\d+)``. FLUX.3 names that contain
        # that substring still resolve; dotted modality keys do not.
        self.assertEqual(parse("single_blocks.5.attention.impl"), 5)
        self.assertEqual(parse("txt_mode_blocks.2.attention"), 2)
        self.assertIsNone(parse("content_mode_blocks.image.3.attention"))
        self.assertIsNone(parse("unrelated.linear"))

    def test_dit_step_sets_forward_timestep(self):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            _dit_step,
        )

        with _dit_step(7):
            self.assertEqual(get_forward_context().current_timestep, 7)

    def test_resolve_quant_config_from_metadata_or_cli(self):
        from unittest.mock import patch

        from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
            ConvRotInt8Config,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
            resolve_flux3_image_dit_quant_config,
        )

        ckpt = Fp8Config(is_checkpoint_fp8_serialized=True)
        with patch(
            "sglang.multimodal_gen.runtime.utils.quantization_utils."
            "get_quant_config_from_safetensors_metadata",
            return_value=ckpt,
        ):
            self.assertIs(resolve_flux3_image_dit_quant_config("dit.safetensors"), ckpt)
            with self.assertRaisesRegex(ValueError, "do not also set --quantization"):
                resolve_flux3_image_dit_quant_config("dit.safetensors", "kitchen_int8")

        with patch(
            "sglang.multimodal_gen.runtime.utils.quantization_utils."
            "get_quant_config_from_safetensors_metadata",
            return_value=None,
        ):
            config = resolve_flux3_image_dit_quant_config(
                "dit.safetensors",
                "kitchen_int8",
                ignored_layers=["adaln.linear"],
            )
        self.assertIsInstance(config, ConvRotInt8Config)
        self.assertEqual(config.ignored_layers, ["adaln.linear"])


class _StubLinear:
    """A FLUX.3 linear whose quant method may have planned a PrimsTS role."""

    def __init__(self, role=None, output="native", **method_attrs):
        self.quant_method = SimpleNamespace(
            prims_ts_role=role,
            activation_format="fp8",
            packed_activation_format=None,
        )
        for name, value in method_attrs.items():
            setattr(self.quant_method, name, value)
        self.output = output
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return self.output, None


class TestFlux3FusedOps(unittest.TestCase):
    _CUDA_X = SimpleNamespace(is_cuda=True)

    @staticmethod
    def _platform(major, minor=0):
        from sglang.kernels import fused_op
        from sglang.kernels.spec import PlatformInfo

        return mock.patch.object(
            fused_op,
            "_platform",
            return_value=PlatformInfo(
                device_type="cuda", cuda_arch_major=major, cuda_arch_minor=minor
            ),
        )

    def test_capability_metadata_allows_sm10x_and_rejects_sm120(self):
        from sglang.kernels.spec import KernelBackend, PlatformInfo
        from sglang.multimodal_gen.runtime.layers import flux3_fused_ops
        from sglang.multimodal_gen.runtime.layers.quantization import flux3_prims_ts

        for arch, expected in (
            ((10, 0), True),
            ((10, 3), True),
            ((10, 7), True),
            ((9, 0), False),
            ((11, 0), False),
            ((12, 0), False),
        ):
            platform = PlatformInfo(
                device_type="cuda", cuda_arch_major=arch[0], cuda_arch_minor=arch[1]
            )
            with self.subTest(arch=arch):
                for op in (
                    flux3_fused_ops._FLUX3_LINEAR,
                    flux3_fused_ops._FLUX3_QKV_QKNORM_ROPE,
                    flux3_fused_ops._FLUX3_GATED_MLP,
                ):
                    with self._platform(*arch):
                        self.assertEqual(
                            op.auto_selected_backend() is KernelBackend.FLASHINFER,
                            expected,
                        )
                with mock.patch.object(PlatformInfo, "detect", return_value=platform):
                    self.assertEqual(
                        flux3_prims_ts.flux3_prims_ts_available(), expected
                    )

    def test_forced_non_flashinfer_backend_disables_prims_ts_planning(self):
        from sglang.kernels import fused_op
        from sglang.kernels.spec import KernelBackend, PlatformInfo
        from sglang.multimodal_gen.runtime.layers.quantization import flux3_prims_ts

        sm100 = PlatformInfo(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0)
        with mock.patch.object(PlatformInfo, "detect", return_value=sm100):
            for forced, expected in (
                (None, True),
                (KernelBackend.FLASHINFER, True),
                (KernelBackend.TORCH, False),
            ):
                with mock.patch.object(fused_op, "_forced_backend", forced):
                    self.assertEqual(
                        flux3_prims_ts.flux3_prims_ts_available(), expected
                    )

    def test_linear_runs_prims_ts_only_for_planned_layers_on_sm10x(self):
        from sglang.kernels import fused_op
        from sglang.kernels.spec import KernelBackend
        from sglang.multimodal_gen.runtime.layers.flux3_fused_ops import flux3_linear
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsRole,
        )

        x = self._CUDA_X
        prims_ts = mock.Mock(return_value="prims_ts")
        planned = _StubLinear(Flux3PrimsTsRole.LINEAR, apply_prims_ts_linear=prims_ts)
        with self._platform(10, 0):
            self.assertEqual(flux3_linear(planned, x, "shared"), "prims_ts")
            self.assertEqual(flux3_linear(_StubLinear(), x), "native")
            # A different planned role does not satisfy a plain projection.
            swiglu = _StubLinear(Flux3PrimsTsRole.SWIGLU)
            self.assertEqual(flux3_linear(swiglu, x), "native")
            with mock.patch.object(fused_op, "_forced_backend", KernelBackend.TORCH):
                self.assertEqual(flux3_linear(planned, x), "native")
        with self._platform(12, 0):
            self.assertEqual(flux3_linear(planned, x), "native")
        prims_ts.assert_called_once_with(planned, x, "shared")
        self.assertEqual(planned.inputs, [x, x])

    def test_qkv_native_runs_linear_then_strided_qknorm_rope(self):
        from sglang.multimodal_gen.runtime.layers import flux3_fused_ops

        heads, head_dim = 4, 8
        qkv = torch.randn(2, 16, 3 * heads * head_dim)
        layer = _StubLinear(output=qkv)
        norm = SimpleNamespace(query_norm=object(), key_norm=object())
        cos_sin, positions = torch.zeros(32, head_dim), torch.arange(32)
        seen = {}

        def _out_of_place(q, k, query_norm, key_norm, size, cache, **kwargs):
            seen.update(kwargs, size=size, cache=cache)
            self.assertEqual(q.shape, (2, 16, heads, head_dim))
            self.assertEqual(q.stride(-1), 1)
            return q + 1, k + 2

        with mock.patch.object(
            flux3_fused_ops, "apply_qk_norm_rope", side_effect=_out_of_place
        ):
            q, k, v = flux3_fused_ops.flux3_qkv_qknorm_rope(
                layer, qkv, norm, cos_sin, positions, heads
            )

        expected_q, expected_k, expected_v = qkv.view(2, 16, 3, heads, head_dim).unbind(
            2
        )
        self.assertTrue(torch.equal(q, expected_q + 1))
        self.assertTrue(torch.equal(k, expected_k + 2))
        self.assertTrue(torch.equal(v, expected_v))
        self.assertTrue(seen["allow_strided_qk"])
        self.assertFalse(seen["is_neox"])
        self.assertIs(seen["positions"], positions)
        self.assertIs(seen["cache"], cos_sin)
        self.assertEqual(seen["size"], head_dim)

    def test_qkv_prims_ts_epilogue_receives_norm_weights(self):
        from sglang.multimodal_gen.runtime.layers.flux3_fused_ops import (
            flux3_qkv_qknorm_rope,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsRole,
        )

        heads, head_dim = 2, 4
        qkv = torch.randn(1, 3, 3 * heads * head_dim)
        epilogue = mock.Mock(return_value=qkv)
        layer = _StubLinear(
            Flux3PrimsTsRole.QKV_QKNORM_ROPE, apply_prims_ts_qkv_qknorm_rope=epilogue
        )
        norm = SimpleNamespace(
            query_norm=SimpleNamespace(weight="q_weight"),
            key_norm=SimpleNamespace(weight="k_weight"),
        )
        with self._platform(10, 7):
            q, k, v = flux3_qkv_qknorm_rope(
                layer, self._CUDA_X, norm, "cos_sin", "positions", heads, "shared"
            )

        epilogue.assert_called_once_with(
            layer,
            self._CUDA_X,
            "q_weight",
            "k_weight",
            "cos_sin",
            "positions",
            "shared",
        )
        self.assertEqual(q.shape, (1, 3, heads, head_dim))
        self.assertEqual(layer.inputs, [])

    def test_gated_mlp_hands_packed_nvfp4_to_mlp_down(self):
        from sglang.multimodal_gen.runtime.layers.flux3_fused_ops import (
            flux3_gated_mlp,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsRole,
        )

        swiglu = mock.Mock(return_value="packed")
        from_packed = mock.Mock(return_value="down")
        mlp_in = _StubLinear(
            Flux3PrimsTsRole.SWIGLU,
            apply_prims_ts_swiglu=swiglu,
            packed_activation_format="nvfp4",
        )
        mlp_out = _StubLinear(
            Flux3PrimsTsRole.LINEAR,
            apply_prims_ts_linear_from_packed=from_packed,
            packed_activation_format="nvfp4",
        )
        with self._platform(10, 0):
            result = flux3_gated_mlp(mlp_in, mlp_out, self._CUDA_X, "shared")

        self.assertEqual(result, "down")
        swiglu.assert_called_once_with(
            mlp_in, self._CUDA_X, "shared", packed_output=True
        )
        from_packed.assert_called_once_with(mlp_out, "packed")

    def test_gated_mlp_bf16_handoff_and_native_fallback(self):
        from sglang.multimodal_gen.runtime.layers import flux3_fused_ops
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsRole,
        )

        hidden = SimpleNamespace(is_cuda=True)
        activated = SimpleNamespace(is_cuda=True)
        swiglu = mock.Mock(return_value=hidden)
        down = mock.Mock(return_value="down")
        mlp_in = _StubLinear(
            Flux3PrimsTsRole.SWIGLU, output="up", apply_prims_ts_swiglu=swiglu
        )
        mlp_out = _StubLinear(Flux3PrimsTsRole.LINEAR, apply_prims_ts_linear=down)
        with self._platform(10, 0):
            result = flux3_fused_ops.flux3_gated_mlp(mlp_in, mlp_out, self._CUDA_X)
        self.assertEqual(result, "down")
        swiglu.assert_called_once_with(mlp_in, self._CUDA_X, None)
        down.assert_called_once_with(mlp_out, hidden, None)

        with (
            self._platform(12, 0),
            mock.patch.object(
                flux3_fused_ops, "_silu_and_mul", return_value=activated
            ) as activation,
        ):
            result = flux3_fused_ops.flux3_gated_mlp(mlp_in, mlp_out, self._CUDA_X)
        activation.assert_called_once_with("up")
        self.assertEqual(result, "native")
        self.assertEqual(mlp_out.inputs, [activated])

    def test_shared_activation_requires_matching_prims_ts_consumers(self):
        from sglang.multimodal_gen.runtime.layers.flux3_fused_ops import (
            flux3_shared_activation,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
            Flux3PrimsTsRole,
        )

        x = self._CUDA_X
        quantize = mock.Mock(return_value="shared")
        mlp_in = _StubLinear(
            Flux3PrimsTsRole.SWIGLU, quantize_prims_ts_activation=quantize
        )
        qkv = _StubLinear(Flux3PrimsTsRole.QKV_QKNORM_ROPE)
        plain_qkv = _StubLinear(Flux3PrimsTsRole.LINEAR)
        nvfp4_qkv = _StubLinear(Flux3PrimsTsRole.LINEAR, activation_format="nvfp4")
        self.assertEqual(flux3_shared_activation(x, qkv, mlp_in), "shared")
        self.assertEqual(flux3_shared_activation(x, plain_qkv, mlp_in), "shared")
        self.assertIsNone(flux3_shared_activation(x, _StubLinear(), mlp_in))
        self.assertIsNone(flux3_shared_activation(x, qkv, _StubLinear()))
        self.assertIsNone(flux3_shared_activation(x, nvfp4_qkv, mlp_in))
        # Off-device inputs never take the PrimsTS operand path.
        cpu_x = SimpleNamespace(is_cuda=False)
        self.assertIsNone(flux3_shared_activation(cpu_x, qkv, mlp_in))
        self.assertEqual(quantize.call_args_list, [mock.call(mlp_in, x)] * 2)


if __name__ == "__main__":
    unittest.main()
