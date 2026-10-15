# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for FLUX.3 LoRA support (served ``LoRAPipeline`` path).

Covers the three things the served ``--lora-path`` route depends on:

* ``Flux3ImagePipeline`` inherits the framework's ``LoRAPipeline`` mixin.
* The arch config's ``dit.``-prefix strip mapping is what the framework actually
  resolves, and the exact key normalization ``load_lora_adapter`` performs turns
  both adapter families' keys into module paths shaped like the native DiT's
  ``named_modules()``.
* The gate/value-aware TP slice for the gated ``mlp_in`` up-projection.

Real-adapter checks are opt-in via ``F3_LORA_OSTRIS_CKPT`` /``F3_LORA_BFL_CKPT``
and skip when unset. Numeric parity on real weights needs a GPU and lives
outside this file.
"""

import os
import re
import unittest

import torch

# module path (sans ``.lora_A``/``.lora_B``) of every layer both adapters target.
_TARGET_RE = re.compile(
    r"^(single_blocks|txt_mode_blocks|"
    r"content_mode_blocks\.(?:image|image_cond))"
    r"\.\d+\.(?:q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out)$"
)

# Real-adapter checks are opt-in: point these at a local .safetensors to enable
# them. Unset (the default) leaves them empty, so the isfile() gates below skip.
_OSTRIS_CKPT = os.environ.get("F3_LORA_OSTRIS_CKPT", "")
_BFL_CKPT = os.environ.get("F3_LORA_BFL_CKPT", "")

_DIT_PREFIX_MAPPING = {r"^dit\.": "", r"^transformer\.": ""}


def _sample_keys(prefix: str) -> list[str]:
    """A handful of representative adapter keys for the given top-level prefix."""
    subs = [
        "single_blocks.0.q_proj",
        "single_blocks.27.mlp_in",
        "txt_mode_blocks.4.attn_out",
        "content_mode_blocks.image_cond.4.v_proj",
    ]
    keys = []
    for s in subs:
        keys.append(f"{prefix}{s}.lora_A.weight")
        keys.append(f"{prefix}{s}.lora_B.weight")
    return keys


def _normalize_like_framework(key: str, lora_map_fn) -> str:
    """Replicate ``LoRAPipeline.load_lora_adapter``'s key transform.

    Mirrors ``lora_pipeline.py`` (``diffusion_model.`` strip, ``.weight`` strip,
    then the model's ``lora_param_names_mapping``).
    """
    name = key.replace("diffusion_model.", "").replace(".weight", "")
    name, _, _ = lora_map_fn(name)
    return name


def _partition_gated(weight: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    """A rank's rows of a fused ``[gate; value]`` weight, as a merged column-parallel
    linear shards it: each half is split across ranks independently."""
    return torch.cat([half.chunk(tp_size)[tp_rank] for half in weight.chunk(2)])


class TestFlux3LoraServedWiring(unittest.TestCase):
    """The mixin + config plumbing the served ``--lora-path`` path relies on."""

    def test_pipeline_inherits_lora_mixin(self):
        from sglang.multimodal_gen.runtime.pipelines.flux3_image import (
            Flux3ImagePipeline,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
            LoRAPipeline,
        )

        self.assertTrue(issubclass(Flux3ImagePipeline, LoRAPipeline))

    def test_arch_config_mapping_is_the_only_lora_mapping(self):
        """The framework resolves ``config.lora_param_names_mapping or
        transformer.lora_param_names_mapping``. The native DiT inherits the
        empty base mapping, so the image arch config must carry the prefix
        strip; emptying it silently drops every adapter key prefix.
        """
        from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
            flux3_image_dit_config,
        )
        from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
            Flux3ImageDiT,
        )

        config_mapping = flux3_image_dit_config().arch_config.lora_param_names_mapping
        self.assertEqual(config_mapping, _DIT_PREFIX_MAPPING)
        self.assertEqual(Flux3ImageDiT.lora_param_names_mapping, {})

    def test_key_normalization_matches_module_paths(self):
        """Assert every normalized key is ``<block-target-module>.lora_A/B`` --
        what ``_apply_lora_to_layers`` looks up against the wrapped
        ``named_modules()``."""
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

        lora_map_fn = get_param_names_mapping(_DIT_PREFIX_MAPPING)

        # BFL/ostris ``dit.``, ComfyUI ``diffusion_model.``, diffusers/FastGen ``transformer.``
        for prefix in ("dit.", "diffusion_model.", "transformer."):
            for key in _sample_keys(prefix):
                name = _normalize_like_framework(key, lora_map_fn)
                self.assertTrue(
                    name.endswith(".lora_A") or name.endswith(".lora_B"), name
                )
                module = name.rsplit(".lora_", 1)[0]
                self.assertRegex(module, _TARGET_RE)


class TestGatedMergedColumnLoraSlice(unittest.TestCase):
    """The gate/value-aware TP slice for the gated ``mlp_in`` up-projection.

    FLUX.3's ``mlp_in`` is a ``MergedColumnParallelLinear`` whose weight is
    ``[gate; value]`` stacked along the output dim, and both adapter families
    store its ``lora_B`` fused the same way (``[2 * mlp_hidden, r]``). The slice
    must therefore shard each half independently -- a single contiguous slice
    would pair one rank's gate rows with another rank's value rows.

    GPU-free: the slice only reads ``output_sizes`` / ``output_partition_sizes``
    off the base layer, so a stub stands in for the real parallel linear (which
    would need an initialized model-parallel group).
    """

    HIDDEN, MLP_HIDDEN, RANK = 64, 96, 8

    def _layer(self, tp_size: int):
        from types import SimpleNamespace

        from sglang.multimodal_gen.runtime.layers.lora.linear import (
            MergedColumnParallelLinearWithLoRA,
        )

        layer = object.__new__(MergedColumnParallelLinearWithLoRA)
        shard = self.MLP_HIDDEN // tp_size
        layer.base_layer = SimpleNamespace(
            output_sizes=[self.MLP_HIDDEN, self.MLP_HIDDEN],
            output_partition_sizes=[shard, shard],
        )
        return layer

    def _slice(self, layer, B: torch.Tensor, tp_rank: int) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.layers.lora import linear as lora_linear

        original = lora_linear.get_tp_rank
        lora_linear.get_tp_rank = lambda: tp_rank
        try:
            return layer.slice_lora_b_weights(B)
        finally:
            lora_linear.get_tp_rank = original

    def test_merged_shard_matches_sharding_the_full_merged_weight(self):
        """Merging the per-rank LoRA slice into the per-rank base shard must equal
        sharding the fully merged weight -- checked against the reference gated
        partition (``_partition_gated``)."""
        torch.manual_seed(0)
        # fp64 so an exact comparison isolates layout errors from rounding.
        W = torch.randn(2 * self.MLP_HIDDEN, self.HIDDEN, dtype=torch.float64)
        A = torch.randn(self.RANK, self.HIDDEN, dtype=torch.float64)
        B = torch.randn(2 * self.MLP_HIDDEN, self.RANK, dtype=torch.float64)
        merged_full = W + B @ A

        for tp_size in (1, 2, 3, 4):
            layer = self._layer(tp_size)
            for tp_rank in range(tp_size):
                B_local = self._slice(layer, B, tp_rank)
                self.assertEqual(
                    B_local.shape, (2 * (self.MLP_HIDDEN // tp_size), self.RANK)
                )
                got = _partition_gated(W, tp_size, tp_rank) + B_local @ A
                want = _partition_gated(merged_full, tp_size, tp_rank)
                self.assertTrue(
                    torch.equal(got, want), f"tp_size={tp_size} rank={tp_rank}"
                )

    def test_differs_from_contiguous_slice_when_sharded(self):
        """Guards the actual bug: at tp=1 a contiguous slice happens to be right,
        so only tp>1 distinguishes the two."""
        torch.manual_seed(0)
        B = torch.randn(2 * self.MLP_HIDDEN, self.RANK)
        for tp_size in (2, 4):
            layer = self._layer(tp_size)
            width = 2 * (self.MLP_HIDDEN // tp_size)
            for tp_rank in range(tp_size):
                B_local = self._slice(layer, B, tp_rank)
                contiguous = B[tp_rank * width : (tp_rank + 1) * width]
                if tp_rank == 0 and tp_size == 2:
                    continue  # rank 0's gate half coincides; the value half does not
                self.assertFalse(
                    torch.equal(B_local, contiguous), f"tp={tp_size} rank={tp_rank}"
                )

    def test_ranks_reassemble_the_full_up_projection(self):
        torch.manual_seed(0)
        B = torch.randn(2 * self.MLP_HIDDEN, self.RANK)
        for tp_size in (2, 4):
            layer = self._layer(tp_size)
            shard = self.MLP_HIDDEN // tp_size
            gates, values = [], []
            for tp_rank in range(tp_size):
                B_local = self._slice(layer, B, tp_rank)
                gates.append(B_local[:shard])
                values.append(B_local[shard:])
            self.assertTrue(torch.equal(torch.cat(gates + values, dim=0), B))

    def test_stacked_layout_still_supported(self):
        """Adapters that store ``B`` stacked per logical shard keep the old path."""
        torch.manual_seed(0)
        tp_size, tp_rank = 2, 1
        layer = self._layer(tp_size)
        shard = self.MLP_HIDDEN // tp_size
        B = torch.randn(2, self.MLP_HIDDEN, self.RANK)
        B_local = self._slice(layer, B, tp_rank)
        self.assertTrue(torch.equal(B_local, B[:, tp_rank * shard :, :]))

    def test_mis_sized_fused_b_is_rejected(self):
        layer = self._layer(tp_size=1)
        with self.assertRaises(ValueError):
            self._slice(layer, torch.zeros(2 * self.MLP_HIDDEN + 1, self.RANK), 0)


class TestFlux3LoraFormatDetection(unittest.TestCase):
    def test_both_families_detect_standard(self):
        from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
            LoRAFormat,
            detect_lora_format_from_state_dict,
        )

        # BFL/ostris ``dit.``, ComfyUI ``diffusion_model.``, diffusers/FastGen ``transformer.``
        for prefix in ("dit.", "diffusion_model.", "transformer."):
            sd = {k: torch.zeros(2, 2) for k in _sample_keys(prefix)}
            self.assertEqual(
                detect_lora_format_from_state_dict(sd),
                LoRAFormat.STANDARD,
                f"prefix {prefix!r} should detect as STANDARD",
            )


class TestFlux3LoraRealAdapters(unittest.TestCase):
    """On the real adapters: every module maps to a block target (0 stragglers)."""

    def _check(self, path: str, expected_modules: int):
        from safetensors import safe_open

        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

        lora_map_fn = get_param_names_mapping(_DIT_PREFIX_MAPPING)
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())

        modules = set()
        for key in keys:
            name = _normalize_like_framework(key, lora_map_fn)
            self.assertTrue(
                name.endswith(".lora_A") or name.endswith(".lora_B"),
                f"unexpected adapter key {key}",
            )
            module = name.rsplit(".lora_", 1)[0]
            self.assertRegex(module, _TARGET_RE)
            modules.add(module)
        self.assertEqual(len(modules), expected_modules)

    def _check_gated_mlp_in_tp_slice(self, path: str):
        """The gated slice's premise, on the real adapter at production shapes.

        Both families store ``mlp_in``'s up-projection fused as
        ``[2 * mlp_hidden, r]`` (the base weight's ``[gate; value]`` layout), so
        the per-rank slice must split it in halves. Checked in bf16 at the real
        3072/9216 sizes -- the fp64 algebra lives in
        ``TestGatedMergedColumnLoraSlice``.
        """
        from types import SimpleNamespace

        from safetensors.torch import load_file

        from sglang.multimodal_gen.runtime.layers.lora import linear as lora_linear

        state = load_file(path)
        key = next(k for k in state if k.endswith("mlp_in.lora_B.weight"))
        B = state[key]
        A = state[key.replace(".lora_B.", ".lora_A.")]
        mlp_hidden, rank = B.shape[0] // 2, B.shape[1]
        self.assertEqual(B.shape[0], 2 * mlp_hidden, "mlp_in lora_B must be fused")
        self.assertEqual(A.shape, (rank, 3072))
        self.assertEqual(mlp_hidden, 9216)

        base = torch.zeros(2 * mlp_hidden, A.shape[1], dtype=B.dtype)
        merged_full = base + (B @ A)

        layer = object.__new__(lora_linear.MergedColumnParallelLinearWithLoRA)
        original = lora_linear.get_tp_rank
        try:
            for tp_size in (1, 2, 4):
                shard = mlp_hidden // tp_size
                layer.base_layer = SimpleNamespace(
                    output_sizes=[mlp_hidden, mlp_hidden],
                    output_partition_sizes=[shard, shard],
                )
                for tp_rank in range(tp_size):
                    lora_linear.get_tp_rank = lambda r=tp_rank: r
                    B_local = layer.slice_lora_b_weights(B)
                    self.assertEqual(B_local.shape, (2 * shard, rank))
                    got = _partition_gated(base, tp_size, tp_rank) + B_local @ A
                    want = _partition_gated(merged_full, tp_size, tp_rank)
                    self.assertTrue(
                        torch.equal(got, want), f"tp_size={tp_size} rank={tp_rank}"
                    )
        finally:
            lora_linear.get_tp_rank = original

    def test_ostris(self):
        if not os.path.isfile(_OSTRIS_CKPT):
            self.skipTest("Ostris adapter absent (set F3_LORA_OSTRIS_CKPT)")
        self._check(_OSTRIS_CKPT, expected_modules=516 // 2)
        self._check_gated_mlp_in_tp_slice(_OSTRIS_CKPT)

    def test_bfl(self):
        if not os.path.isfile(_BFL_CKPT):
            self.skipTest("BFL adapter absent (set F3_LORA_BFL_CKPT)")
        self._check(_BFL_CKPT, expected_modules=516 // 2)
        self._check_gated_mlp_in_tp_slice(_BFL_CKPT)


if __name__ == "__main__":
    unittest.main()
