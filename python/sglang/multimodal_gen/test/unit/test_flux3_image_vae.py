# SPDX-License-Identifier: Apache-2.0
"""Tests for the FLUX.3 image VAE (:class:`Flux3ImageVAE`): registry, fused
GroupNorm+SiLU wiring and spatial-parallel decode wiring (GPU-free, no checkpoint).
"""

import unittest

import torch


class TestFlux3ImageVAEConfigAndRegistry(unittest.TestCase):
    """Cheap checks: no GPU, no checkpoint (meta-device construction)."""

    def test_registry_resolution(self):
        from sglang.multimodal_gen.runtime.models.registry import (
            _ALIAS_TO_MODEL,
            ModelRegistry,
        )

        cls, _arch = ModelRegistry.resolve_model_cls("Flux3ImageVAE")
        self.assertEqual(cls.__name__, "Flux3ImageVAE")
        self.assertEqual(_ALIAS_TO_MODEL.get("AutoEncoder"), "Flux3ImageVAE")

    def test_fused_norm_act_is_wired(self):
        """apply_group_norm_silu only fuses when handed an nn.SiLU; without it
        every GroupNorm -> swish chain silently runs eager."""
        from sglang.multimodal_gen.configs.models.vaes.flux3_image import (
            Flux3ImageVAEConfig,
        )
        from sglang.multimodal_gen.runtime.models.vaes.flux3_image_vae import (
            Flux3ImageVAE,
        )

        with torch.device("meta"):
            vae = Flux3ImageVAE(Flux3ImageVAEConfig())
        for module in (vae.encoder, vae.decoder, vae.encoder.mid.block_1):
            self.assertIsInstance(module.act, torch.nn.SiLU)


class TestFlux3ImageVAESpatialParallel(unittest.TestCase):
    """Spatial-parallel decode wiring (GPU-free). The real sharded numerics are
    validated by tools/compare_flux3_image_vae_sp.py under torchrun."""

    def test_single_gpu_no_spatial_parallel(self):
        # No decode-parallel group in a plain test process -> serial fallback:
        # plain nn.Conv2d decoder, bit-identical to the serial VAE.
        from sglang.multimodal_gen.configs.models.vaes.flux3_image import (
            Flux3ImageVAEConfig,
        )
        from sglang.multimodal_gen.runtime.layers.parallel_conv import (
            SpatialParallelConv2d,
        )
        from sglang.multimodal_gen.runtime.models.vaes.flux3_image_vae import (
            Flux3ImageVAE,
        )

        with torch.device("meta"):
            vae = Flux3ImageVAE(Flux3ImageVAEConfig())
        self.assertFalse(vae._spatial_parallel)
        self.assertEqual(vae._sp_upsamples, 0)
        self.assertEqual(
            sum(
                1 for m in vae.decoder.modules() if isinstance(m, SpatialParallelConv2d)
            ),
            0,
        )

    def test_enable_spatial_parallel_swaps_convs(self):
        # Fake a world>1 decode group (mock the two accessors) so the public
        # SpatialParallelConv2d constructs without real distributed init; then
        # assert the swap + upsample count + state_dict-key preservation.
        from unittest import mock

        import torch.nn as nn

        from sglang.multimodal_gen.configs.models.vaes.flux3_image import (
            Flux3ImageVAEConfig,
        )
        from sglang.multimodal_gen.runtime.layers.parallel_conv import (
            SpatialParallelConv2d,
        )
        from sglang.multimodal_gen.runtime.models.vaes.flux3_image_vae import (
            Flux3ImageVAE,
            Upsample,
            enable_spatial_parallel_decoder,
        )

        pc = "sglang.multimodal_gen.runtime.layers.parallel_conv"
        with torch.device("meta"):
            vae = Flux3ImageVAE(Flux3ImageVAEConfig())
        before = set(vae.decoder.state_dict().keys())
        n_upsamples = sum(1 for m in vae.decoder.modules() if isinstance(m, Upsample))

        with (
            mock.patch(f"{pc}.get_decode_parallel_rank", return_value=0),
            mock.patch(f"{pc}.get_decode_parallel_world_size", return_value=2),
            torch.device("meta"),
        ):
            returned = enable_spatial_parallel_decoder(vae.decoder)

        self.assertEqual(returned, n_upsamples)
        self.assertEqual(returned, 3)  # ch_mult=[1,2,4,4] -> 3 upsamples
        # state_dict keys unchanged (SpatialParallelConv2d subclasses nn.Conv2d).
        self.assertEqual(before, set(vae.decoder.state_dict().keys()))
        # every plain Conv2d became SpatialParallelConv2d.
        self.assertGreater(
            sum(
                1 for m in vae.decoder.modules() if isinstance(m, SpatialParallelConv2d)
            ),
            0,
        )
        self.assertEqual(
            sum(1 for m in vae.decoder.modules() if type(m) is nn.Conv2d), 0
        )


if __name__ == "__main__":
    unittest.main()
