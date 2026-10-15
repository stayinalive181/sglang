# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT, pack_cfg
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
    latent_ids,
    scaled_ids,
    validate_generation,
)


class TestFlux3PackedConditioning(unittest.TestCase):
    def test_cfg_pack_preserves_sample_boundaries_and_empty_condition(self):
        image = (torch.randn(4, 8), torch.randn(4, 4))
        text = [
            (torch.randn(2, 16), torch.randn(2, 4)),
            (torch.randn(5, 16), torch.randn(5, 4)),
        ]
        packed = pack_cfg(
            text,
            None,
            image,
            {"image": 8, "image_cond": 8, TEXT: 16},
        )
        self.assertEqual(packed.mode_indices["image"], [(0, 4), (4, 8)])
        self.assertEqual(packed.mode_indices["image_cond"], [(0, 0), (0, 0)])
        self.assertEqual(packed.mode_indices[TEXT], [(0, 2), (2, 7)])
        self.assertEqual(packed.data["image"].shape, (8, 8))

    def test_image_ids_use_zero_time_and_sequence_axes(self):
        ids = latent_ids(2, 3)
        self.assertTrue(torch.equal(ids[:, 0], torch.zeros(6)))
        self.assertTrue(torch.equal(ids[:, 3], torch.zeros(6)))
        scaled = scaled_ids(ids, 0.5)
        torch.testing.assert_close(scaled[:, 1:3], ids[:, 1:3].float() * 0.5)

    def test_t2i_and_i2i_validation(self):
        self.assertFalse(
            validate_generation(
                "t2i",
                prompt="prompt",
                image_path=None,
            )
        )
        self.assertTrue(
            validate_generation(
                "i2i",
                prompt="prompt",
                image_path="reference.png",
            )
        )
        with self.assertRaisesRegex(ValueError, "requires image_path"):
            validate_generation(
                "i2i",
                prompt="prompt",
                image_path=None,
            )

    def test_references_follow_the_reference_preprocessing(self):
        """BFL's rules: one reference is fitted under 4 MP and reflect-padded to
        the 16-px grid, sets the output size, and is cropped back out; several
        are each fitted under 1 MP and centre-cropped; at most 10."""
        import numpy as np
        from PIL import Image

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            crop_output,
            load_references,
            output_size,
        )

        rng = np.random.default_rng(0)
        odd = Image.fromarray(rng.integers(0, 255, (1001, 1503, 3), dtype=np.uint8))
        (tensor,), size, crop = load_references([odd])
        self.assertEqual(size, (1504, 1008))
        self.assertEqual(tuple(tensor.shape), (3, 1008, 1504))
        self.assertEqual(crop, (0, 3, 1503, 1004))
        torch.testing.assert_close(
            crop_output(tensor, crop),
            torch.from_numpy(np.asarray(odd)).permute(2, 0, 1).float() / 127.5 - 1,
        )
        # Above 4 MP the canvas shrinks under the limit, padding included.
        _, size, _ = load_references([Image.new("RGB", (4000, 3000))])
        self.assertLessEqual(size[0] * size[1], 4 * 1024 * 1024)
        self.assertEqual((size[0] % 16, size[1] % 16), (0, 0))

        tensors, size, crop = load_references(
            [Image.new("RGB", (2000, 1000)), Image.new("RGB", (500, 333))]
        )
        self.assertEqual(
            [tuple(t.shape) for t in tensors], [(3, 720, 1440), (3, 320, 496)]
        )
        self.assertEqual((size, crop), ((1440, 720), None))
        with self.assertRaisesRegex(ValueError, "1 to 10"):
            load_references([Image.new("RGB", (64, 64))] * 11)

        # Unset sides take the first reference's size; the crop needs both unset.
        self.assertEqual(
            output_size(
                None, None, (1504, 1008), crop=(0, 3, 1503, 1004), default=(1024, 1024)
            ),
            (1504, 1008, (0, 3, 1503, 1004)),
        )
        self.assertEqual(
            output_size(
                1024, None, (1504, 1008), crop=(0, 3, 1503, 1004), default=(1024, 1024)
            ),
            (1024, 1008, None),
        )
        self.assertEqual(
            output_size(None, None, None, crop=None, default=(1024, 1024)),
            (1024, 1024, None),
        )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "t2i.*i2i"):
            validate_generation(
                "unsupported",
                prompt="prompt",
                image_path=None,
            )


if __name__ == "__main__":
    unittest.main()
