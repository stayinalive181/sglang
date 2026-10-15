# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
    Flux3ImageDenoiseStage,
)


def test_packed_flux3_disables_cache_dit():
    stage = Flux3ImageDenoiseStage.__new__(Flux3ImageDenoiseStage)
    assert not stage._cache_dit_requested_for_batch(SimpleNamespace())
