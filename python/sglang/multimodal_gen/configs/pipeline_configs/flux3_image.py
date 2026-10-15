# SPDX-License-Identifier: Apache-2.0
"""Pipeline config and sampling params for image-only FLUX.3."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.flux3 import (
    Flux3ArchConfig,
    Flux3DiTConfig,
)
from sglang.multimodal_gen.configs.models.vaes.flux3_image import Flux3ImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
        Flux3ImagePackage,
    )

_IMAGE_MODES = frozenset({"t2i", "i2i"})
# Release repos: the distilled flow map and the CFG base model.
FLUX3_IMAGE_HF_PATHS = [
    "black-forest-labs/flux-3-image-klein",
    "black-forest-labs/flux-3-image-klein-base",
    "herzogenhorn/f3-image-small",
]
# Lower-case substrings of a Hub id or path that name a FLUX.3 image model.
FLUX3_IMAGE_NAME_PATTERNS = ("flux-3-image", "flux.3-image", "flux3-image")


def is_flux3_image_package(model_path: str) -> bool:
    """Whether a local directory holds a FLUX.3 image package config, whatever
    its name; the image DiT layout identifies it."""
    try:
        raw = json.loads((Path(model_path).expanduser() / "config.json").read_text())
    except (OSError, ValueError):
        return False
    dit = raw.get("dit") if isinstance(raw, dict) else None
    return (
        isinstance(dit, dict)
        and dit.get("out_modes") == ["image"]
        and isinstance(dit.get("in_channels"), dict)
        and "image_cond" in dit["in_channels"]
    )


def flux3_image_dit_config(**arch_overrides: Any) -> Flux3DiTConfig:
    """The FLUX 3 DiT config with image streams (defaults: image-small)."""
    arch = {
        "in_channels": {"image": 128, "image_cond": 128},
        "sequence": {"x_image": "image", "x_image_cond": "image_cond"},
        "out_modes": ["image"],
        "vec_in_dim": None,
        # Empty: the native DiT owns its packed-QKV checkpoint mapping.
        "param_names_mapping": {},
        # Adapter key prefixes: BFL/ostris ``dit.`` and diffusers ``transformer.``.
        "lora_param_names_mapping": {r"^dit\.": "", r"^transformer\.": ""},
        **arch_overrides,
    }
    return Flux3DiTConfig(arch_config=Flux3ArchConfig(**arch))


@dataclass
class Flux3ImageSamplingParams(SamplingParams):
    """Request params for the image-only FLUX.3 pipeline."""

    data_type: DataType = DataType.IMAGE
    _default_height = 1024
    _default_width = 1024

    # The reference CFG pairs the prompt with an empty negative prompt.
    negative_prompt: str = ""
    # ``None`` means "use the server default".
    flux3_mode: str | None = None

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset({"flux3_mode"})

    def prepare_synthetic_warmup_request_for_queue(
        self, req: Any, server_args: Any
    ) -> None:
        # Warmup copies the request's steps as the full-workload target before
        # any adjustment; ``None`` (server default) has to be resolved here.
        for key in (
            "warmup_target_num_inference_steps",
            "cache_dit_num_inference_steps",
        ):
            if key in req.extra and req.extra[key] is None:
                req.extra[key] = server_args.pipeline_config.flux3_steps
        mode = self.flux3_mode or server_args.pipeline_config.flux3_mode
        if mode == "t2i" or not req.image_path:
            # TI2I permits references for i2i requests on the same server, but
            # this synthetic warmup is unconditioned; with no reference (none
            # is synthesized under breakable CUDA graphs) it warms the t2i path.
            req.image_path = None
            self.flux3_mode = "t2i"

    def _validate(self) -> None:
        super()._validate()
        if self.flux3_mode is not None and self.flux3_mode not in _IMAGE_MODES:
            raise ValueError(
                f"flux3_mode must be one of {sorted(_IMAGE_MODES)}, "
                f"got {self.flux3_mode!r}"
            )


@dataclass
class Flux3ImagePipelineConfig(PipelineConfig):
    """Model-level config for the served FLUX.3 pipeline.

    ``flux3_*`` fields default to the image-small package contract and are read
    by the FLUX.3 text/denoise/decode stages."""

    # One server serves text-to-image and image edits.
    task_type: ModelTaskType = ModelTaskType.TI2I

    # Standard model configs the framework reads during ServerArgs setup. The
    # custom load_modules builds the real components; these keep generic tooling
    # (task-type checks, vae scale factors) sane. The text encoder is the real
    # Qwen3-VL embedder built directly in load_modules (not a diffusers text
    # encoder), so text_encoder_configs is empty.
    dit_config: DiTConfig = field(default_factory=flux3_image_dit_config)
    vae_config: VAEConfig = field(default_factory=Flux3ImageVAEConfig)
    text_encoder_configs: tuple = field(default_factory=tuple)
    text_encoder_precisions: tuple = field(default_factory=tuple)
    text_encoder_extra_args: list = field(default_factory=list)
    preprocess_text_funcs: tuple = field(default_factory=tuple)
    postprocess_text_funcs: tuple = field(default_factory=tuple)

    # FLUX.3 loads and preprocesses its own conditioning image.
    skip_input_image_preprocess: bool = True

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_slicing: bool = False
    vae_sp: bool = False
    # FLUX.3 timestep embedding is not the diffusers embedded-CFG guidance path.
    should_use_guidance: bool = False

    # --- FLUX.3 generation knobs (read by the model-specific stages) ---
    flux3_mode: str = "t2i"  # t2i | i2i
    # ``None`` takes the package sampler's value (see validate_server_args).
    flux3_steps: int | None = None
    flux3_cfg_scale: float | None = None
    flux3_alpha: float | None = None
    # The served image-small package (a local directory), resolved once by
    # validate_server_args; the loader builds the components from it.
    flux3_package: "Flux3ImagePackage | None" = field(default=None, repr=False)
    # Dynamic compilation covers the package-defined variable token buckets.
    # Reference-parity tests explicitly construct an eager encoder.
    flux3_compile_text_encoder: bool = True
    # Weight-only FP8 store; dequant on each encode. Default off.
    flux3_text_encoder_fp8: bool = False
    # Keep the visual VAE on GPU after a real request. Default off.
    flux3_keep_visual_vae_resident: bool = False
    # Keep DiT non-layer weights on GPU through VAE decode. Default off.
    flux3_dit_keep_non_layer: bool = False
    skip_empty_cache_after_request: bool = True

    # Plain class attrs (NOT dataclass fields — no annotation).
    _IMAGE_MODES = _IMAGE_MODES

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            # 15.6 GB of DiT weights on top of an 8.3 GB text encoder: 28.3 GB of
            # components before any activation. A 32 GB card cannot run this with
            # the DiT resident, and the default empty tuple makes the gate in
            # _should_auto_enable_dit_layerwise_offload always false, so auto mode
            # leaves it resident until the request OOMs.
            dit_layerwise_offload_modes=("auto", "memory"),
        )

    def check_pipeline_config(self) -> None:
        """Validate image-only FLUX.3 settings."""
        if self.flux3_mode not in self._IMAGE_MODES:
            raise ValueError(
                f"flux3_mode must be one of {sorted(self._IMAGE_MODES)}, "
                f"got {self.flux3_mode!r}"
            )
        super().check_pipeline_config()

    def validate_server_args(self, server_args: Any) -> None:
        """Resolve the served package once, adopt its sampler defaults and
        reject CFG on a flow map."""
        super().validate_server_args(server_args)
        from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
            resolve_flux3_image_package,
        )

        self.flux3_package = resolve_flux3_image_package(
            server_args.model_path,
            variant=server_args.model_variant,
            revision=server_args.revision,
        )
        package = self.flux3_package.config
        # A step-distilled flow map samples without CFG.
        if package.dit.flow_map:
            if self.flux3_cfg_scale not in (None, 1.0):
                raise ValueError(
                    "the FLUX.3 flow-map (distilled) DiT samples without CFG; "
                    "drop --flux3-cfg-scale"
                )
            if server_args.enable_cfg_parallel:
                raise ValueError(
                    "the FLUX.3 flow-map (distilled) DiT has no CFG branch to "
                    "parallelize; drop --enable-cfg-parallel"
                )
        else:
            if self.flux3_cfg_scale is None:
                self.flux3_cfg_scale = package.sampler.guidance
        if self.flux3_steps is None:
            self.flux3_steps = package.sampler.steps
        if self.flux3_alpha is None:
            self.flux3_alpha = package.sampler.alpha

    def supports_dynamic_batching(self):
        """FLUX.3's driver, samplers, and CFG doubling are single-sample per
        forward (one sample plus its CFG negative), so it cannot process a
        merged prompt batch."""
        return False

    def served_modes(self) -> frozenset[str]:
        """Image modes accepted by one FLUX.3 server."""
        return self._IMAGE_MODES

    @staticmethod
    def add_pipeline_cli_args(parser):
        """Register FLUX.3 generation flags. Invoked by the diffusion generate CLI
        for ``--pipeline-class-name Flux3ImagePipeline`` (generic resolved-config hook).
        All defaults are ``None`` so a flag only overrides the config default when
        the user actually passes it (``update_config_from_args`` skips ``None``)."""

        def _bool(v: str) -> bool:
            return str(v).lower() in ("1", "true", "yes", "y", "on")

        g = parser.add_argument_group("FLUX.3")
        g.add_argument(
            "--flux3-mode",
            dest="flux3_mode",
            default=None,
            choices=["t2i", "i2i"],
            help="FLUX.3 generation mode (default t2i).",
        )
        g.add_argument(
            "--flux3-steps",
            dest="flux3_steps",
            type=int,
            default=None,
            help="Denoise steps.",
        )
        g.add_argument(
            "--flux3-cfg-scale",
            dest="flux3_cfg_scale",
            type=float,
            default=None,
            help="CFG scale.",
        )
        g.add_argument(
            "--flux3-alpha",
            dest="flux3_alpha",
            type=float,
            default=None,
            help="Timestep-shift alpha.",
        )
        g.add_argument(
            "--flux3-compile-text-encoder",
            dest="flux3_compile_text_encoder",
            type=_bool,
            default=None,
            help="Compile the text encoder's language model (default true).",
        )
        g.add_argument(
            "--flux3-text-encoder-fp8",
            dest="flux3_text_encoder_fp8",
            type=_bool,
            default=None,
            help="Store the Qwen3-VL encoder as weight-only FP8 (default false).",
        )
        g.add_argument(
            "--flux3-keep-visual-vae-resident",
            dest="flux3_keep_visual_vae_resident",
            type=_bool,
            default=None,
            help="Keep the visual VAE on GPU between requests (default false).",
        )
        g.add_argument(
            "--flux3-dit-keep-non-layer",
            dest="flux3_dit_keep_non_layer",
            type=_bool,
            default=None,
            help="Keep DiT non-layer weights on GPU through VAE decode (default false).",
        )
        g.add_argument(
            "--flux3-skip-empty-cache-after-request",
            dest="skip_empty_cache_after_request",
            type=_bool,
            default=None,
            help="Skip post-request empty_cache (default true).",
        )
        return parser


def register():
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=Flux3ImageSamplingParams,
        pipeline_config_cls=Flux3ImagePipelineConfig,
        hf_model_paths=FLUX3_IMAGE_HF_PATHS,
        model_detectors=[
            lambda hf_id: any(p in hf_id.lower() for p in FLUX3_IMAGE_NAME_PATTERNS),
        ],
    )
