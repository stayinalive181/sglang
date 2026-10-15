# SPDX-License-Identifier: Apache-2.0
"""Strict resolver for the FLUX.3 image-small model package."""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path, PurePosixPath

FORMAT_VERSION = 1
MODEL_NAME = "f3-image-small"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "model.safetensors"
QUANT_CONFIG_NAME = "hf_quant_config.json"
FP8R_VARIANT = "fp8r"
NVFP4_VARIANT = "nvfp4"
# The NVFP4 decode the runtime implements; a manifest must state every clause.
_NVFP4_SCALE_CLAUSES = (
    "e2m1(weight) * weight_scale[:, k // 16] * weight_scale_2",
    "weight_scale_2 = amax(weight) / (448 * 6)",
    "input_scale = p99.9 of per-call input amax / (448 * 6)",
)
# Variant directory -> the quantization its config must declare. ``sd`` is the
# step-distilled flow-map DiT; ``sd-*`` are its quantized exports.
VARIANT_QUANTIZATION = {
    FP8R_VARIANT: FP8R_VARIANT,
    NVFP4_VARIANT: NVFP4_VARIANT,
    "sd": None,
    f"sd-{FP8R_VARIANT}": FP8R_VARIANT,
    f"sd-{NVFP4_VARIANT}": NVFP4_VARIANT,
}
TORCH_DTYPES = ("bfloat16", "float16", "float32")


class Flux3ImagePackageError(ValueError):
    pass


@dataclass(frozen=True)
class Flux3ImageDiTPackageConfig:
    hidden: int
    heads: int
    depth: int
    depth_single: int
    mlp_ratio: float
    axes_dim: tuple[int, ...]
    theta: int
    in_channels: dict[str, int]
    out_modes: tuple[str, ...]
    vec_in_dim: int | None = None
    image_rope_scale: float = 0.5
    # A flow-map DiT also embeds the time each step jumps to (``ref_time_in``).
    flow_map: bool = False


@dataclass(frozen=True)
class Flux3ImageTextEncoderPackageConfig:
    path: str
    tokenizer: str
    layers: tuple[int, ...]
    pad_multiple: int
    max_tokens: int
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class Flux3ImageVAEPackageConfig:
    path: str
    dtype: str = "float32"


@dataclass(frozen=True)
class Flux3ImageSamplerPackageConfig:
    steps: int
    # CFG scale; ``None`` for a flow map, which samples without CFG.
    guidance: float | None
    alpha: float
    size: tuple[int, int]


@dataclass(frozen=True)
class Flux3ImagePackageConfig:
    format_version: int
    model: str
    quantization: str | None
    torch_dtype: str
    dit: Flux3ImageDiTPackageConfig
    text_encoder: Flux3ImageTextEncoderPackageConfig
    vae: Flux3ImageVAEPackageConfig
    sampler: Flux3ImageSamplerPackageConfig

    @classmethod
    def from_dict(cls, raw: dict) -> Flux3ImagePackageConfig:
        raw = _section(
            raw,
            cls,
            "config",
            nested={"dit", "text_encoder", "vae", "sampler"},
        )
        config = cls(
            **{
                key: value
                for key, value in raw.items()
                if key not in ("dit", "text_encoder", "vae", "sampler")
            },
            dit=_build(
                Flux3ImageDiTPackageConfig,
                raw["dit"],
                "dit",
                tuples=("axes_dim", "out_modes"),
            ),
            text_encoder=_build(
                Flux3ImageTextEncoderPackageConfig,
                raw["text_encoder"],
                "text_encoder",
                tuples=("layers",),
            ),
            vae=_build(Flux3ImageVAEPackageConfig, raw["vae"], "vae"),
            sampler=_build(
                Flux3ImageSamplerPackageConfig,
                raw["sampler"],
                "sampler",
                tuples=("size",),
            ),
        )
        _validate_config(config)
        return config


@dataclass(frozen=True)
class Flux3ImagePackage:
    root: Path
    config: Flux3ImagePackageConfig
    weights: Path
    vae: Path
    text_encoder: Path
    tokenizer: Path
    manifest: Path
    directory: Path | None = None
    variant: str | None = None
    quant_config: Path | None = None


def load_flux3_image_package_config(path: str | Path) -> Flux3ImagePackageConfig:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise Flux3ImagePackageError(
            f"FLUX.3 package config not found: {path}"
        ) from None
    except json.JSONDecodeError as exc:
        raise Flux3ImagePackageError(f"{path}: invalid JSON ({exc})") from None
    try:
        return Flux3ImagePackageConfig.from_dict(raw)
    except Flux3ImagePackageError as exc:
        raise Flux3ImagePackageError(f"{path}: {exc}") from None


def resolve_flux3_image_package(
    source: str | Path, variant: str | None = None
) -> Flux3ImagePackage:
    """Resolve the BF16 root package or a supported shared-asset variant."""
    directory = Path(source).expanduser()
    if directory.is_file():
        if directory.name not in {
            CONFIG_NAME,
            WEIGHTS_NAME,
            MANIFEST_NAME,
            QUANT_CONFIG_NAME,
        }:
            raise Flux3ImagePackageError(
                f"expected a FLUX.3 package directory, {CONFIG_NAME}, or "
                f"{WEIGHTS_NAME}; got {directory}"
            )
        directory = directory.parent
    if not directory.is_dir():
        raise Flux3ImagePackageError(f"FLUX.3 package directory not found: {directory}")

    direct_variant = directory.parent.name == "variants"
    if direct_variant:
        detected_variant = directory.name
        root = directory.parent.parent
        if variant is not None and variant != detected_variant:
            raise Flux3ImagePackageError(
                f"variant path selects {detected_variant!r}, but variant={variant!r}"
            )
        variant = detected_variant
    else:
        root = directory
        if variant is not None:
            directory = root / "variants" / variant

    if variant is not None and variant not in VARIANT_QUANTIZATION:
        raise Flux3ImagePackageError(
            f"unsupported FLUX.3 package variant {variant!r}; "
            f"supported variants: {', '.join(map(repr, VARIANT_QUANTIZATION))}"
        )
    if not directory.is_dir():
        raise Flux3ImagePackageError(
            f"FLUX.3 package variant {variant!r} not found: {directory}"
        )

    config = load_flux3_image_package_config(directory / CONFIG_NAME)
    if config.model != MODEL_NAME:
        raise Flux3ImagePackageError(
            f"unsupported FLUX.3 package model {config.model!r}; expected {MODEL_NAME!r}"
        )
    expected_quantization = VARIANT_QUANTIZATION.get(variant)
    if config.quantization != expected_quantization:
        raise Flux3ImagePackageError(
            f"FLUX.3 package variant {variant!r} requires "
            f"quantization={expected_quantization!r}, got {config.quantization!r}"
        )
    if config.torch_dtype != "bfloat16":
        raise Flux3ImagePackageError(
            f"FLUX.3 package expected torch_dtype='bfloat16', "
            f"got {config.torch_dtype!r}"
        )

    weights = directory / WEIGHTS_NAME
    vae = root / _relative(config.vae.path, "vae.path")
    text_encoder = root / _relative(config.text_encoder.path, "text_encoder.path")
    tokenizer = root / _relative(
        config.text_encoder.tokenizer, "text_encoder.tokenizer"
    )
    manifest = directory / MANIFEST_NAME
    quant_config = (
        directory / QUANT_CONFIG_NAME if expected_quantization is not None else None
    )
    required = {
        "DiT weights": weights,
        "image VAE": vae,
        "text encoder": text_encoder,
        "tokenizer": tokenizer,
        "manifest": manifest,
    }
    if quant_config is not None:
        required[f"{expected_quantization.upper()} quantization config"] = quant_config
    missing = [
        f"{name}: {path}" for name, path in required.items() if not path.exists()
    ]
    if missing:
        raise Flux3ImagePackageError(
            "FLUX.3 package is incomplete; missing " + ", ".join(missing)
        )
    _validate_manifest(manifest, config)
    return Flux3ImagePackage(
        root=root,
        config=config,
        weights=weights,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        manifest=manifest,
        directory=directory,
        variant=variant,
        quant_config=quant_config,
    )


def _section(raw, cls, name: str, nested=frozenset()) -> dict:
    if not isinstance(raw, dict):
        raise Flux3ImagePackageError(
            f"{name} must be an object, got {type(raw).__name__}"
        )
    names = {item.name for item in fields(cls)}
    required = {
        item.name
        for item in fields(cls)
        if item.default is MISSING and item.default_factory is MISSING
    }
    unknown = set(raw) - names
    missing = required - set(raw)
    if unknown:
        raise Flux3ImagePackageError(f"unknown key(s) in {name}: {sorted(unknown)}")
    if missing:
        raise Flux3ImagePackageError(f"missing key(s) in {name}: {sorted(missing)}")
    for key in nested:
        if not isinstance(raw.get(key), dict):
            raise Flux3ImagePackageError(f"{name}.{key} must be an object")
    return raw


def _build(cls, raw, name: str, tuples=()):
    raw = dict(_section(raw, cls, name))
    for key in tuples:
        if key in raw:
            if not isinstance(raw[key], list):
                raise Flux3ImagePackageError(f"{name}.{key} must be a list")
            raw[key] = tuple(raw[key])
    return cls(**raw)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise Flux3ImagePackageError(message)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _relative(path: str, what: str) -> str:
    value = PurePosixPath(path)
    _check(
        isinstance(path, str)
        and bool(path)
        and not value.is_absolute()
        and ".." not in value.parts,
        f"{what} must be a relative path inside the package, got {path!r}",
    )
    return path


def _validate_config(config: Flux3ImagePackageConfig) -> None:
    _check(
        config.format_version == FORMAT_VERSION,
        f"format_version must be {FORMAT_VERSION}, got {config.format_version!r}",
    )
    _check(
        config.torch_dtype in TORCH_DTYPES,
        f"unsupported torch_dtype {config.torch_dtype!r}",
    )
    dit = config.dit
    for key in ("hidden", "heads", "depth", "depth_single", "theta"):
        _check(
            _is_int(getattr(dit, key)) and getattr(dit, key) > 0,
            f"dit.{key} must be a positive integer",
        )
    _check(
        dit.hidden % dit.heads == 0,
        f"dit.hidden {dit.hidden} must be divisible by dit.heads {dit.heads}",
    )
    _check(
        all(_is_int(axis) and axis > 0 and axis % 2 == 0 for axis in dit.axes_dim),
        "dit.axes_dim must contain positive even integers",
    )
    _check(
        sum(dit.axes_dim) == dit.hidden // dit.heads,
        "sum(dit.axes_dim) must equal the attention head dimension",
    )
    _check(
        dit.in_channels == {"image": 128, "image_cond": 128, "text": 20480},
        f"unsupported image-small channels: {dit.in_channels}",
    )
    _check(
        dit.out_modes == ("image",),
        f"unsupported image-small output modes: {dit.out_modes}",
    )
    _check(dit.vec_in_dim is None, "image-small must not define vec_in_dim")
    _check(
        _is_num(dit.image_rope_scale) and dit.image_rope_scale > 0,
        "dit.image_rope_scale must be positive",
    )
    _check(isinstance(dit.flow_map, bool), "dit.flow_map must be a boolean")
    text = config.text_encoder
    _relative(text.path, "text_encoder.path")
    _relative(text.tokenizer, "text_encoder.tokenizer")
    _check(bool(text.layers), "text_encoder.layers must not be empty")
    _check(
        _is_int(text.pad_multiple) and text.pad_multiple > 0,
        "text_encoder.pad_multiple must be positive",
    )
    _check(
        _is_int(text.max_tokens) and text.max_tokens >= text.pad_multiple,
        "text_encoder.max_tokens must be >= pad_multiple",
    )
    _relative(config.vae.path, "vae.path")
    _check(
        config.vae.dtype in TORCH_DTYPES,
        f"unsupported VAE dtype {config.vae.dtype!r}",
    )
    sampler = config.sampler
    _check(
        _is_int(sampler.steps) and sampler.steps > 0, "sampler.steps must be positive"
    )
    if config.dit.flow_map:
        _check(sampler.guidance is None, "a flow-map sampler.guidance must be null")
    else:
        _check(_is_num(sampler.guidance), "sampler.guidance must be numeric")
    _check(
        _is_num(sampler.alpha) and sampler.alpha > 0,
        "sampler.alpha must be positive",
    )
    _check(
        len(sampler.size) == 2
        and all(
            _is_int(value) and value > 0 and value % 16 == 0 for value in sampler.size
        ),
        "sampler.size must be positive multiples of 16",
    )


def _validate_manifest(path: Path, config: Flux3ImagePackageConfig) -> None:
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Flux3ImagePackageError(f"{path}: invalid JSON ({exc})") from None
    _check(
        manifest.get("format_version") == 1,
        f"{path}: unsupported manifest format_version",
    )
    _check(
        manifest.get("model", config.model) == config.model,
        f"{path}: manifest model does not match config",
    )
    _check(
        manifest.get("quantization", config.quantization) == config.quantization,
        f"{path}: manifest quantization does not match config",
    )
    if config.quantization == FP8R_VARIANT:
        expected = {
            "dtype": "float8_e4m3fn",
            "scale_convention": "weight.float() * weight_scale[:, None]",
        }
        for key, value in expected.items():
            _check(
                manifest.get(key) == value,
                f"{path}: expected {key}={value!r}, got {manifest.get(key)!r}",
            )
        # Exports may keep different layers in BF16, so counts are only self-consistent.
        matrices = manifest.get("fp8_matrices")
        _check(
            _is_int(matrices)
            and matrices > 0
            and manifest.get("weight_scales") == matrices,
            f"{path}: expected one weight scale per FP8 matrix, got "
            f"fp8_matrices={matrices!r}, weight_scales={manifest.get('weight_scales')!r}",
        )
    elif config.quantization == NVFP4_VARIANT:
        _check(
            str(manifest.get("dtype", "")).startswith("nvfp4: weight uint8"),
            f"{path}: manifest dtype does not describe packed NVFP4 weights",
        )
        _check(
            manifest.get("group_size") == 16,
            f"{path}: expected group_size=16, got {manifest.get('group_size')!r}",
        )
        # Extra clauses only describe how scales were calibrated, not how they decode.
        clauses = str(manifest.get("scale_convention", "")).split("; ")
        missing = [c for c in _NVFP4_SCALE_CLAUSES if c not in clauses]
        _check(not missing, f"{path}: scale_convention lacks {missing!r}")
        # Exports may keep different layers in BF16, so counts are only self-consistent.
        # Older exports store the per-block scale count under weight_scales.
        matrices = manifest.get("nvfp4_matrices")
        counts = {
            key: manifest.get(key)
            for key in ("weight_scale_tensors", "weight_scale_2", "input_scales")
        }
        if counts["weight_scale_tensors"] is None:
            counts["weight_scale_tensors"] = manifest.get("weight_scales")
        _check(
            _is_int(matrices)
            and matrices > 0
            and all(count == matrices for count in counts.values()),
            f"{path}: expected {matrices!r} NVFP4 matrices with one scale set "
            f"each, got {counts!r}",
        )
