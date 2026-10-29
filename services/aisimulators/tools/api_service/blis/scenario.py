"""Build request-scoped BLIS v0.13 Scenario and Deployment documents."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .manifest import BlisUnsupportedRequest, ModelResolution

_COEFFICIENTS = [
    "cost-model-primitives",
    "cost-model-collectives",
    "cost-model-host-overheads",
    "cost-model-attention",
    "cost-model-recurrent",
    "cost-model-memory",
]
_WEIGHT_QUANTIZATION = {
    "bf16": None,
    "fp16": None,
    "half": None,
    "fp8": "fp8",
    "int8": "int8",
    "int4": "int4",
    "mxfp4": "mxfp4",
    "nvfp4": "nvfp4",
}
_CACHE_DTYPE = {
    "bf16": "bf16",
    "fp16": "fp16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}


@dataclass(frozen=True)
class ScenarioRequest:
    model: ModelResolution
    hardware: str
    system: str
    gpus_per_node: int
    engine_version: str
    tp: int
    pp: int
    dp: int
    moe_tp_size: int | None
    moe_ep_size: int | None
    enable_expert_parallel: bool
    isl: int
    osl: int
    concurrency: int
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    weight_quantization: str | None = None
    moe_quantization: str | None = None
    cache_dtype: str | None = None
    prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    fmha_quantization: str | None = None


def _scenario_name(request: ScenarioRequest) -> str:
    canonical = json.dumps(
        asdict(request),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    safe_model = re.sub(r"[^a-z0-9-]+", "-", request.model.catalog_name.lower()).strip("-")
    return f"configiq-{safe_model}-{digest}"


def _optional_quantization(raw: str | None, supported: dict[str, str | None], name: str) -> str | None:
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if not normalized:
        return None
    if normalized not in supported:
        choices = ", ".join(sorted(supported))
        raise BlisUnsupportedRequest(f"BLIS cannot represent {name}={raw!r}; supported values: {choices}")
    return supported[normalized]


def normalize_weight_quantization(raw: str | None) -> str | None:
    """Return the BLIS scenario value for a supported weight-precision setting."""
    return _optional_quantization(raw, _WEIGHT_QUANTIZATION, "weight quantization")


def build_documents(request: ScenarioRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a validated-shape Scenario and Deployment for one single-node run."""
    if not re.fullmatch(r"\d+(?:\.\d+){1,2}(?:[-+][A-Za-z0-9.-]+)?", request.engine_version):
        raise BlisUnsupportedRequest(f"BLIS requires a numeric engine version, got {request.engine_version!r}")
    if min(request.tp, request.pp, request.dp, request.isl, request.osl, request.concurrency) < 1:
        raise BlisUnsupportedRequest("BLIS parallelism and workload values must be positive integers")
    if request.gpus_per_node < 1:
        raise BlisUnsupportedRequest(f"BLIS has no valid node layout for system {request.system!r}")
    total_gpus = request.tp * request.pp * request.dp
    if total_gpus > request.gpus_per_node:
        raise BlisUnsupportedRequest(
            f"BLIS scenario generation is currently single-node: {total_gpus} GPUs requested, "
            f"but {request.system} has {request.gpus_per_node} GPUs per node"
        )
    if request.moe_tp_size not in (None, 0, request.tp):
        raise BlisUnsupportedRequest(
            "BLIS derives expert tensor parallelism from the pool TP size; "
            f"it cannot represent moe_tp_size={request.moe_tp_size} with tp_size={request.tp}"
        )
    derived_ep_size = request.tp * request.dp if request.enable_expert_parallel else 1
    if request.moe_ep_size not in (None, 0, derived_ep_size):
        raise BlisUnsupportedRequest(
            "BLIS derives expert parallel width from TP × DP; "
            f"it cannot represent moe_ep_size={request.moe_ep_size} for TP={request.tp}, DP={request.dp}"
        )
    if request.enable_chunked_prefill is not None:
        raise BlisUnsupportedRequest("BLIS scenarios do not expose the vLLM enable_chunked_prefill setting")
    if request.fmha_quantization:
        raise BlisUnsupportedRequest("BLIS scenarios do not expose an independent FMHA quantization setting")
    raw_weight_quant = request.weight_quantization.strip().lower() if request.weight_quantization else None
    raw_moe_quant = request.moe_quantization.strip().lower() if request.moe_quantization else None
    if raw_moe_quant and raw_weight_quant != raw_moe_quant:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")
    moe_quant = _optional_quantization(request.moe_quantization, _WEIGHT_QUANTIZATION, "MoE quantization")

    name = _scenario_name(request)
    engine: dict[str, Any] = {
        "block_size": 16,
        "max_num_batched_tokens": request.max_num_batched_tokens or max(8192, request.isl),
        "max_num_seqs": request.max_num_seqs or request.concurrency,
        "max_model_len": request.max_model_len if request.max_model_len is not None else "auto",
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": (
            request.gpu_memory_utilization
            if request.gpu_memory_utilization is not None
            else 0.92
        ),
    }
    quantization = normalize_weight_quantization(request.weight_quantization)
    cache_dtype = _optional_quantization(request.cache_dtype, _CACHE_DTYPE, "KV-cache dtype")
    if moe_quant and moe_quant != quantization:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")
    if quantization:
        engine["quantization"] = quantization
    if cache_dtype:
        engine["cache_dtype"] = cache_dtype
    # vLLM's schema default is prefix caching enabled. State it explicitly so
    # the emitted Scenario and normalized response agree on the applied setting.
    engine["enable_prefix_caching"] = request.prefix_caching if request.prefix_caching is not None else True

    scenario = {
        "kind": "Scenario",
        "name": name,
        "engine_version": request.engine_version,
        "model": request.model.catalog_name,
        "coefficients": list(_COEFFICIENTS),
        "cluster": {
            "hardware": request.hardware,
            "nodes": 1,
            "gpus_per_node": request.gpus_per_node,
        },
    }
    deployment = {
        "kind": "Deployment",
        "name": name,
        "pools": [
            {
                "role": "colocated",
                "nodes": 1,
                "parallel": {
                    "tp": request.tp,
                    "pp": request.pp,
                    "dp": request.dp,
                    "enable_expert_parallel": request.enable_expert_parallel,
                },
                "engine": engine,
            }
        ],
    }
    return scenario, deployment


def write_scenario(directory: str | Path, request: ScenarioRequest) -> str:
    """Write a uniquely named two-document YAML scenario and return its filename."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    scenario, deployment = build_documents(request)
    filename = f"{scenario['name']}.yaml"
    target = target_dir / filename
    target.write_text(
        yaml.safe_dump_all(
            [scenario, deployment],
            explicit_start=False,
            sort_keys=False,
            allow_unicode=False,
        )
    )
    return filename


def default_engine_version(backend_version: str | None) -> str:
    """Use an explicit numeric pin when supplied, otherwise the image's BLIS default."""
    if backend_version and backend_version.strip():
        version = backend_version.strip()
        if version in {"current", "previous", "next"}:
            raise BlisUnsupportedRequest(
                "BLIS requires an exact backend_version; it cannot resolve AISimulate version aliases"
            )
        return version
    return os.environ.get("BLIS_ENGINE_VERSION", "0.31.0")
