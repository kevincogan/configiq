"""BLIS request orchestration, support gating, and bounded recommendation search."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from aisimulate_core.sdk.perf_database import load_system_spec

from .manifest import (
    BlisUnsupportedRequest,
    ModelResolution,
    SupportManifest,
    load_support_manifest,
)
from .runner import BlisExecutionError, RunRequest, run_blis
from .runtime import BlisRuntime, resolve_runtime
from .scenario import ScenarioRequest, default_engine_version, normalize_weight_quantization, write_scenario

_MAX_RECOMMEND_BUDGET_SECONDS = 60.0
_DEFAULT_RUN_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class EstimateInput:
    model_path: str
    system: str
    backend: str
    backend_version: str | None
    isl: int
    osl: int
    concurrency: int
    tp: int
    pp: int
    dp: int
    moe_tp_size: int | None = None
    moe_ep_size: int | None = None
    max_seq_len: int | None = None
    max_num_seqs: int | None = None
    gpu_memory_utilization: float | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    prefix: int = 0
    prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    model_config_provided: bool = False
    unsupported_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecommendInput:
    model_path: str
    system: str
    backend: str
    backend_version: str | None
    isl: int
    osl: int
    concurrency: int | None
    request_rate: float | None
    ttft: float
    tpot: float
    request_latency: float | None
    prefix: int
    min_candidate_gpus: int | None
    max_candidate_gpus: int | None
    top_n: int
    search_budget_seconds: int
    max_seq_len: int | None = None
    database_mode: str = "HYBRID"
    inclusive_tpot: bool = False
    model_config_provided: bool = False


@dataclass(frozen=True)
class BlisPrediction:
    summary: dict[str, float | None]
    provenance: dict[str, str | None]
    serving_config: dict[str, Any]
    warnings: list[str]


@dataclass(frozen=True)
class BlisRecommendation:
    configs: list[dict[str, Any]]
    search_summary: dict[str, Any]
    provenance: dict[str, str | None]


@dataclass(frozen=True)
class _ResolvedRequest:
    manifest: SupportManifest
    runtime: BlisRuntime
    model: ModelResolution
    hardware: str
    gpus_per_node: int
    engine_version: str


def _resolve_request(model_path: str, system: str, backend: str, backend_version: str | None) -> _ResolvedRequest:
    if backend != "vllm":
        raise BlisUnsupportedRequest(
            f"BLIS scenarios currently model vLLM deployments, not backend {backend!r}; "
            "select vLLM or choose AISimulate explicitly"
        )
    manifest = load_support_manifest()
    model = manifest.resolve_model(model_path)
    hardware = manifest.resolve_hardware(system)
    try:
        system_spec = load_system_spec(system)
    except Exception as exc:
        raise BlisUnsupportedRequest(f"BLIS cannot load the GPU node layout for {system!r}: {exc}") from exc
    node = system_spec.get("node", {}) if isinstance(system_spec, dict) else {}
    gpus_per_node = int(node.get("num_gpus_per_node", 0)) if isinstance(node, dict) else 0
    if gpus_per_node < 1:
        raise BlisUnsupportedRequest(f"BLIS has no single-node GPU layout for {system!r}")
    return _ResolvedRequest(
        manifest=manifest,
        runtime=resolve_runtime(),
        model=model,
        hardware=hardware,
        gpus_per_node=gpus_per_node,
        engine_version=default_engine_version(backend_version),
    )


def _provenance(resolved: _ResolvedRequest, model_path: str, system: str) -> dict[str, str | None]:
    return {
        "simulator": "blis",
        "requested_model": model_path,
        "resolved_model": resolved.model.catalog_name,
        "model_source": resolved.model.origin,
        "system": system,
        "hardware": resolved.hardware,
        "model_config_relation": resolved.model.config_relation,
        "engine_version": resolved.engine_version,
    }


def _model_warnings(resolved: _ResolvedRequest, *, model_config_provided: bool = False) -> list[str]:
    warnings: list[str] = []
    relation = resolved.model.config_relation
    if relation and relation != "identical":
        warnings.append(f"The selected simulator uses a model configuration marked {relation}.")
    if model_config_provided:
        warnings.append("The selected simulator used its resolved catalog model; the supplied model_config was not applied.")
    return warnings


def _weight_quantization(gemm: str | None, moe: str | None) -> tuple[str | None, str | None]:
    normalized_gemm = gemm.strip().lower() if gemm else None
    normalized_moe = moe.strip().lower() if moe else None
    # The default UI MoE mode names the MXFP4 weight format; scenario engines
    # accept the weight format, not the serving kernel implementation.
    if normalized_moe == "w4a16_mxfp4":
        normalized_moe = "mxfp4"
    elif normalized_moe and normalized_moe not in {"fp16", "bf16", "half", "mxfp4", "nvfp4", "fp8"}:
        raise BlisUnsupportedRequest(
            f"BLIS cannot represent the requested MoE kernel quantization {moe!r}"
        )
    if normalized_moe and normalized_gemm != normalized_moe:
        raise BlisUnsupportedRequest("BLIS cannot represent separate GEMM and MoE quantization settings")
    return normalized_gemm, normalized_moe


def _scenario_request(
    resolved: _ResolvedRequest,
    *,
    system: str,
    isl: int,
    osl: int,
    concurrency: int,
    tp: int,
    pp: int,
    dp: int,
    moe_tp_size: int | None,
    moe_ep_size: int | None,
    max_seq_len: int | None,
    max_num_seqs: int | None,
    gpu_memory_utilization: float | None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    prefix_caching: bool | None = None,
    enable_chunked_prefill: bool | None = None,
) -> ScenarioRequest:
    gemm_quant_mode, moe_quant_mode = _weight_quantization(gemm_quant_mode, moe_quant_mode)
    return ScenarioRequest(
        model=resolved.model,
        hardware=resolved.hardware,
        system=system,
        gpus_per_node=resolved.gpus_per_node,
        engine_version=resolved.engine_version,
        tp=tp,
        pp=pp,
        dp=dp,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        enable_expert_parallel=moe_ep_size is not None and moe_ep_size > 1,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        max_num_seqs=max_num_seqs,
        max_model_len=max_seq_len if max_seq_len is not None else isl + osl,
        max_num_batched_tokens=max(8192, isl),
        gpu_memory_utilization=gpu_memory_utilization,
        weight_quantization=gemm_quant_mode,
        moe_quantization=moe_quant_mode,
        cache_dtype=kvcache_quant_mode,
        prefix_caching=prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        fmha_quantization=fmha_quant_mode,
    )


def _metrics_summary(metrics: dict[str, Any], total_gpus: int) -> dict[str, float | None]:
    completed = metrics.get("completed_requests")
    if not isinstance(completed, (int, float)) or completed <= 0:
        dropped = metrics.get("dropped_unservable", 0)
        raise BlisUnsupportedRequest(
            f"BLIS completed no requests (dropped_unservable={dropped}); no performance estimate is available"
        )

    def number(name: str) -> float | None:
        value = metrics.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    summary = {
        "ttft_ms": number("ttft_mean_ms"),
        "tpot_ms": number("itl_mean_ms"),
        "e2e_latency_ms": number("e2e_mean_ms"),
        "output_throughput_tok_s": number("tokens_per_sec"),
        "num_total_gpus": float(total_gpus),
        "power_w": None,
    }
    if summary["ttft_ms"] is None or summary["tpot_ms"] is None:
        raise BlisExecutionError("BLIS completed requests but omitted TTFT or ITL metrics")
    return summary


def _serving_config(scenario: ScenarioRequest) -> dict[str, Any]:
    weight_quantization = normalize_weight_quantization(scenario.weight_quantization)
    return {
        "backend": "vllm",
        "tensor_parallel_size": scenario.tp,
        "max_model_len": scenario.max_model_len or (scenario.isl + scenario.osl),
        "max_num_seqs": scenario.max_num_seqs or scenario.concurrency,
        "max_num_batched_tokens": scenario.max_num_batched_tokens or max(8192, scenario.isl),
        "gpu_memory_utilization": (
            scenario.gpu_memory_utilization
            if scenario.gpu_memory_utilization is not None
            else 0.92
        ),
        # Chunked prefill is not a Scenario field. Leave it unknown rather than
        # claiming a default that the generated document did not specify.
        "enable_chunked_prefill": None,
        "enable_prefix_caching": scenario.prefix_caching if scenario.prefix_caching is not None else True,
        "quantization": weight_quantization or "model default",
    }


def _run_scenario(
    resolved: _ResolvedRequest,
    scenario: ScenarioRequest,
    *,
    prefix: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="configiq-blis-") as scenario_directory:
        filename = write_scenario(scenario_directory, scenario)
        return run_blis(
            resolved.runtime,
            RunRequest(
                scenario=filename,
                scenario_directory=scenario_directory,
                isl=scenario.isl,
                osl=scenario.osl,
                concurrency=scenario.concurrency,
                # BLIS treats concurrency as virtual users and num-requests as
                # the total simulation slice. The API uses a bounded smoke run
                # of one request per virtual user, matching its current input.
                num_requests=scenario.concurrency,
                prefix_tokens=prefix,
            ),
            timeout_seconds=timeout_seconds,
        )


def estimate(request: EstimateInput) -> BlisPrediction:
    if request.unsupported_options:
        options = ", ".join(request.unsupported_options)
        raise BlisUnsupportedRequest(f"BLIS cannot represent these request options: {options}")
    resolved = _resolve_request(request.model_path, request.system, request.backend, request.backend_version)
    if request.prefix > 0 and request.prefix_caching is False:
        raise BlisUnsupportedRequest("a shared prefix requires prefix caching to be enabled in BLIS")
    scenario = _scenario_request(
        resolved,
        system=request.system,
        isl=request.isl,
        osl=request.osl,
        concurrency=request.concurrency,
        tp=request.tp,
        pp=request.pp,
        dp=request.dp,
        moe_tp_size=request.moe_tp_size,
        moe_ep_size=request.moe_ep_size,
        max_seq_len=request.max_seq_len,
        max_num_seqs=request.max_num_seqs,
        gpu_memory_utilization=request.gpu_memory_utilization,
        gemm_quant_mode=request.gemm_quant_mode,
        moe_quant_mode=request.moe_quant_mode,
        kvcache_quant_mode=request.kvcache_quant_mode,
        fmha_quant_mode=request.fmha_quant_mode,
        prefix_caching=True if request.prefix > 0 and request.prefix_caching is None else request.prefix_caching,
        enable_chunked_prefill=request.enable_chunked_prefill,
    )
    metrics = _run_scenario(
        resolved,
        scenario,
        prefix=request.prefix,
        timeout_seconds=float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS)),
    )
    total_gpus = request.tp * request.pp * request.dp
    return BlisPrediction(
        summary=_metrics_summary(metrics, total_gpus),
        provenance=_provenance(resolved, request.model_path, request.system),
        serving_config=_serving_config(scenario),
        warnings=_model_warnings(resolved, model_config_provided=request.model_config_provided) + (
            [f"Simulator marked {int(metrics['dropped_unservable'])} request(s) unservable."]
            if isinstance(metrics.get("dropped_unservable"), (int, float)) and metrics["dropped_unservable"] > 0
            else []
        ),
    )


def recommend(request: RecommendInput) -> BlisRecommendation:
    if request.request_rate is not None:
        raise BlisUnsupportedRequest(
            "BLIS recommendation currently searches concurrency workloads; choose a target concurrency "
            "or select AISimulate for request-rate search"
        )
    if request.database_mode != "HYBRID":
        raise BlisUnsupportedRequest("BLIS does not accept the AISimulate database_mode option")
    concurrency = request.concurrency or 1
    resolved = _resolve_request(request.model_path, request.system, request.backend, request.backend_version)
    total_budget_seconds = min(max(request.search_budget_seconds, 1), int(_MAX_RECOMMEND_BUDGET_SECONDS))
    upper = min(request.max_candidate_gpus or resolved.gpus_per_node, resolved.gpus_per_node)
    lower = request.min_candidate_gpus or 1
    if lower > upper:
        raise BlisUnsupportedRequest(
            f"Requested minimum GPU count {lower} exceeds the effective BLIS search ceiling {upper} "
            f"(configured limit {request.max_candidate_gpus}, single-node hardware limit {resolved.gpus_per_node})"
        )
    candidates: list[int] = []
    candidate = 1
    while candidate <= upper:
        if candidate >= lower:
            candidates.append(candidate)
        candidate *= 2
    started = time.monotonic()
    evaluated: list[int] = []
    feasible: list[dict[str, Any]] = []
    timed_out = False
    execution_failures: list[BlisExecutionError] = []

    for tp in candidates:
        remaining = total_budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            break
        timeout = min(
            float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS)),
            remaining,
        )
        scenario = _scenario_request(
            resolved,
            system=request.system,
            isl=request.isl,
            osl=request.osl,
            concurrency=concurrency,
            tp=tp,
            pp=1,
            dp=1,
            moe_tp_size=None,
            moe_ep_size=None,
            max_seq_len=request.max_seq_len,
            max_num_seqs=concurrency,
            gpu_memory_utilization=None,
            prefix_caching=True if request.prefix > 0 else None,
        )
        try:
            metrics = _run_scenario(resolved, scenario, prefix=request.prefix, timeout_seconds=timeout)
        except BlisExecutionError as exc:
            # A failed/timed-out candidate is an unsuccessful point in the bounded
            # search, not grounds to substitute AISimulate. Keep searching only
            # within the remaining global budget.
            evaluated.append(tp)
            execution_failures.append(exc)
            continue
        evaluated.append(tp)
        completed = metrics.get("completed_requests")
        if not isinstance(completed, (int, float)) or completed <= 0:
            continue
        if metrics.get("dropped_unservable", 0):
            continue
        ttft = metrics.get("ttft_mean_ms")
        itl = metrics.get("itl_mean_ms")
        e2e = metrics.get("e2e_mean_ms")
        if not all(isinstance(value, (int, float)) for value in (ttft, itl, e2e)):
            continue
        if request.request_latency is not None:
            if e2e > request.request_latency:
                continue
        elif ttft > request.ttft or itl > request.tpot:
            continue
        tokens_per_second = metrics.get("tokens_per_sec")
        reported_tpot = (
            (float(ttft) + float(itl) * (request.osl - 1)) / request.osl
            if request.inclusive_tpot and request.osl > 0
            else float(itl)
        )
        feasible.append({
            "total_gpus_needed": tp,
            "replicas_needed": 1,
            "num_total_gpus": tp,
            "tp": tp,
            "pp": 1,
            "dp": 1,
            "bs": concurrency,
            "ttft": float(ttft),
            "tpot": reported_tpot,
            "request_latency": float(e2e),
            "tokens_per_second": float(tokens_per_second) if isinstance(tokens_per_second, (int, float)) else None,
            "tokens_per_second_per_gpu": (
                float(tokens_per_second) / tp if isinstance(tokens_per_second, (int, float)) and tp > 0 else None
            ),
            "tokens_per_second_per_user": (
                float(tokens_per_second) / concurrency
                if isinstance(tokens_per_second, (int, float)) and concurrency > 0
                else None
            ),
            "concurrency": concurrency,
            "model": request.model_path,
            "system": request.system,
            "backend": request.backend,
            "backend_version": request.backend_version,
            "mode": "agg",
            "simulator": "blis",
            "provenance": _provenance(resolved, request.model_path, request.system),
            "warnings": _model_warnings(resolved, model_config_provided=request.model_config_provided),
        })
        if len(feasible) >= request.top_n:
            break

    if not feasible and candidates and len(execution_failures) == len(evaluated):
        raise execution_failures[-1]

    return BlisRecommendation(
        configs=feasible,
        provenance=_provenance(resolved, request.model_path, request.system),
        search_summary={
            "strategy": "tensor_parallel_only",
            "candidate_gpus": candidates,
            "evaluated_gpus": evaluated,
            "max_candidates": len(candidates),
            "configured_max_gpus": request.max_candidate_gpus or resolved.gpus_per_node,
            "hardware_max_gpus": resolved.gpus_per_node,
            "effective_max_gpus": upper,
            "time_budget_seconds": total_budget_seconds,
            "budget_exhausted": timed_out,
        },
    )
