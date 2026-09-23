# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulators REST API.

Minimal service wrapping the aisimulate SDK for GPU recommendation,
single-point performance estimation, and memory estimation.
See docs/api/openapi.yaml for the full spec.
"""

import argparse
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from types import SimpleNamespace

import pandas as pd
import uvicorn
from configiq.systems import load_device_names_from_perf_data, supported_systems
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, model_validator

from aisimulate_core.sdk.common import get_default_models
from aisimulate_core.sdk.errors import NoFeasibleConfigError
from aisimulate_core.sdk.memory import estimate_kv_cache
from aisimulate_core.sdk.perf_database import load_system_spec
from aisimulate_core.sdk.utils import get_model_config_from_model_path
from aisimulate.sdk.task_v2 import Task

# Optional observability + MCP, provided by the shared configiq package
# (configiq[otel,mcp]). Kept out of the base install/container because otel
# tracing opens an OTLP exporter eagerly at import; every use is guarded so the
# service runs unchanged when the extras are absent.
try:
    from configiq import observability
    _OBS = True
except ImportError:
    _OBS = False

try:
    from configiq import mcp as mcp_support
    _MCP = True
except ImportError:
    _MCP = False

logger = logging.getLogger(__name__)

# ─── Pydantic models ────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None], description="Backend version.")
    target_request_rate: float | None = Field(default=None, examples=[None], description="Target req/s.")
    target_concurrency: float | None = Field(default=None, examples=[32], description="Target concurrent users.")
    isl: int = Field(default=4000, description="Input sequence length.")
    osl: int = Field(default=1000, description="Output sequence length.")
    max_seq_len: int | None = Field(default=None, gt=0, description="Maximum sequence length for KV cache allocation. Defaults to isl + osl.")
    prefill_max_seq_len: int | None = Field(default=None, gt=0, description="Optional maximum sequence length for disaggregated prefill workers.")
    decode_max_seq_len: int | None = Field(default=None, gt=0, description="Optional maximum sequence length for disaggregated decode workers.")
    ttft: float = Field(default=2000.0, description="TTFT target (ms).")
    tpot: float = Field(default=30.0, description="TPOT target (ms).")
    request_latency: float | None = Field(default=None, description="E2E latency target (ms).")
    prefix: int = Field(default=0, description="Prefix cache length.")
    database_mode: str = Field(default="HYBRID", description="Perf database mode.")
    top_n: int = Field(default=5, ge=1, le=20, examples=[2], description="Number of configs to return.")
    inclusive_tpot: bool = Field(
        default=False,
        description="Report TPOT as (ttft + tpot * (osl - 1)) / osl, spreading TTFT across all output tokens. "
        "Useful for comparing with benchmarks that report inclusive TPOT (e.g. GuideLLM).",
    )
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        examples=[None],
        description="Pre-fetched HuggingFace model config.json. Skips HF download. "
        "Leave null to let the service resolve the model from HuggingFace.",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def exactly_one_load_target(self):
        has_rate = self.target_request_rate is not None
        has_conc = self.target_concurrency is not None
        if has_rate == has_conc:
            raise ValueError("Exactly one of target_request_rate or target_concurrency must be provided.")
        return self


class EstimateRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None], description="Backend version.")
    isl: int = Field(default=4000, description="Input sequence length.")
    osl: int = Field(default=1000, description="Output sequence length.")
    max_seq_len: int | None = Field(default=None, gt=0, description="Maximum sequence length for KV cache allocation.")
    prefill_max_seq_len: int | None = Field(default=None, gt=0, description="Prefill worker sequence-length override.")
    decode_max_seq_len: int | None = Field(default=None, gt=0, description="Decode worker sequence-length override.")
    tp_size: int = Field(default=1, description="Tensor parallel size.")
    pp_size: int = Field(default=1, description="Pipeline parallel size.")
    batch_size: int = Field(default=128, description="Batch size (max concurrent requests).")
    database_mode: str = Field(default="HYBRID", description="Perf database mode.")
    gemm_quant_mode: str | None = Field(default=None)
    kvcache_quant_mode: str | None = Field(default=None)
    fmha_quant_mode: str | None = Field(default=None)
    moe_tp_size: int | None = Field(default=None)
    moe_ep_size: int | None = Field(default=None)
    attention_dp_size: int = Field(default=1)
    # Serving mode. When 'disagg', the prefill_*/decode_* fields below drive
    # separate prefill and decode pools; the top-level tp/pp/batch fields act
    # as fallbacks for any per-role field left unset.
    mode: Literal["agg", "disagg"] = Field(default="agg", description="Serving mode: 'agg' or 'disagg'.")
    decode_system: str | None = Field(default=None, description="GPU system for disagg decode workers; defaults to `system`.")
    prefill_tp_size: int | None = Field(default=None)
    prefill_pp_size: int | None = Field(default=None)
    prefill_moe_tp_size: int | None = Field(default=None)
    prefill_moe_ep_size: int | None = Field(default=None)
    prefill_batch_size: int | None = Field(default=None)
    prefill_num_workers: int | None = Field(default=None)
    decode_tp_size: int | None = Field(default=None)
    decode_pp_size: int | None = Field(default=None)
    decode_moe_tp_size: int | None = Field(default=None)
    decode_moe_ep_size: int | None = Field(default=None)
    decode_batch_size: int | None = Field(default=None)
    decode_num_workers: int | None = Field(default=None)
    inclusive_tpot: bool = Field(
        default=False,
        description="Report TPOT as (ttft + tpot * (osl - 1)) / osl, spreading TTFT across all output tokens. "
        "Useful for comparing with benchmarks that report inclusive TPOT (e.g. GuideLLM).",
    )
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        examples=[None],
        description="Pre-fetched HuggingFace model config.json. Skips HF download. "
        "Leave null to let the service resolve the model from HuggingFace.",
    )

    model_config = {"populate_by_name": True}


class MemoryBreakdown(BaseModel):
    weights_bytes: int = 0
    activations_bytes: int = 0
    runtime_overhead_bytes: int = 0
    comm_overhead_bytes: int = 0
    kv_cache_bytes: int = 0


class ServingConfig(BaseModel):
    backend: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    enable_chunked_prefill: bool
    enable_prefix_caching: bool
    quantization: str


class WorkerConfig(BaseModel):
    """Parallelism and serving config for one worker role in a disagg deployment."""
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    cp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    num_workers: int | None = None
    batch_size: int | None = None
    memory_gb: float | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    backend_version: str | None = None
    memory_breakdown: MemoryBreakdown | None = None


class RecommendConfig(BaseModel):
    total_gpus_needed: int | None = None
    replicas_needed: int | None = None
    num_total_gpus: int | None = None
    # Parallelism (agg only; None for disagg — see prefill/decode_config)
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    cp: int | None = None
    bs: int | None = None
    ttft: float | None = None
    tpot: float | None = None
    request_latency: float | None = None
    concurrency: int | None = None
    request_rate: float | None = None
    tokens_per_second: float | None = None
    tokens_per_second_per_gpu: float | None = None
    tokens_per_second_per_user: float | None = None
    memory: float | None = None
    model: str | None = None
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    power_w: float | None = None
    # Optional detail sections (include=config / include=memory)
    serving_config: ServingConfig | None = None
    memory_breakdown: MemoryBreakdown | None = None
    # Disagg detail (present for disagg results)
    prefill_config: WorkerConfig | None = None
    decode_config: WorkerConfig | None = None
    mode: str = "agg"


class RecommendResponse(BaseModel):
    configs: list[RecommendConfig]
    chosen_mode: str


class EstimateResponse(BaseModel):
    """Single-point performance estimate for a given parallelism configuration."""
    ttft: float
    tpot: float
    request_latency: float | None = None
    tokens_per_second: float | None = None
    tokens_per_second_per_gpu: float | None = None
    tokens_per_second_per_user: float | None = None
    memory: float | None = None
    concurrency: int | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    power_w: float | None = None
    serving_config: ServingConfig | None = None
    memory_breakdown: MemoryBreakdown | None = None
    # Serving mode + disagg detail (present when mode='disagg')
    mode: str = "agg"
    prefill_config: WorkerConfig | None = None
    decode_config: WorkerConfig | None = None


class MemoryRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None])
    max_num_tokens: int = Field(default=8192)
    max_batch_size: int = Field(default=128)
    memory_fraction_kind: str = Field(default="of_total")
    memory_fraction_value: float = Field(default=1.0, ge=0.0, le=1.0)
    tp_size: int = Field(default=1)
    pp_size: int = Field(default=1)
    attention_dp_size: int = Field(default=1)
    moe_tp_size: int | None = Field(default=None)
    moe_ep_size: int | None = Field(default=None)
    gemm_quant_mode: str | None = Field(default=None)
    moe_quant_mode: str | None = Field(default=None)
    kvcache_quant_mode: str | None = Field(default=None)
    fmha_quant_mode: str | None = Field(default=None)
    comm_quant_mode: str | None = Field(default=None)
    tolerance_fraction: float | None = Field(default=None)
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        examples=[None],
        description="Pre-fetched HuggingFace model config.json. Skips HF download. "
        "Leave null to let the service resolve the model from HuggingFace.",
    )

    model_config = {"populate_by_name": True}


class MemoryResponse(BaseModel):
    total_gpu_capacity_bytes: int
    total_kv_size_bytes: int
    kv_size_per_token_bytes: int
    total_kv_size_tokens: int
    source: str
    memory_breakdown: MemoryBreakdown
    tolerance_adjusted: dict[str, Any] | None = None


class SystemDetail(BaseModel):
    id: str
    name: str
    vendor: str
    architecture: str
    memory_bytes: int
    tdp_watts: float
    gpus_per_node: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

_COLUMN_MAP = {
    "tokens/s": "tokens_per_second",
    "tokens/s/gpu": "tokens_per_second_per_gpu",
    "tokens/s/user": "tokens_per_second_per_user",
    "num_total_gpus": "num_total_gpus",
    "version": "backend_version",
}

_INT_FIELDS = frozenset({
    "total_gpus_needed", "replicas_needed", "num_total_gpus",
    "tp", "pp", "dp", "moe_tp", "moe_ep", "cp", "bs", "concurrency",
})

_SM_ARCHITECTURE = {
    70: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    89: "ada-lovelace",
    90: "hopper",
    100: "blackwell",
    103: "blackwell",
    120: "blackwell",
}

_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


class _noop_context:
    def __init__(self, model_path: str):
        self._path = model_path

    def __enter__(self):
        return self._path

    def __exit__(self, *args):
        pass


class _tempdir_context:
    def __init__(self, model_path: str, config_dict: dict):
        self._original = model_path
        self._config = config_dict
        self._tmpdir: Any = None

    def __enter__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        config_path = Path(self._tmpdir.name) / "config.json"
        config_path.write_text(json.dumps(self._config))
        return self._tmpdir.name

    def __exit__(self, *args):
        if self._tmpdir:
            self._tmpdir.cleanup()


def _with_model_config(model_path: str, config_dict: dict | None):
    # An empty dict carries no usable config — treat anything falsy as
    # "not supplied" so the SDK resolves the model itself instead of loading a
    # bogus config.json. (A non-empty but invalid config, such as Swagger UI's
    # {"additionalProp1": {}} placeholder, still reaches the SDK and surfaces as
    # a clear 422 via the KeyError branch in _common_error_handler.)
    if not config_dict:
        return _noop_context(model_path)
    return _tempdir_context(model_path, config_dict)


def _coerce_int(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _coerce_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _inclusive_tpot(ttft: float | None, tpot: float | None, osl: int) -> float | None:
    """Spread TTFT across all output tokens: (ttft + tpot * (osl - 1)) / osl.

    Presentation-only transform for benchmark-comparable TPOT (e.g. GuideLLM).
    Returns ``tpot`` unchanged when inputs are insufficient.
    """
    if ttft is None or tpot is None or osl <= 0:
        return tpot
    return (ttft + tpot * (osl - 1)) / osl


def _aisimulate_runner_factory():
    from aisimulate.stack import resolve_runner_factory

    return resolve_runner_factory("engine")


def _aisimulate_prediction_config(req: EstimateRequest) -> dict[str, Any]:
    from aisimulate.config.cli import CorePredictionConfig

    def worker(role: str) -> dict[str, Any]:
        if role == "agg":
            tp, pp, dp, moe_tp, moe_ep, batch = (
                req.tp_size, req.pp_size, req.attention_dp_size, req.moe_tp_size, req.moe_ep_size, req.batch_size
            )
            context = req.max_seq_len or req.isl + req.osl
        else:
            tp = req.prefill_tp_size if role == "prefill" else req.decode_tp_size
            pp = req.prefill_pp_size if role == "prefill" else req.decode_pp_size
            dp = req.attention_dp_size
            moe_tp = (
                req.prefill_moe_tp_size if role == "prefill" else req.decode_moe_tp_size
            ) or req.moe_tp_size
            moe_ep = (
                req.prefill_moe_ep_size if role == "prefill" else req.decode_moe_ep_size
            ) or req.moe_ep_size
            batch = req.prefill_batch_size if role == "prefill" else req.decode_batch_size
            context = (
                req.prefill_max_seq_len if role == "prefill" else req.decode_max_seq_len
            ) or req.max_seq_len or req.isl + req.osl
        result = {
            "parallelism": {
                "replicas": (req.prefill_num_workers if role == "prefill" else req.decode_num_workers) or 1,
                "tensor": tp or req.tp_size,
                "pipeline": pp or req.pp_size,
                "attention_data": dp or req.attention_dp_size,
                "moe_tensor": moe_tp or 1,
                "moe_expert": moe_ep or 1,
            },
            "scheduler": {"max_sequences": batch or req.batch_size},
        }
        if role != "agg":
            result["context_length"] = context
        return result

    mode = "disaggregated" if req.mode == "disagg" else "aggregated"
    workers = {"prefill": worker("prefill"), "decode": worker("decode")} if req.mode == "disagg" else {
        "aggregated": worker("agg")
    }
    raw = {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": req.isl, "output_tokens": req.osl},
            "load": {"type": "concurrency", "concurrency": req.batch_size},
            "stop": {"requests": 1},
        },
        "engine": {
            "mode": mode,
            "model": req.model_path,
            "hardware": req.system,
            "backend": req.backend,
            "backend_version": req.backend_version,
            "database_mode": req.database_mode,
            "context_length": req.max_seq_len or req.isl + req.osl,
            "workers": workers,
        },
    }
    return CorePredictionConfig.model_validate(raw)


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return _coerce_float(value)
    return None


def _run_aisimulate_recommendation(req: RecommendRequest):
    from aisimulate.sdk.picking import pick_load_match

    with _with_model_config(req.model_path, req.model_config_data) as effective_path:
        spec = load_system_spec(req.system)
        gpus_per_node = int(spec["node"]["num_gpus_per_node"])
        budgets = []
        budget = gpus_per_node
        while budget <= 64:
            budgets.append(budget)
            budget *= 2

        for total_gpus in budgets:
            candidates = [1 << i for i in range(total_gpus.bit_length()) if (1 << i) <= total_gpus]
            common = {
                "isl": req.isl,
                "osl": req.osl,
                "ttft": req.ttft,
                "tpot": req.tpot,
                "request_latency": req.request_latency,
                "prefix": req.prefix,
                "total_gpus": total_gpus,
                "database_mode": req.database_mode,
            }
            tasks = {
                "agg": Task(
                    serving_mode="agg",
                    model_path=effective_path,
                    system_name=req.system,
                    backend_name=req.backend,
                    backend_version=req.backend_version,
                    agg_num_gpu_candidates=candidates,
                    **common,
                ),
                "disagg": Task(
                    serving_mode="disagg",
                    prefill_model_path=effective_path,
                    decode_model_path=effective_path,
                    prefill_system_name=req.system,
                    decode_system_name=req.system,
                    prefill_backend_name=req.backend,
                    decode_backend_name=req.backend,
                    prefill_backend_version=req.backend_version,
                    decode_backend_version=req.backend_version,
                    prefill_num_gpu_candidates=candidates,
                    decode_num_gpu_candidates=candidates,
                    **common,
                ),
            }

            def run_task(item):
                name, task = item
                return name, task, task.run()

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(run_task, tasks.items()))

            best_configs = {}
            best_scores = {}
            for name, task, pareto_df in results:
                if pareto_df is None or pareto_df.empty:
                    continue
                picked = pick_load_match(
                    pareto_df=pareto_df,
                    serving_mode=task.serving_mode,
                    target_tpot=task.tpot,
                    target_request_latency=task.request_latency,
                    target_request_rate=req.target_request_rate,
                    target_concurrency=req.target_concurrency,
                    top_n=req.top_n,
                )
                best = picked.get("best_config_df")
                if best is not None and not best.empty:
                    best_configs[name] = best
                    best_scores[name] = picked.get("best_throughput", 0.0)

            if best_configs:
                chosen = max(best_scores, key=best_scores.get)
                return SimpleNamespace(best_configs=best_configs, chosen_exp=chosen)

        return SimpleNamespace(best_configs={}, chosen_exp="none")


def _run_aisimulate_prediction(req: EstimateRequest, include: set[str]):
    from aisimulate.predict import run_prediction
    from aisimulate.sweeper.replay import ReplayOutputRequirements

    with _with_model_config(req.model_path, req.model_config_data) as effective_path:
        prediction_req = req.model_copy(update={"model_path": effective_path})
        return run_prediction(
            _aisimulate_prediction_config(prediction_req),
            stack="engine",
            runner_factory=_aisimulate_runner_factory(),
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_memory_diagnostics="memory" in include,
            ),
        )


def _worker_config_from_row(row: pd.Series, prefix: str, req: RecommendRequest) -> WorkerConfig | None:
    def g(col: str) -> Any:
        v = row.get(f"({prefix}){col}")
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    tp = _coerce_int(g("tp"))
    if tp is None:
        return None
    return WorkerConfig(
        tp=tp,
        pp=_coerce_int(g("pp")),
        dp=_coerce_int(g("dp")),
        cp=_coerce_int(g("cp")),
        moe_tp=_coerce_int(g("moe_tp")),
        moe_ep=_coerce_int(g("moe_ep")),
        num_workers=_coerce_int(g("workers")),
        batch_size=_coerce_int(g("bs")),
        memory_gb=_coerce_float(g("memory")),
        gemm=g("gemm"),
        kvcache=g("kvcache"),
        fmha=g("fmha"),
        moe=g("moe"),
        comm=g("comm"),
        backend_version=row.get(f"({prefix})version") or req.backend_version,
    )


def _row_to_config(row: pd.Series, req: RecommendRequest) -> RecommendConfig:
    is_disagg = "(p)tp" in row.index

    d: dict[str, Any] = {}
    for col, val in row.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        col_str = str(col)
        if col_str.startswith(("(p)", "(d)", "(e)")):
            continue
        key = _COLUMN_MAP.get(col_str, col_str)
        if isinstance(val, float) and val == int(val) and key in _INT_FIELDS:
            val = int(val)
        d[key] = val

    d.setdefault("system", req.system)
    d.setdefault("backend", req.backend)
    d.setdefault("backend_version", req.backend_version)

    cfg = RecommendConfig.model_validate(d)

    if is_disagg:
        cfg.prefill_config = _worker_config_from_row(row, "p", req)
        cfg.decode_config = _worker_config_from_row(row, "d", req)
        if cfg.memory is None:
            # (p)/(d)/(e)memory are each per-GPU peak usage (GB) for that worker
            # type — each is checked against a single GPU's capacity, so they are
            # NOT additive. The meaningful single figure is the worst-case
            # per-GPU across all pools (prefill, decode, and encode).
            phase_mems = [
                m for m in (
                    _coerce_float(row.get("(p)memory")),
                    _coerce_float(row.get("(d)memory")),
                    _coerce_float(row.get("(e)memory")),
                )
                if m is not None
            ]
            if phase_mems:
                cfg.memory = max(phase_mems)

    return cfg


def _build_serving_config(
    backend: str,
    tp: int,
    isl: int,
    osl: int,
    concurrency: int,
    gemm: str | None,
    prefix: int = 0,
    max_seq_len: int | None = None,
) -> ServingConfig:
    quant_map = {"fp8": "fp8", "fp8_block": "fp8", "int8": "int8"}
    quantization = quant_map.get(gemm or "", "auto")
    return ServingConfig(
        backend=backend,
        tensor_parallel_size=tp,
        max_model_len=max_seq_len or isl + osl,
        max_num_seqs=min(concurrency, 256),
        gpu_memory_utilization=_DEFAULT_GPU_MEMORY_UTILIZATION,
        enable_chunked_prefill=isl >= 4096 or concurrency >= 64,
        enable_prefix_caching=prefix > 0,
        quantization=quantization,
    )


def _build_memory_breakdown(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None,
    tp: int,
    pp: int,
    isl: int,
    osl: int,
    concurrency: int,
    gemm_quant: str | None = None,
    kvcache_quant: str | None = None,
    moe_tp: int | None = None,
    moe_ep: int | None = None,
    max_seq_len: int | None = None,
) -> MemoryBreakdown | None:
    try:
        raw = estimate_kv_cache(
            model_path=model_path,
            system=system,
            backend=backend,
            backend_version=backend_version,
            max_num_tokens=max_seq_len or isl + osl,
            max_batch_size=concurrency,
            memory_fraction_kind="of_total",
            memory_fraction_value=_DEFAULT_GPU_MEMORY_UTILIZATION,
            tp_size=tp,
            pp_size=pp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            gemm_quant_mode=gemm_quant if gemm_quant and gemm_quant != "half" else None,
            kvcache_quant_mode=kvcache_quant if kvcache_quant and kvcache_quant != "half" else None,
        )
    except Exception:
        logger.debug("memory breakdown unavailable for %s on %s", model_path, system)
        return None

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryBreakdown(
        weights_bytes=int(breakdown.get("weights_bytes", 0)),
        activations_bytes=int(breakdown.get("activations_bytes", 0)),
        runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
        comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
        kv_cache_bytes=int(raw.get("total_kv_size_bytes", 0)),
    )


def _common_error_handler(e: Exception, op: str, model_path: str, backend: str, system: str) -> None:
    msg = str(e)
    if isinstance(e, NoFeasibleConfigError):
        raise HTTPException(status_code=422, detail=msg)
    if isinstance(e, KeyError):
        # A bare KeyError here almost always means a supplied model_config is
        # incomplete (e.g. missing 'architectures'); str(KeyError('x')) is "'x'".
        detail = (
            f"Invalid or incomplete model_config for model={model_path}: missing key {msg}. "
            "Omit model_config (or send null) to resolve the model from HuggingFace."
        )
        raise HTTPException(status_code=422, detail=detail)
    if isinstance(e, (ValueError, AttributeError)):
        if "system_spec" in msg or "NoneType" in msg or "unsupported model" in msg.lower():
            detail = f"No performance data available for model={model_path}, backend={backend}, system={system}."
            raise HTTPException(status_code=422, detail=detail)
        raise HTTPException(status_code=422, detail=msg)
    logger.exception("%s failed", op)
    raise HTTPException(status_code=500, detail=msg)


def _architecture_from_sm(sm_version: int) -> str:
    sm_arch = _SM_ARCHITECTURE.get(sm_version, f"sm_{sm_version}")
    if sm_arch == f"sm_{sm_version}":
        return "Other"
    return sm_arch


# Cache of system_id -> vendor device name, populated at startup from the
# aisimulate SDK (via configiq.systems). aicostings loads the same map from
# the same source so the two services never drift on GPU naming.
# Only systems with genuine perf-data display names appear here; systems without
# perf data are absent and get hidden from /systems (see get_systems). Startup
# fails outright if this ends up empty, so the map is never empty at request time.
_DEVICE_DISPLAY_NAMES: dict[str, str] = {}


def _parse_include(include: str | None) -> set[str]:
    if not include:
        return set()
    return {s.strip().lower() for s in include.split(",")}


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AISimulators API",
    description="GPU recommendation, performance estimation, and memory estimation for LLM inference.",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

# Initialize OpenTelemetry (tracing + metrics + HTTP middleware) if the optional
# extra is present.
if _OBS:
    observability.enable(app, service_name="aisimulators", service_version="1.0.0",
                         meter_name="aisimulators.api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose the API as MCP tools if the optional extra is present.
if _MCP:
    mcp_support.mount(app, name="aisimulators",
                      description="GPU recommendation and performance estimation for LLM inference")
else:
    logger.info("MCP server unavailable (install with: pip install '.[mcp]')")


@app.on_event("startup")
def startup_event():
    """Load GPU display names from the aisimulate SDK at startup.

    Systems without a loaded display name (no perf benchmark data) are excluded
    from /systems responses (see get_systems). If no display names load at all,
    startup fails — the API refuses to serve without valid perf data. Logs
    coverage gaps for ops visibility.
    """
    global _DEVICE_DISPLAY_NAMES
    _DEVICE_DISPLAY_NAMES = load_device_names_from_perf_data()
    if not _DEVICE_DISPLAY_NAMES:
        raise RuntimeError(
            "No GPU display names loaded from perf data; refusing to start "
            "without valid performance data."
        )

    # Log which supported systems lack perf data (they are hidden from /systems).
    missing = supported_systems() - set(_DEVICE_DISPLAY_NAMES.keys())
    if missing:
        logger.warning(f"No display names for {len(missing)} GPU systems: {missing}")


@app.post("/recommend")
def post_recommend(
    req: RecommendRequest,
    include: str | None = Query(default=None, examples=["config,memory"], description="Comma-separated extras: config, memory."),
):
    """Find optimal GPU configuration for a workload."""
    try:
        result = _run_aisimulate_recommendation(req)
    except (ValueError, AttributeError, Exception) as e:
        _common_error_handler(e, "recommend", req.model_path, req.backend, req.system)

    best = result.best_configs.get(result.chosen_exp)
    chosen = result.chosen_exp
    if best is None or best.empty:
        raise HTTPException(status_code=422, detail="No configuration meets the specified requirements.")

    includes = _parse_include(include)
    want_config = "config" in includes
    want_memory = "memory" in includes
    is_disagg = chosen.startswith("disagg") if chosen else False

    configs = []
    with _with_model_config(req.model_path, req.model_config_data) as effective_path:
        for _, row in best.head(req.top_n).iterrows():
            cfg = _row_to_config(row, req)
            if req.inclusive_tpot:
                cfg.tpot = _inclusive_tpot(cfg.ttft, cfg.tpot, req.osl)
            backend = cfg.backend or req.backend
            bv = cfg.backend_version or req.backend_version

            if is_disagg:
                if want_memory:
                    for worker in [cfg.prefill_config, cfg.decode_config]:
                        if worker and worker.tp:
                            worker.memory_breakdown = _build_memory_breakdown(
                                effective_path, req.system, backend,
                                bv or worker.backend_version,
                                worker.tp or 1, worker.pp or 1,
                                req.isl, req.osl, worker.num_workers or 1,
                                worker.gemm if worker.gemm and worker.gemm != "half" else None,
                                worker.kvcache if worker.kvcache and worker.kvcache != "half" else None,
                                worker.moe_tp, worker.moe_ep,
                            )
            else:
                tp = cfg.tp or 1
                concurrency = cfg.concurrency or 128
                if want_config:
                    cfg.serving_config = _build_serving_config(
                        backend, tp, req.isl, req.osl, concurrency, cfg.gemm, req.prefix,
                        max_seq_len=req.max_seq_len,
                    )
                if want_memory:
                    cfg.memory_breakdown = _build_memory_breakdown(
                        effective_path, req.system, backend, bv,
                        tp, cfg.pp or 1, req.isl, req.osl, concurrency,
                        cfg.gemm, cfg.kvcache, cfg.moe_tp, cfg.moe_ep,
                        max_seq_len=req.max_seq_len,
                    )
            configs.append(cfg)

    return RecommendResponse(configs=configs, chosen_mode=chosen)


@app.post("/estimate")
def post_estimate(
    req: EstimateRequest,
    include: str | None = Query(default=None, examples=["config,memory"], description="Comma-separated extras: config, memory."),
):
    """Single-point performance estimate for a given parallelism configuration.

    Given a model, GPU system, backend, and explicit parallelism settings
    (TP/PP/batch_size), returns predicted TTFT, TPOT, throughput, and memory.
    Use this when you know your deployment configuration and want to predict
    its performance. Use /recommend when you want to find the optimal config.
    """
    is_disagg = req.mode == "disagg"

    # Resolve per-role parallelism, falling back to the top-level agg fields.
    p_tp = req.prefill_tp_size or req.tp_size
    p_pp = req.prefill_pp_size or req.pp_size
    p_bs = req.prefill_batch_size or req.batch_size
    p_workers = req.prefill_num_workers or 1
    p_moe_tp = req.prefill_moe_tp_size or req.moe_tp_size
    p_moe_ep = req.prefill_moe_ep_size or req.moe_ep_size
    d_tp = req.decode_tp_size or req.tp_size
    d_pp = req.decode_pp_size or req.pp_size
    d_bs = req.decode_batch_size or req.batch_size
    d_workers = req.decode_num_workers or 1
    d_moe_tp = req.decode_moe_tp_size or req.moe_tp_size
    d_moe_ep = req.decode_moe_ep_size or req.moe_ep_size

    try:
        prediction = _run_aisimulate_prediction(req, _parse_include(include))
    except (ValueError, AttributeError, Exception) as e:
        _common_error_handler(e, "estimate", req.model_path, req.backend, req.system)

    includes = _parse_include(include)

    tpot = _metric(prediction.summary, "tpot_ms", "mean_tpot_ms", "itl_ms")
    if req.inclusive_tpot:
        tpot = _inclusive_tpot(_metric(prediction.summary, "ttft_ms", "mean_ttft_ms"), tpot, req.osl)

    resp = EstimateResponse(
        ttft=_metric(prediction.summary, "ttft_ms", "mean_ttft_ms") or 0.0,
        tpot=tpot,
        request_latency=_metric(prediction.summary, "e2e_latency_ms", "mean_e2e_latency_ms"),
        tokens_per_second=_metric(prediction.summary, "output_throughput_tok_s", "tokens_per_second"),
        tokens_per_second_per_gpu=_metric(prediction.summary, "output_throughput_tok_s_per_gpu"),
        tokens_per_second_per_user=_metric(prediction.summary, "output_throughput_tok_s_per_user"),
        memory=None,
        concurrency=req.batch_size,
        tp=None if is_disagg else req.tp_size,
        pp=None if is_disagg else req.pp_size,
        dp=None if is_disagg else req.attention_dp_size,
        system=req.system,
        backend=req.backend,
        backend_version=req.backend_version,
        gemm=req.gemm_quant_mode,
        kvcache=req.kvcache_quant_mode,
        power_w=_metric(prediction.summary, "power_w"),
        mode=req.mode,
    )

    if is_disagg:
        # (p)/(d)memory are per-GPU peaks; the single figure is the worst case.
        p_mem = None
        d_mem = None
        phase_mems = [m for m in (p_mem, d_mem) if m is not None]
        if phase_mems:
            resp.memory = max(phase_mems)

        gemm_q = req.gemm_quant_mode if req.gemm_quant_mode and req.gemm_quant_mode != "half" else None
        kv_q = req.kvcache_quant_mode if req.kvcache_quant_mode and req.kvcache_quant_mode != "half" else None
        want_memory = "memory" in includes

        resp.prefill_config = WorkerConfig(
            tp=p_tp, pp=p_pp, moe_tp=p_moe_tp, moe_ep=p_moe_ep,
            num_workers=p_workers, batch_size=p_bs, memory_gb=p_mem,
            backend_version=resp.backend_version,
        )
        resp.decode_config = WorkerConfig(
            tp=d_tp, pp=d_pp, moe_tp=d_moe_tp, moe_ep=d_moe_ep,
            num_workers=d_workers, batch_size=d_bs, memory_gb=d_mem,
            backend_version=resp.backend_version,
        )

        if want_memory:
            # The decode pool may run on a different GPU system than prefill.
            decode_sys = req.decode_system or req.system
            with _with_model_config(req.model_path, req.model_config_data) as effective_path:
                for worker, worker_sys in (
                    (resp.prefill_config, req.system),
                    (resp.decode_config, decode_sys),
                ):
                    if worker and worker.tp:
                        worker.memory_breakdown = _build_memory_breakdown(
                            effective_path, worker_sys, req.backend, req.backend_version,
                            worker.tp, worker.pp or 1, req.isl, req.osl,
                            worker.batch_size or req.batch_size,
                            gemm_q, kv_q, worker.moe_tp, worker.moe_ep,
                            max_seq_len=(req.prefill_max_seq_len if worker is resp.prefill_config else req.decode_max_seq_len)
                            or req.max_seq_len,
                        )
        return resp

    if "config" in includes:
        resp.serving_config = _build_serving_config(
            resp.backend or req.backend, req.tp_size, req.isl, req.osl,
            req.batch_size, resp.gemm, 0, max_seq_len=req.max_seq_len,
        )

    if "memory" in includes:
        with _with_model_config(req.model_path, req.model_config_data) as effective_path:
            resp.memory_breakdown = _build_memory_breakdown(
                effective_path, req.system, req.backend, req.backend_version,
                req.tp_size, req.pp_size, req.isl, req.osl, req.batch_size,
                req.gemm_quant_mode if req.gemm_quant_mode and req.gemm_quant_mode != "half" else None,
                req.kvcache_quant_mode if req.kvcache_quant_mode and req.kvcache_quant_mode != "half" else None,
                req.moe_tp_size, req.moe_ep_size, max_seq_len=req.max_seq_len,
            )

    return resp


@app.post("/memory", response_model=MemoryResponse)
def post_memory(req: MemoryRequest):
    """Estimate GPU memory breakdown for a model configuration."""
    try:
        with _with_model_config(req.model_path, req.model_config_data) as effective_path:
            raw = estimate_kv_cache(
                model_path=effective_path,
                system=req.system,
                backend=req.backend,
                backend_version=req.backend_version,
                max_num_tokens=req.max_num_tokens,
                max_batch_size=req.max_batch_size,
                memory_fraction_kind=req.memory_fraction_kind,
                memory_fraction_value=req.memory_fraction_value,
                tp_size=req.tp_size,
                pp_size=req.pp_size,
                attention_dp_size=req.attention_dp_size,
                moe_tp_size=req.moe_tp_size,
                moe_ep_size=req.moe_ep_size,
                gemm_quant_mode=req.gemm_quant_mode,
                moe_quant_mode=req.moe_quant_mode,
                kvcache_quant_mode=req.kvcache_quant_mode,
                fmha_quant_mode=req.fmha_quant_mode,
                comm_quant_mode=req.comm_quant_mode,
                tolerance_fraction=req.tolerance_fraction,
            )
    except (ValueError, AttributeError, Exception) as e:
        _common_error_handler(e, "memory", req.model_path, req.backend, req.system)

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryResponse(
        total_gpu_capacity_bytes=raw["total_gpu_capacity_bytes"],
        total_kv_size_bytes=raw["total_kv_size_bytes"],
        kv_size_per_token_bytes=raw["kv_size_per_token_bytes"],
        total_kv_size_tokens=raw["total_kv_size_tokens"],
        source=raw.get("source", "unknown"),
        memory_breakdown=MemoryBreakdown(
            weights_bytes=int(breakdown.get("weights_bytes", 0)),
            activations_bytes=int(breakdown.get("activations_bytes", 0)),
            runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
            comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
            kv_cache_bytes=raw["total_kv_size_bytes"],
        ),
        tolerance_adjusted=raw.get("tolerance_adjusted"),
    )


def _model_specs(model_id: str) -> dict:
    """Return metadata for a single catalog model."""
    try:
        cfg = get_model_config_from_model_path(model_id)
    except Exception:
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}

    def _pos_int(val) -> int | None:
        return int(val) if val and int(val) > 0 else None

    return {
        "id": model_id,
        "num_experts": _pos_int(cfg.get("num_experts")),
        "num_experts_per_tok": _pos_int(cfg.get("topk")),
        "context_length": _pos_int(cfg.get("context")),
        "num_attn_heads": _pos_int(cfg.get("n")),
        "num_kv_heads": _pos_int(cfg.get("n_kv")),
        "architecture": cfg.get("architecture") or None,
    }


@app.get("/models")
def get_models(
    include: str | None = Query(default=None, examples=["specs"], description="Comma-separated extras: specs."),
):
    """List supported models."""
    models = sorted(get_default_models())
    if include and "specs" in _parse_include(include):
        return {"models": [_model_specs(m) for m in models]}
    return {"models": models}


@app.get("/systems")
def get_systems(
    include: str | None = Query(default=None, examples=["specs"], description="Comma-separated extras: specs."),
):
    """List supported GPU systems."""
    includes = _parse_include(include)
    want_specs = "specs" in includes

    systems = []
    for sys_id in sorted(supported_systems()):
        device_name = _DEVICE_DISPLAY_NAMES.get(sys_id)
        if device_name is None:
            # No perf-data display name -> no benchmark data; hide it.
            continue
        entry: dict[str, Any] = {
            "id": sys_id,
            "name": device_name,
        }
        if want_specs:
            try:
                spec = load_system_spec(sys_id)
                gpu = spec.get("gpu", {})
                node = spec.get("node", {})
                sm = int(gpu.get("sm_version", 0))
                sm_arch = _architecture_from_sm(sm)
                if sm_arch == "Other":
                    vendor_name = ""
                else:
                    vendor_name = "nvidia"
                entry.update({
                    "vendor": vendor_name,
                    "architecture": sm_arch,
                    "memory_bytes": int(gpu.get("mem_capacity", 0)),
                    "memory_bandwidth_bytes": int(gpu.get("mem_bw", 0)),
                    "bf16_tflops": float(gpu.get("bfloat16_tc_flops", 0)) / 1e12,
                    "tdp_watts": float(gpu.get("power", 0)),
                    "gpus_per_node": int(node.get("num_gpus_per_node", 0)),
                })
            except Exception:
                logger.warning("failed to load spec for %s", sys_id)
        systems.append(entry)
    return {"systems": systems}


@app.get("/metrics")
def get_metrics(request: Request):
    """Expose metrics in Prometheus or OTLP JSON format via content negotiation.

    - Accept: application/json → OTLP JSON (for pmdaopentelemetry and similar)
    - anything else (default)  → Prometheus/OpenMetrics text
    """
    if not _OBS:
        raise HTTPException(
            status_code=503,
            detail="Metrics unavailable (install with: pip install '.[otel]')",
        )
    return observability.metrics_response(request.headers.get("accept", "text/plain"))


# ─── Entrypoint ──────────────────────────────────────────────────────────────

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    uvicorn.run(app, host=args.server_name, port=args.server_port)
