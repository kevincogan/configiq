"""Run BLIS with explicit image resources and parse its stable metrics section."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from .runtime import BlisRuntime

METRICS_MARKER = "=== Simulation Metrics ==="


class BlisExecutionError(RuntimeError):
    """BLIS failed, timed out, or returned output outside its API contract."""


class BlisTimeoutError(BlisExecutionError):
    """A BLIS run exceeded its configured timeout."""


@dataclass(frozen=True)
class RunRequest:
    scenario: str
    scenario_directory: str
    isl: int
    osl: int
    concurrency: int
    num_requests: int
    prefix_tokens: int = 0


def build_run_command(runtime: BlisRuntime, request: RunRequest) -> list[str]:
    """Build argv only from resolved image paths and validated numeric inputs."""
    command = [
        runtime.binary,
        "run",
        "--catalog",
        runtime.catalog,
        "--registry",
        runtime.registry,
        "--scenarios",
        request.scenario_directory,
        "--scenario",
        request.scenario,
        "--prompt-tokens",
        str(request.isl),
        "--output-tokens",
        str(request.osl),
        "--concurrency",
        str(request.concurrency),
        "--num-requests",
        str(request.num_requests),
    ]
    if request.prefix_tokens > 0:
        command.extend(["--prefix-tokens", str(request.prefix_tokens)])
    return command


def parse_metrics(stdout: str) -> dict[str, Any]:
    marker_index = stdout.find(METRICS_MARKER)
    if marker_index < 0:
        raise BlisExecutionError(f"BLIS output is missing {METRICS_MARKER!r}")
    payload = stdout[marker_index + len(METRICS_MARKER):].strip()
    json_start = payload.find("{")
    if json_start < 0:
        raise BlisExecutionError("BLIS metrics section does not contain a JSON object")
    try:
        metrics, _ = json.JSONDecoder().raw_decode(payload, json_start)
    except json.JSONDecodeError as exc:
        raise BlisExecutionError(f"BLIS metrics JSON is invalid: {exc.msg}") from exc
    if not isinstance(metrics, dict):
        raise BlisExecutionError("BLIS metrics payload must be a JSON object")
    return metrics


def run_blis(
    runtime: BlisRuntime,
    request: RunRequest,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    command = build_run_command(runtime, request)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.environ.get("BLIS_HOME") or None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BlisTimeoutError(f"BLIS simulation timed out after {timeout_seconds:g}s") from exc
    except OSError as exc:
        raise BlisExecutionError(f"could not start BLIS: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise BlisExecutionError(f"BLIS exited with status {completed.returncode}: {detail or 'no diagnostics'}")
    return parse_metrics(completed.stdout)
