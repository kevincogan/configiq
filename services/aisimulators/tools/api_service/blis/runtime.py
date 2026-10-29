"""Resolve the pinned BLIS runtime paths without accepting arbitrary executables."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .manifest import BlisManifestError


@dataclass(frozen=True)
class BlisRuntime:
    binary: str
    catalog: str
    registry: str
    scenarios: str


def _required_directory(name: str, default: str) -> str:
    path = Path(os.environ.get(name, default)).expanduser().resolve()
    if not path.is_dir():
        raise BlisManifestError(f"BLIS runtime directory {name} is unavailable: {path}")
    return str(path)


def resolve_runtime() -> BlisRuntime:
    raw_binary = os.environ.get("BLIS_BINARY") or shutil.which("blis")
    if not raw_binary:
        raise BlisManifestError("BLIS binary is unavailable; set BLIS_BINARY or install blis on PATH")
    binary_path = Path(raw_binary).expanduser()
    if not binary_path.is_absolute():
        located = shutil.which(raw_binary)
        if not located:
            raise BlisManifestError(f"BLIS_BINARY {raw_binary!r} is not on PATH")
        binary_path = Path(located)
    binary_path = binary_path.resolve()
    if binary_path.name not in {"blis", "blis.exe"}:
        raise BlisManifestError("BLIS_BINARY must refer to the BLIS executable")
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise BlisManifestError(f"BLIS_BINARY is not executable: {binary_path}")

    blis_home = os.environ.get("BLIS_HOME")
    if blis_home and not binary_path.is_relative_to(Path(blis_home).expanduser().resolve()):
        raise BlisManifestError("BLIS_BINARY must be located inside BLIS_HOME")

    return BlisRuntime(
        binary=str(binary_path),
        catalog=_required_directory("BLIS_CATALOG", "/opt/blis/catalog"),
        registry=_required_directory("BLIS_REGISTRY", "/opt/blis/registry"),
        scenarios=_required_directory("BLIS_SCENARIOS", "/opt/blis/scenarios"),
    )
