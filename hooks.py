from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

from helpers import plugins

from usr.plugins.gliner2.helpers.config import (
    GLINER2_LOCAL_PACKAGE_SPEC,
    GLINER2_PACKAGE_SPEC,
    normalize_config,
    validate_text,
)
from usr.plugins.gliner2.helpers.gliner2_client import (
    clear_clients,
    get_client,
    get_runtime_status,
)


def _get_config(agent=None) -> dict[str, Any]:
    return normalize_config(plugins.get_plugin_config("gliner2", agent=agent) or {})


def save_plugin_config(
    settings: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Normalize persisted values without replacing Agent Zero's loaded config."""
    clear_clients()
    return normalize_config(settings)


def _append_entity_text(flat: list[str], seen: set[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        candidate = value.get("text")
        if candidate is None:
            candidate = value.get("value")
        if candidate is not None and not isinstance(candidate, (dict, list, tuple)):
            _append_entity_text(flat, seen, candidate)
            return
        for child in value.values():
            _append_entity_text(flat, seen, child)
        return
    if isinstance(value, (list, tuple, set)):
        for child in value:
            _append_entity_text(flat, seen, child)
        return

    text = str(value).strip()
    key = text.casefold()
    if text and key not in seen:
        seen.add(key)
        flat.append(text)


def _flatten_entities(result: Any) -> list[str]:
    if not isinstance(result, (dict, list, tuple)):
        return []

    entities = result.get("entities", result) if isinstance(result, dict) else result
    flat: list[str] = []
    seen: set[str] = set()
    _append_entity_text(flat, seen, entities)
    return flat


def _sanitize_install_output(value: str) -> str:
    without_credentials = re.sub(
        r"(https?://)[^/\s:@]+:[^/\s@]+@",
        r"\1***:***@",
        value or "",
    )
    return without_credentials[-6_000:]


def install(
    mode: str | None = None,
    config: dict[str, Any] | None = None,
    raise_on_error: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Install into Agent Zero's framework runtime.

    Plugin Hub calls this hook automatically after install/update. The settings
    panel calls the same hook for repair or mode changes, keeping all dependency
    execution in hooks.py.
    """
    loaded = config
    if loaded is None:
        loaded = plugins.get_plugin_config("gliner2") or {}
    cfg = normalize_config(loaded)
    install_mode = str(mode or cfg["gliner2_mode"]).strip().lower()
    if install_mode not in {"local", "api"}:
        install_mode = "local"

    package = (
        GLINER2_LOCAL_PACKAGE_SPEC
        if install_mode == "local"
        else GLINER2_PACKAGE_SPEC
    )
    command = [sys.executable, "-m", "pip", "install", package]
    try:
        result = subprocess.run(
            ["python", "-m", "pip", "install", package],
            executable=sys.executable,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        payload = {
            "ok": result.returncode == 0,
            "mode": install_mode,
            "package": package,
            "command": " ".join(command),
            "stdout": _sanitize_install_output(result.stdout),
            "stderr": _sanitize_install_output(result.stderr),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        payload = {
            "ok": False,
            "mode": install_mode,
            "package": package,
            "command": " ".join(command),
            "stdout": _sanitize_install_output(str(exc.stdout or "")),
            "stderr": "Dependency installation timed out after 15 minutes.",
            "exit_code": -1,
        }

    if payload["ok"]:
        clear_clients()
        return payload
    if raise_on_error:
        detail = payload["stderr"] or payload["stdout"] or "pip returned an error."
        raise RuntimeError(f"GLiNER2 dependency installation failed: {detail}")
    return payload


def pre_update(**kwargs: Any) -> dict[str, Any]:
    clear_clients()
    return {"ok": True, "message": "GLiNER2 runtime cache cleared for update."}


def provide_memory_keywords(
    agent=None,
    text: str = "",
    **kwargs: Any,
) -> list[str] | None:
    config = _get_config(agent=agent)
    if not config["gliner2_enabled"]:
        return None
    if not config["gliner2_memory_keyword_extraction"]:
        return None

    normalized_text, error = validate_text(text, config)
    if error:
        return None

    client = get_client(config)
    if not client.is_available(load_model=False):
        return None

    result = client.extract_entities(
        text=normalized_text,
        schema=config["gliner2_memory_entity_types"],
        threshold=config["gliner2_entity_threshold"],
        max_len=config["gliner2_max_len"] or None,
    )
    keywords = _flatten_entities(result)
    return keywords or None


def enrich_knowledge_metadata(
    agent=None,
    text: str = "",
    metadata: dict[str, Any] | None = None,
    log_item=None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    config = _get_config(agent=agent)
    if not config["gliner2_enabled"]:
        return None
    if not config["gliner2_knowledge_import_enrichment"]:
        return None

    normalized_text, error = validate_text(text, config)
    if error:
        return None

    client = get_client(config)
    if not client.is_available(load_model=False):
        return None

    result = client.extract_entities(
        text=normalized_text,
        schema=config["gliner2_import_entity_types"],
        threshold=config["gliner2_entity_threshold"],
        max_len=config["gliner2_max_len"] or None,
    )
    flat = _flatten_entities(result)
    if not flat:
        return None

    entities = result.get("entities", {}) if isinstance(result, dict) else {}
    return {
        "gliner2_entities": entities,
        "gliner2_entity_flat": " ".join(flat),
        "gliner2_enriched": True,
    }


def status(
    agent=None,
    load_model: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    return get_runtime_status(_get_config(agent=agent), load_model=load_model)
