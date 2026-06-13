from __future__ import annotations

import subprocess
import sys
import json
from typing import Any

from helpers import plugins

from usr.plugins.gliner2.helpers.gliner2_client import get_client, get_runtime_status


DEFAULT_ENTITY_TYPES = ["person", "organization", "location", "product", "date"]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _coerce_string_list(value: Any, default: list[str] | None = None) -> list[str]:
    fallback = list(default or [])
    raw = value
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return fallback
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError:
            raw = [part.strip() for part in stripped.replace("\n", ",").split(",")]

    if not isinstance(raw, list):
        return fallback

    result: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result or fallback


def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    mode = str(cfg.get("gliner2_mode", "local") or "local").strip().lower()
    if mode not in {"local", "api"}:
        mode = "local"

    cfg["gliner2_enabled"] = _coerce_bool(cfg.get("gliner2_enabled"), True)
    cfg["gliner2_mode"] = mode
    cfg["gliner2_model"] = str(
        cfg.get("gliner2_model", "fastino/gliner2-base-v1") or "fastino/gliner2-base-v1"
    ).strip()
    cfg["gliner2_api_key_env"] = str(
        cfg.get("gliner2_api_key_env", "PIONEER_API_KEY") or "PIONEER_API_KEY"
    ).strip()
    cfg["gliner2_quantize"] = _coerce_bool(cfg.get("gliner2_quantize"), False)
    cfg["gliner2_compile"] = _coerce_bool(cfg.get("gliner2_compile"), False)
    cfg["gliner2_entity_threshold"] = _coerce_float(
        cfg.get("gliner2_entity_threshold"), 0.5, 0.0, 1.0
    )
    cfg["gliner2_utility_replacement_enabled"] = _coerce_bool(
        cfg.get("gliner2_utility_replacement_enabled"), True
    )
    cfg["gliner2_fallback_to_utility_model"] = _coerce_bool(
        cfg.get("gliner2_fallback_to_utility_model"), True
    )
    cfg["gliner2_usage_logging"] = _coerce_bool(
        cfg.get("gliner2_usage_logging"), True
    )
    cfg["gliner2_memory_keyword_extraction"] = _coerce_bool(
        cfg.get("gliner2_memory_keyword_extraction"), True
    )
    cfg["gliner2_recall_query_enrichment"] = _coerce_bool(
        cfg.get("gliner2_recall_query_enrichment"), False
    )
    cfg["gliner2_memory_post_filter"] = _coerce_bool(
        cfg.get("gliner2_memory_post_filter"), True
    )
    cfg["gliner2_post_filter_threshold"] = _coerce_float(
        cfg.get("gliner2_post_filter_threshold"), 0.5, 0.0, 1.0
    )
    cfg["gliner2_consolidation_triage"] = _coerce_bool(
        cfg.get("gliner2_consolidation_triage"), True
    )
    cfg["gliner2_consolidation_triage_threshold"] = _coerce_float(
        cfg.get("gliner2_consolidation_triage_threshold"), 0.65, 0.0, 1.0
    )
    cfg["gliner2_knowledge_import_enrichment"] = _coerce_bool(
        cfg.get("gliner2_knowledge_import_enrichment"), True
    )
    cfg["gliner2_tool_enabled"] = _coerce_bool(cfg.get("gliner2_tool_enabled"), True)
    cfg["gliner2_operation_timeout_seconds"] = _coerce_int(
        cfg.get("gliner2_operation_timeout_seconds"), 30, 1, 600
    )
    cfg["gliner2_memory_entity_types"] = _coerce_string_list(
        cfg.get("gliner2_memory_entity_types"), DEFAULT_ENTITY_TYPES
    )
    cfg["gliner2_import_entity_types"] = _coerce_string_list(
        cfg.get("gliner2_import_entity_types"), DEFAULT_ENTITY_TYPES
    )
    return cfg


def _get_config(agent=None) -> dict[str, Any]:
    return _normalize_config(plugins.get_plugin_config("gliner2", agent=agent) or {})


def get_plugin_config(default: dict[str, Any] | None = None, **kwargs):
    return _normalize_config(default)


def save_plugin_config(settings: dict[str, Any] | None = None, **kwargs):
    return _normalize_config(settings)


def _append_entity_text(flat: list[str], seen: set[str], value: Any) -> None:
    if isinstance(value, dict):
        text = str(value.get("text") or value.get("value") or "").strip()
        if text and text not in seen:
            seen.add(text)
            flat.append(text)
        return

    text = str(value).strip()
    if text and text not in seen:
        seen.add(text)
        flat.append(text)


def _flatten_entities(result: dict[str, Any] | None) -> list[str]:
    if not isinstance(result, (dict, list)):
        return []

    entities = result.get("entities", result) if isinstance(result, dict) else result
    flat: list[str] = []
    seen: set[str] = set()

    if isinstance(entities, list):
        for value in entities:
            _append_entity_text(flat, seen, value)
        return flat

    if not isinstance(entities, dict):
        _append_entity_text(flat, seen, entities)
        return flat

    for values in entities.values():
        if isinstance(values, list):
            for value in values:
                _append_entity_text(flat, seen, value)
        else:
            _append_entity_text(flat, seen, values)
    return flat


def install(mode: str | None = None, config: dict[str, Any] | None = None, **kwargs):
    cfg = _normalize_config(config)
    install_mode = str(mode or cfg.get("gliner2_mode", "local") or "local").lower()
    package = "gliner2[local]" if install_mode == "local" else "gliner2"
    command = [sys.executable, "-m", "pip", "install", package]
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "ok": result.returncode == 0,
        "mode": install_mode,
        "package": package,
        "command": " ".join(command),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def pre_update(**kwargs):
    return {"ok": True, "message": "No pre-update migration required."}


def provide_memory_keywords(agent=None, text: str = "", **kwargs):
    config = _get_config(agent=agent)
    if not config.get("gliner2_enabled", True):
        return None
    if not config.get("gliner2_memory_keyword_extraction", True):
        return None
    if not text.strip():
        return None

    client = get_client(config)
    if not client.is_available(load_model=False):
        return None

    result = client.extract_entities(
        text=text,
        schema=config.get("gliner2_memory_entity_types", []) or [],
        threshold=float(config.get("gliner2_entity_threshold", 0.5) or 0.5),
    )
    keywords = _flatten_entities(result)
    return keywords or None


def enrich_knowledge_metadata(
    agent=None,
    text: str = "",
    metadata: dict[str, Any] | None = None,
    log_item=None,
    **kwargs,
):
    config = _get_config(agent=agent)
    if not config.get("gliner2_enabled", True):
        return None
    if not config.get("gliner2_knowledge_import_enrichment", True):
        return None
    if not text.strip():
        return None

    client = get_client(config)
    if not client.is_available(load_model=False):
        return None

    result = client.extract_entities(
        text=text,
        schema=config.get("gliner2_import_entity_types", []) or [],
        threshold=float(config.get("gliner2_entity_threshold", 0.5) or 0.5),
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


def status(agent=None, load_model: bool = False, **kwargs):
    return get_runtime_status(_get_config(agent=agent), load_model=load_model)
