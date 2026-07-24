from __future__ import annotations

import json
import re
from typing import Any


PLUGIN_VERSION = "2.0.0"
MIN_GLINER2_VERSION = "1.3.2"
GLINER2_PACKAGE_SPEC = "gliner2>=1.3.2,<2"
GLINER2_LOCAL_PACKAGE_SPEC = "gliner2[local]>=1.3.2,<2"

DEFAULT_MODEL = "fastino/gliner2-base-v1"
DEFAULT_ENTITY_TYPES = ["person", "organization", "location", "product", "date"]

DEFAULT_MAX_TEXT_CHARS = 50_000
HARD_MAX_TEXT_CHARS = 200_000
DEFAULT_MAX_SCHEMA_ITEMS = 100
HARD_MAX_SCHEMA_ITEMS = 500
MAX_SCHEMA_JSON_CHARS = 30_000


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


def _coerce_float(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _coerce_int(
    value: Any,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _coerce_string_list(
    value: Any,
    default: list[str] | None = None,
) -> list[str]:
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
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result or fallback


def _coerce_env_name(value: Any) -> str:
    candidate = str(value or "PIONEER_API_KEY").strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
        return candidate
    return "PIONEER_API_KEY"


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})

    mode = str(cfg.get("gliner2_mode", "local") or "local").strip().lower()
    if mode not in {"local", "api"}:
        mode = "local"

    device = str(cfg.get("gliner2_device", "auto") or "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        device = "auto"

    cfg["gliner2_enabled"] = _coerce_bool(cfg.get("gliner2_enabled"), True)
    cfg["gliner2_mode"] = mode
    cfg["gliner2_model"] = str(
        cfg.get("gliner2_model", DEFAULT_MODEL) or DEFAULT_MODEL
    ).strip()
    cfg["gliner2_device"] = device
    cfg["gliner2_api_key_env"] = _coerce_env_name(
        cfg.get("gliner2_api_key_env", "PIONEER_API_KEY")
    )
    cfg["gliner2_api_base_url"] = str(
        cfg.get("gliner2_api_base_url", "") or ""
    ).strip()
    cfg["gliner2_api_timeout_seconds"] = _coerce_float(
        cfg.get("gliner2_api_timeout_seconds"), 30.0, 1.0, 300.0
    )
    cfg["gliner2_api_max_retries"] = _coerce_int(
        cfg.get("gliner2_api_max_retries"), 3, 0, 10
    )
    cfg["gliner2_quantize"] = _coerce_bool(
        cfg.get("gliner2_quantize"), False
    )
    cfg["gliner2_compile"] = _coerce_bool(cfg.get("gliner2_compile"), False)
    cfg["gliner2_entity_threshold"] = _coerce_float(
        cfg.get("gliner2_entity_threshold"), 0.5, 0.0, 1.0
    )
    cfg["gliner2_max_len"] = _coerce_int(
        cfg.get("gliner2_max_len"), 0, 0, 8192
    )
    cfg["gliner2_max_text_chars"] = _coerce_int(
        cfg.get("gliner2_max_text_chars"),
        DEFAULT_MAX_TEXT_CHARS,
        1_000,
        HARD_MAX_TEXT_CHARS,
    )
    cfg["gliner2_max_schema_items"] = _coerce_int(
        cfg.get("gliner2_max_schema_items"),
        DEFAULT_MAX_SCHEMA_ITEMS,
        1,
        HARD_MAX_SCHEMA_ITEMS,
    )
    cfg["gliner2_operation_timeout_seconds"] = _coerce_int(
        cfg.get("gliner2_operation_timeout_seconds"), 30, 1, 600
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
    cfg["gliner2_tool_enabled"] = _coerce_bool(
        cfg.get("gliner2_tool_enabled"), True
    )
    cfg["gliner2_memory_entity_types"] = _coerce_string_list(
        cfg.get("gliner2_memory_entity_types"), DEFAULT_ENTITY_TYPES
    )
    cfg["gliner2_import_entity_types"] = _coerce_string_list(
        cfg.get("gliner2_import_entity_types"), DEFAULT_ENTITY_TYPES
    )
    return cfg


def count_schema_items(value: Any) -> int:
    if isinstance(value, dict):
        return len(value) + sum(count_schema_items(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) + sum(count_schema_items(child) for child in value)
    return 0


def validate_text(text: Any, config: dict[str, Any]) -> tuple[str, str | None]:
    value = str(text or "").strip()
    if not value:
        return "", "Missing required field: text"

    maximum = int(config.get("gliner2_max_text_chars", DEFAULT_MAX_TEXT_CHARS))
    if len(value) > maximum:
        return (
            "",
            f"Text is {len(value):,} characters; this plugin is configured for "
            f"at most {maximum:,} characters per call.",
        )
    return value, None


def validate_schema_size(
    schema: Any,
    config: dict[str, Any],
) -> str | None:
    try:
        serialized = json.dumps(schema, ensure_ascii=False)
    except (TypeError, ValueError):
        return "Schema must be JSON serializable."

    if len(serialized) > MAX_SCHEMA_JSON_CHARS:
        return (
            f"Schema is too large ({len(serialized):,} JSON characters); "
            f"the limit is {MAX_SCHEMA_JSON_CHARS:,}."
        )

    count = count_schema_items(schema)
    maximum = int(config.get("gliner2_max_schema_items", DEFAULT_MAX_SCHEMA_ITEMS))
    if count > maximum:
        return (
            f"Schema contains {count} items; this plugin is configured for "
            f"at most {maximum} items per call."
        )
    return None


def normalize_schema_for_task(
    task: str,
    schema: Any,
    config: dict[str, Any],
) -> tuple[Any, str | None]:
    task_name = str(task or "").strip().lower()

    if task_name == "entities":
        final_schema = schema or config.get("gliner2_memory_entity_types", [])
        if not isinstance(final_schema, (list, dict)) or not final_schema:
            return None, "Entities expects a non-empty JSON array or object schema."
    elif task_name == "relations":
        final_schema = schema or []
        if not isinstance(final_schema, (list, dict)) or not final_schema:
            return None, "Relations expects a non-empty JSON array or object schema."
    elif task_name == "classify":
        final_schema = schema or {}
        if not isinstance(final_schema, dict) or not final_schema:
            return (
                None,
                "Classify expects a non-empty JSON object, for example "
                '{"sentiment": ["positive", "negative"]}.',
            )
    elif task_name == "json":
        final_schema = schema or {}
        if not isinstance(final_schema, dict) or not final_schema:
            return (
                None,
                "JSON extraction expects a non-empty JSON object, for example "
                '{"product": ["name", "price"]}.',
            )
        if any(not isinstance(fields, list) for fields in final_schema.values()):
            return (
                None,
                "Each JSON structure must map to an array of GLiNER2 field specs.",
            )
    else:
        return None, "Unsupported task. Use one of: entities, classify, json, relations."

    size_error = validate_schema_size(final_schema, config)
    if size_error:
        return None, size_error
    return final_schema, None
