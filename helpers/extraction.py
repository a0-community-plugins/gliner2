from __future__ import annotations

from typing import Any

from usr.plugins.gliner2.helpers.config import (
    normalize_schema_for_task,
    validate_text,
)
from usr.plugins.gliner2.helpers.gliner2_client import GLiNER2Client


def execute_extraction(
    client: GLiNER2Client,
    config: dict[str, Any],
    *,
    task: str,
    text: Any,
    schema: Any,
    include_confidence: bool = False,
    include_spans: bool = False,
) -> tuple[Any | None, str | None]:
    task_name = str(task or "").strip().lower()
    normalized_text, text_error = validate_text(text, config)
    if text_error:
        return None, text_error

    normalized_schema, schema_error = normalize_schema_for_task(
        task_name, schema, config
    )
    if schema_error:
        return None, schema_error

    threshold = float(config.get("gliner2_entity_threshold", 0.5) or 0.5)
    configured_max_len = int(config.get("gliner2_max_len", 0) or 0)
    max_len = configured_max_len or None

    if task_name == "entities":
        result = client.extract_entities(
            text=normalized_text,
            schema=normalized_schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )
    elif task_name == "classify":
        result = client.classify_text(
            text=normalized_text,
            schema=normalized_schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )
    elif task_name == "json":
        result = client.extract_json(
            text=normalized_text,
            schema=normalized_schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )
    else:
        result = client.extract_relations(
            text=normalized_text,
            schema=normalized_schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )

    if result is None:
        return None, client.last_error or "GLiNER2 returned no result."
    return result, None
