from __future__ import annotations

import json
from typing import Any

from helpers import plugins
from helpers.tool import Response, Tool

from usr.plugins.gliner2.helpers.gliner2_client import get_client


def get_plugin_config(agent=None):
    return plugins.get_plugin_config("gliner2", agent=agent)


def normalize_schema_for_task(task: str, schema: Any, config: dict[str, Any]) -> tuple[Any, str | None]:
    task_name = str(task or "").strip().lower()

    if task_name == "entities":
        final_schema = schema or config.get("gliner2_memory_entity_types", [])
        if not isinstance(final_schema, (list, dict)):
            return None, "Entities task expects a JSON array or object schema."
        return final_schema, None

    if task_name == "relations":
        final_schema = schema or []
        if not isinstance(final_schema, (list, dict)):
            return None, "Relations task expects a JSON array or object schema."
        return final_schema, None

    if task_name == "classify":
        final_schema = schema or {}
        if not isinstance(final_schema, dict):
            return None, "Classify task expects a JSON object schema, for example {\"sentiment\": [\"positive\", \"negative\"]}."
        return final_schema, None

    if task_name == "json":
        final_schema = schema or {}
        if not isinstance(final_schema, dict):
            return None, "JSON task expects a JSON object schema, for example {\"product\": [\"name\", \"price\"]}."
        return final_schema, None

    return None, "Unsupported task. Use one of: entities, classify, json, relations."


class GLiNER2Extract(Tool):
    async def execute(
        self,
        task: str = "",
        text: str = "",
        schema=None,
        include_confidence: bool = False,
        include_spans: bool = False,
        **kwargs,
    ) -> Response:
        config = get_plugin_config(agent=self.agent) or {}
        if not config.get("gliner2_enabled", True):
            return Response(
                message="GLiNER2 plugin is disabled in settings.",
                break_loop=False,
            )
        if not config.get("gliner2_tool_enabled", True):
            return Response(
                message="GLiNER2 direct tool use is disabled in settings.",
                break_loop=False,
            )
        if not text.strip():
            return Response(message="Missing required argument: text", break_loop=False)

        client = get_client(config)
        if not client.is_available():
            return Response(
                message=(
                    "GLiNER2 is unavailable. Install `gliner2[local]` for local mode "
                    "or `gliner2` for API mode, then retry."
                ),
                break_loop=False,
            )

        task_name = str(task or "").strip().lower()
        normalized_schema, schema_error = normalize_schema_for_task(task_name, schema, config)
        if schema_error:
            return Response(message=schema_error, break_loop=False)

        result = None

        if task_name == "entities":
            result = client.extract_entities(
                text=text,
                schema=normalized_schema,
                threshold=float(config.get("gliner2_entity_threshold", 0.5) or 0.5),
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        elif task_name == "classify":
            result = client.classify_text(
                text=text,
                schema=normalized_schema,
                include_confidence=include_confidence,
            )
        elif task_name == "json":
            result = client.extract_json(
                text=text,
                schema=normalized_schema,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        elif task_name == "relations":
            result = client.extract_relations(
                text=text,
                schema=normalized_schema,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )

        if result is None:
            error = client.last_error or "GLiNER2 did not return a result."
            return Response(message=error, break_loop=False)

        return Response(
            message=json.dumps(result, indent=2, sort_keys=True),
            break_loop=False,
        )
