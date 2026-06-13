from __future__ import annotations

import asyncio
import json
from typing import Any

from helpers import plugins
from helpers.tool import Response, Tool

from usr.plugins.gliner2.helpers.gliner2_client import get_loaded_client_or_status


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


def _model_not_ready_message(status: dict[str, Any]) -> str:
    state = status.get("model_state", "not_loaded")
    if not status.get("package_installed"):
        error = status.get("error") or "GLiNER2 package is not installed."
    elif state == "loading":
        error = "GLiNER2 model is loading in the background. Retry after model_state is loaded."
    elif state == "load_failed":
        error = status.get("error") or "GLiNER2 model load failed."
    else:
        error = "GLiNER2 model is not loaded yet; background loading has been started."

    payload = {
        "ok": False,
        "error": error,
        "model_state": state,
        "model_loading": bool(status.get("model_loading")),
        "model_load_started": bool(status.get("model_load_started")),
        "model_loading_seconds": status.get("model_loading_seconds"),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


async def _run_with_timeout(timeout_seconds: int, func, *args, **kwargs):
    return await asyncio.wait_for(
        asyncio.to_thread(func, *args, **kwargs),
        timeout=max(1, int(timeout_seconds)),
    )


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

        task_name = str(task or "").strip().lower()
        normalized_schema, schema_error = normalize_schema_for_task(task_name, schema, config)
        if schema_error:
            return Response(message=schema_error, break_loop=False)

        client, status = get_loaded_client_or_status(config)
        if client is None:
            return Response(message=_model_not_ready_message(status), break_loop=False)

        timeout_seconds = int(config.get("gliner2_operation_timeout_seconds", 30) or 30)

        try:
            result = None
            if task_name == "entities":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.extract_entities,
                    text=text,
                    schema=normalized_schema,
                    threshold=float(config.get("gliner2_entity_threshold", 0.5) or 0.5),
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
            elif task_name == "classify":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.classify_text,
                    text=text,
                    schema=normalized_schema,
                    include_confidence=include_confidence,
                )
            elif task_name == "json":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.extract_json,
                    text=text,
                    schema=normalized_schema,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
            elif task_name == "relations":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.extract_relations,
                    text=text,
                    schema=normalized_schema,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
        except TimeoutError:
            return Response(
                message=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "GLiNER2 extraction timed out after "
                            f"{timeout_seconds} seconds."
                        ),
                        "model_state": "loaded",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                break_loop=False,
            )

        if result is None:
            error = client.last_error or "GLiNER2 did not return a result."
            return Response(message=error, break_loop=False)

        return Response(
            message=json.dumps(result, indent=2, sort_keys=True),
            break_loop=False,
        )
