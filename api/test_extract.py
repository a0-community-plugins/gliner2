from __future__ import annotations

import asyncio

from helpers.api import ApiHandler, Request, Response
from helpers import plugins

from usr.plugins.gliner2.helpers.gliner2_client import get_loaded_client_or_status
from usr.plugins.gliner2.tools.gliner2_extract import normalize_schema_for_task


async def _run_with_timeout(timeout_seconds: int, func, *args, **kwargs):
    return await asyncio.wait_for(
        asyncio.to_thread(func, *args, **kwargs),
        timeout=max(1, int(timeout_seconds)),
    )


class TestExtract(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        config = plugins.get_plugin_config(
            "gliner2",
            project_name=str(input.get("project_name", "") or ""),
            agent_profile=str(input.get("agent_profile", "") or ""),
        ) or {}
        task = str(input.get("task", "entities") or "entities").strip().lower()
        text = str(input.get("text", "") or "").strip()
        schema = input.get("schema")
        include_confidence = bool(input.get("include_confidence", False))
        include_spans = bool(input.get("include_spans", False))

        if not text:
            return {"ok": False, "error": "Missing required field: text"}

        normalized_schema, schema_error = normalize_schema_for_task(task, schema, config)
        if schema_error:
            return {"ok": False, "error": schema_error}

        client, status = get_loaded_client_or_status(config)
        if client is None:
            return {
                "ok": False,
                "error": (
                    status.get("error")
                    or (
                        "GLiNER2 model is loading in the background. "
                        "Refresh status and retry after model_state is loaded."
                    )
                ),
                "model_state": status.get("model_state"),
                "model_loading": status.get("model_loading"),
                "model_load_started": status.get("model_load_started"),
                "model_loading_seconds": status.get("model_loading_seconds"),
            }

        timeout_seconds = int(config.get("gliner2_operation_timeout_seconds", 30) or 30)

        try:
            if task == "entities":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.extract_entities,
                    text=text,
                    schema=normalized_schema,
                    threshold=float(config.get("gliner2_entity_threshold", 0.5) or 0.5),
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
            elif task == "classify":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.classify_text,
                    text=text,
                    schema=normalized_schema,
                    include_confidence=include_confidence,
                )
            elif task == "json":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.extract_json,
                    text=text,
                    schema=normalized_schema,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
            elif task == "relations":
                result = await _run_with_timeout(
                    timeout_seconds,
                    client.extract_relations,
                    text=text,
                    schema=normalized_schema,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
        except TimeoutError:
            return {
                "ok": False,
                "error": f"GLiNER2 extraction timed out after {timeout_seconds} seconds.",
                "model_state": "loaded",
            }

        if result is None:
            return {"ok": False, "error": client.last_error or "GLiNER2 returned no result."}

        return {"ok": True, "result": result}
