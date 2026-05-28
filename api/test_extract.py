from __future__ import annotations

from helpers.api import ApiHandler, Request, Response
from helpers import plugins

from usr.plugins.gliner2.helpers.gliner2_client import get_client
from usr.plugins.gliner2.tools.gliner2_extract import normalize_schema_for_task


class TestExtract(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        config = plugins.get_plugin_config(
            "gliner2",
            project_name=str(input.get("project_name", "") or ""),
            agent_profile=str(input.get("agent_profile", "") or ""),
        ) or {}
        client = get_client(config)
        if not client.is_available():
            return {
                "ok": False,
                "error": client.last_error or "GLiNER2 is unavailable.",
            }

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

        if task == "entities":
            result = client.extract_entities(
                text=text,
                schema=normalized_schema,
                threshold=float(config.get("gliner2_entity_threshold", 0.5) or 0.5),
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        elif task == "classify":
            result = client.classify_text(text=text, schema=normalized_schema, include_confidence=include_confidence)
        elif task == "json":
            result = client.extract_json(
                text=text,
                schema=normalized_schema,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        elif task == "relations":
            result = client.extract_relations(
                text=text,
                schema=normalized_schema,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )

        if result is None:
            return {"ok": False, "error": client.last_error or "GLiNER2 returned no result."}

        return {"ok": True, "result": result}
