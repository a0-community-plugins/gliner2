from __future__ import annotations

import asyncio

from helpers import plugins
from helpers.api import ApiHandler, Request, Response

from usr.plugins.gliner2.helpers.config import normalize_config
from usr.plugins.gliner2.helpers.extraction import execute_extraction
from usr.plugins.gliner2.helpers.gliner2_client import get_loaded_client_or_status


class TestExtract(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        loaded = (
            plugins.get_plugin_config(
                "gliner2",
                project_name=str(input.get("project_name", "") or ""),
                agent_profile=str(input.get("agent_profile", "") or ""),
            )
            or {}
        )
        override = input.get("settings")
        if isinstance(override, dict):
            loaded = {**loaded, **override}
        config = normalize_config(loaded)
        if not config["gliner2_enabled"]:
            return {"ok": False, "error": "GLiNER2 is disabled in this scope."}

        client, status = get_loaded_client_or_status(config)
        if client is None:
            return {
                "ok": False,
                "error": (
                    status.get("error")
                    or next(iter(status.get("blockers") or []), None)
                    or "GLiNER2 is starting. Retry when client_state is ready."
                ),
                "client_state": status.get("client_state"),
                "client_loading": status.get("client_loading"),
                "client_loading_seconds": status.get("client_loading_seconds"),
                "model_load_started": status.get("model_load_started"),
            }

        timeout_seconds = config["gliner2_operation_timeout_seconds"]
        try:
            result, error = await asyncio.wait_for(
                asyncio.to_thread(
                    execute_extraction,
                    client,
                    config,
                    task=input.get("task", "entities"),
                    text=input.get("text", ""),
                    schema=input.get("schema"),
                    include_confidence=bool(input.get("include_confidence", False)),
                    include_spans=bool(input.get("include_spans", False)),
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return {
                "ok": False,
                "error": (
                    f"GLiNER2 extraction exceeded the {timeout_seconds}-second "
                    "operation timeout."
                ),
                "client_state": "ready",
            }
        if error:
            return {"ok": False, "error": error}
        return {"ok": True, "result": result}
