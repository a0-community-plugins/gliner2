from __future__ import annotations

import asyncio
import json
from typing import Any

from helpers import plugins
from helpers.tool import Response, Tool

from usr.plugins.gliner2.helpers.config import (
    GLINER2_LOCAL_PACKAGE_SPEC,
    GLINER2_PACKAGE_SPEC,
    normalize_config,
    normalize_schema_for_task,
)
from usr.plugins.gliner2.helpers.extraction import execute_extraction
from usr.plugins.gliner2.helpers.gliner2_client import get_loaded_client_or_status


def get_plugin_config(agent=None) -> dict[str, Any]:
    return normalize_config(plugins.get_plugin_config("gliner2", agent=agent) or {})


class GLiNER2Extract(Tool):
    async def execute(
        self,
        task: str = "",
        text: str = "",
        schema=None,
        include_confidence: bool = False,
        include_spans: bool = False,
        **kwargs: Any,
    ) -> Response:
        config = get_plugin_config(agent=self.agent)
        if not config["gliner2_enabled"]:
            return Response(
                message="GLiNER2 is disabled in this settings scope.",
                break_loop=False,
            )
        if not config["gliner2_tool_enabled"]:
            return Response(
                message="Direct GLiNER2 tool use is disabled in settings.",
                break_loop=False,
            )

        client, status = get_loaded_client_or_status(config)
        if client is None:
            error = (
                status.get("error")
                or next(iter(status.get("blockers") or []), None)
                or "GLiNER2 is starting. Retry when client_state is ready."
            )
            return Response(
                message=json.dumps(
                    {
                        "ok": False,
                        "error": error,
                        "client_state": status.get("client_state"),
                        "client_loading": status.get("client_loading"),
                        "client_loading_seconds": status.get(
                            "client_loading_seconds"
                        ),
                        "model_load_started": status.get("model_load_started"),
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                break_loop=False,
            )

        timeout_seconds = config["gliner2_operation_timeout_seconds"]
        try:
            result, error = await asyncio.wait_for(
                asyncio.to_thread(
                    execute_extraction,
                    client,
                    config,
                    task=task,
                    text=text,
                    schema=schema,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return Response(
                message=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "GLiNER2 extraction exceeded the "
                            f"{timeout_seconds}-second operation timeout."
                        ),
                        "client_state": "ready",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                break_loop=False,
            )
        if error:
            if client.last_error and error == client.last_error:
                package = (
                    GLINER2_LOCAL_PACKAGE_SPEC
                    if config["gliner2_mode"] == "local"
                    else GLINER2_PACKAGE_SPEC
                )
                error = (
                    f"{client.last_error} Open GLiNER2 settings and use "
                    f"Install / repair, or install `{package}` in the Agent Zero "
                    "framework runtime."
                )
            return Response(message=error, break_loop=False)

        return Response(
            message=json.dumps(result, indent=2, sort_keys=True, default=str),
            break_loop=False,
        )


__all__ = ["GLiNER2Extract", "normalize_schema_for_task"]
