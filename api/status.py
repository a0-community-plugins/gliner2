from __future__ import annotations

from helpers.api import ApiHandler, Request, Response
from helpers import plugins

from usr.plugins.gliner2.helpers.gliner2_client import get_runtime_status


class Status(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        config = plugins.get_plugin_config(
            "gliner2",
            project_name=str(input.get("project_name", "") or ""),
            agent_profile=str(input.get("agent_profile", "") or ""),
        ) or {}
        return get_runtime_status(config, load_model=bool(input.get("load_model", False)))
