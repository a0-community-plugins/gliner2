from __future__ import annotations

import asyncio

from helpers import plugins
from helpers.api import ApiHandler, Request, Response

from usr.plugins.gliner2.helpers.config import normalize_config
from usr.plugins.gliner2.helpers.gliner2_client import get_runtime_status


class Status(ApiHandler):
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
        return await asyncio.to_thread(
            get_runtime_status,
            config,
            bool(input.get("load_model", False)),
        )
