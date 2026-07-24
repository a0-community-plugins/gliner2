from __future__ import annotations

import asyncio

from helpers import plugins
from helpers.api import ApiHandler, Request, Response

from usr.plugins.gliner2.helpers.config import normalize_config


class Install(ApiHandler):
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
        mode = str(input.get("mode", "") or config["gliner2_mode"])
        return await asyncio.to_thread(
            plugins.call_plugin_hook,
            "gliner2",
            "install",
            {"ok": False, "error": "Install hook unavailable."},
            mode=mode,
            config=config,
            raise_on_error=False,
        )
