from __future__ import annotations

from helpers.api import ApiHandler, Request, Response
from helpers import plugins


class Install(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        config = plugins.get_plugin_config(
            "gliner2",
            project_name=str(input.get("project_name", "") or ""),
            agent_profile=str(input.get("agent_profile", "") or ""),
        ) or {}
        mode = str(input.get("mode", "") or config.get("gliner2_mode", "local"))
        result = plugins.call_plugin_hook(
            "gliner2",
            "install",
            default={"ok": False, "error": "Install hook unavailable."},
            mode=mode,
            config=config,
        )
        return result
