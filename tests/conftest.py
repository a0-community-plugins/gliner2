from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, message: str = "", break_loop: bool = False):
        self.message = message
        self.break_loop = break_loop


class _Tool:
    agent = None


class _Extension:
    agent = None


class _ApiHandler:
    pass


class _Request:
    pass


def _namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


try:
    plugin_spec = importlib.util.find_spec("usr.plugins.gliner2")
except ModuleNotFoundError:
    plugin_spec = None

if plugin_spec is None:
    _namespace("usr", PLUGIN_ROOT.parent.parent)
    _namespace("usr.plugins", PLUGIN_ROOT.parent)
    _namespace("usr.plugins.gliner2", PLUGIN_ROOT)


try:
    import helpers  # type: ignore
except ModuleNotFoundError:
    helpers_module = types.ModuleType("helpers")
    helpers_module.__path__ = []  # type: ignore[attr-defined]
    sys.modules["helpers"] = helpers_module
else:
    helpers_module = sys.modules["helpers"]

try:
    from helpers import plugins as _real_plugins  # noqa: F401
except (ImportError, ModuleNotFoundError):
    plugins_module = types.ModuleType("helpers.plugins")
    plugins_module.get_plugin_config = lambda *args, **kwargs: {}
    plugins_module.call_plugin_hook = lambda *args, **kwargs: kwargs.get("default")
    helpers_module.plugins = plugins_module
    sys.modules["helpers.plugins"] = plugins_module

try:
    from helpers.tool import Response as _real_response  # noqa: F401
except (ImportError, ModuleNotFoundError):
    tool_module = types.ModuleType("helpers.tool")
    tool_module.Response = _Response
    tool_module.Tool = _Tool
    sys.modules["helpers.tool"] = tool_module

try:
    from helpers.extension import Extension as _real_extension  # noqa: F401
except (ImportError, ModuleNotFoundError):
    extension_module = types.ModuleType("helpers.extension")
    extension_module.Extension = _Extension
    sys.modules["helpers.extension"] = extension_module

try:
    from helpers.api import ApiHandler as _real_api_handler  # noqa: F401
except (ImportError, ModuleNotFoundError):
    api_module = types.ModuleType("helpers.api")
    api_module.ApiHandler = _ApiHandler
    api_module.Request = _Request
    api_module.Response = _Response
    sys.modules["helpers.api"] = api_module
