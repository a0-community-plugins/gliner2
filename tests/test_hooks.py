from __future__ import annotations

from types import SimpleNamespace

from usr.plugins.gliner2 import hooks
from usr.plugins.gliner2.helpers.config import (
    GLINER2_LOCAL_PACKAGE_SPEC,
    GLINER2_PACKAGE_SPEC,
)


def test_plugin_does_not_override_agent_zero_config_loading() -> None:
    assert not hasattr(hooks, "get_plugin_config")


def test_save_hook_normalizes_and_clears_cached_clients(monkeypatch) -> None:
    cleared: list[bool] = []
    monkeypatch.setattr(hooks, "clear_clients", lambda: cleared.append(True))

    saved = hooks.save_plugin_config(
        {
            "gliner2_mode": "API",
            "gliner2_enabled": "false",
            "gliner2_entity_threshold": "0.7",
        }
    )

    assert cleared == [True]
    assert saved["gliner2_mode"] == "api"
    assert saved["gliner2_enabled"] is False
    assert saved["gliner2_entity_threshold"] == 0.7


def test_flatten_entities_handles_current_shapes_and_casefolds_duplicates() -> None:
    result = {
        "entities": {
            "person": [
                {"text": "Ada Lovelace", "confidence": 0.98},
                "ada lovelace",
            ],
            "organization": [{"value": "Royal Society"}],
        }
    }

    assert hooks._flatten_entities(result) == ["Ada Lovelace", "Royal Society"]


def test_install_selects_stable_package_spec(monkeypatch) -> None:
    calls: list[list[str]] = []
    executables: list[str] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        executables.append(kwargs["executable"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(hooks.subprocess, "run", fake_run)
    monkeypatch.setattr(hooks, "clear_clients", lambda: None)

    local_result = hooks.install(
        mode="local",
        config={"gliner2_mode": "local"},
        raise_on_error=False,
    )
    api_result = hooks.install(
        mode="api",
        config={"gliner2_mode": "api"},
        raise_on_error=False,
    )

    assert local_result["package"] == GLINER2_LOCAL_PACKAGE_SPEC
    assert api_result["package"] == GLINER2_PACKAGE_SPEC
    assert calls[0][-1] == GLINER2_LOCAL_PACKAGE_SPEC
    assert calls[1][-1] == GLINER2_PACKAGE_SPEC
    assert executables == [hooks.sys.executable, hooks.sys.executable]


def test_install_sanitizes_authenticated_index_urls(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="Looking in indexes: https://user:password@example.invalid/simple",
            stderr="failed",
        )

    monkeypatch.setattr(hooks.subprocess, "run", fake_run)

    result = hooks.install(
        mode="api",
        config={"gliner2_mode": "api"},
        raise_on_error=False,
    )

    assert result["ok"] is False
    assert "password" not in result["stdout"]
    assert "***:***@" in result["stdout"]
