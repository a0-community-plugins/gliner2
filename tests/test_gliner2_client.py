from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path


def _load_client_module():
    module_path = Path(__file__).resolve().parents[1] / "helpers" / "gliner2_client.py"
    module_name = f"gliner2_client_under_test_{time.monotonic_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_status_load_model_starts_background_load_without_blocking(
    tmp_path,
    monkeypatch,
):
    fake_module = tmp_path / "gliner2.py"
    fake_module.write_text(
        """
import time


class GLiNER2:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        time.sleep(0.75)
        return cls()

    def extract_entities(self, text, schema, **kwargs):
        return {"entities": ["Ada Lovelace"]}
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "gliner2", raising=False)

    gliner2_client = _load_client_module()
    config = {"gliner2_model": "fake/slow"}

    start = time.monotonic()
    status = gliner2_client.get_runtime_status(config, load_model=True)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    assert status["package_installed"] is True
    assert status["model_state"] == "loading"
    assert status["model_loading"] is True

    client = gliner2_client.get_client(config)
    deadline = time.monotonic() + 2
    while client.is_loading() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert client.model is not None


def test_client_readiness_check_does_not_load_model(tmp_path, monkeypatch):
    fake_module = tmp_path / "gliner2.py"
    fake_module.write_text(
        """
class GLiNER2:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("readiness check should not load")
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "gliner2", raising=False)

    gliner2_client = _load_client_module()
    client = gliner2_client.get_client({"gliner2_model": "fake/no-load"})

    assert client.is_available(load_model=False) is False
    assert client.model is None


def test_get_loaded_client_or_status_starts_load_without_blocking(
    tmp_path,
    monkeypatch,
):
    fake_module = tmp_path / "gliner2.py"
    fake_module.write_text(
        """
import time


class GLiNER2:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        time.sleep(0.75)
        return cls()
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "gliner2", raising=False)

    gliner2_client = _load_client_module()
    config = {"gliner2_model": "fake/slow"}

    start = time.monotonic()
    client, status = gliner2_client.get_loaded_client_or_status(config)
    elapsed = time.monotonic() - start

    assert client is None
    assert elapsed < 0.5
    assert status["model_state"] == "loading"
    assert status["model_load_started"] is True

    client = gliner2_client.get_client(config)
    deadline = time.monotonic() + 2
    while client.is_loading() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert client.model is not None
