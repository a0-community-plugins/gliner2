from __future__ import annotations

import os
from typing import Any

from usr.plugins.gliner2.helpers import gliner2_client as client_module


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def extract_entities(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("entities", args, kwargs))
        return {"entities": {"person": ["Ada"]}}

    def classify_text(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("classify", args, kwargs))
        return {"sentiment": "positive"}

    def extract_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("json", args, kwargs))
        return {"product": {"name": "Analytical Engine"}}

    def extract_relations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("relations", args, kwargs))
        return {"relation_extraction": {"worked_with": [["Ada", "Charles"]]}}


def test_api_key_is_passed_explicitly_without_global_env_mutation(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    model = _FakeModel()

    class FakeGLiNER2:
        @classmethod
        def from_api(cls, **kwargs: Any) -> _FakeModel:
            captured.update(kwargs)
            return model

    monkeypatch.setattr(
        client_module,
        "_import_gliner2",
        lambda mode="local": (FakeGLiNER2, ""),
    )
    monkeypatch.setenv("CUSTOM_GLINER_KEY", "test-secret")
    monkeypatch.delenv("PIONEER_API_KEY", raising=False)

    client = client_module.GLiNER2Client(
        {
            "gliner2_mode": "api",
            "gliner2_api_key_env": "CUSTOM_GLINER_KEY",
            "gliner2_api_base_url": "https://example.invalid/api",
            "gliner2_api_timeout_seconds": 17,
            "gliner2_api_max_retries": 2,
        }
    )

    assert client.is_available() is True
    assert captured["api_key"] == "test-secret"
    assert captured["api_base_url"] == "https://example.invalid/api"
    assert captured["timeout"] == 17.0
    assert captured["max_retries"] == 2
    assert "PIONEER_API_KEY" not in os.environ


def test_api_mode_supports_lightweight_public_api_client(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeGLiNER2API(_FakeModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(
        client_module,
        "_import_gliner2",
        lambda mode="local": (FakeGLiNER2API, ""),
    )
    monkeypatch.setenv("PIONEER_API_KEY", "test-secret")

    client = client_module.GLiNER2Client({"gliner2_mode": "api"})

    assert client.is_available() is True
    assert isinstance(client.model, FakeGLiNER2API)
    assert captured["api_key"] == "test-secret"


def test_local_cpu_skips_gpu_only_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    model = _FakeModel()

    class FakeGLiNER2:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> _FakeModel:
            captured["name"] = name
            captured.update(kwargs)
            return model

    monkeypatch.setattr(
        client_module,
        "_import_gliner2",
        lambda mode="local": (FakeGLiNER2, ""),
    )
    monkeypatch.setattr(
        client_module,
        "_get_torch_runtime_status",
        lambda: {
            "installed": True,
            "version": "test",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": "",
        },
    )

    client = client_module.GLiNER2Client(
        {
            "gliner2_mode": "local",
            "gliner2_device": "auto",
            "gliner2_quantize": True,
            "gliner2_compile": True,
        }
    )

    assert client.is_available() is True
    assert captured["map_location"] == "cpu"
    assert "quantize" not in captured
    assert "compile" not in captured
    assert "skipped" in client.last_warning


def test_all_task_wrappers_forward_threshold_and_max_len(monkeypatch) -> None:
    model = _FakeModel()

    class FakeGLiNER2:
        @classmethod
        def from_api(cls, **kwargs: Any) -> _FakeModel:
            return model

    monkeypatch.setattr(
        client_module,
        "_import_gliner2",
        lambda mode="local": (FakeGLiNER2, ""),
    )
    monkeypatch.setenv("PIONEER_API_KEY", "test-secret")
    client = client_module.GLiNER2Client({"gliner2_mode": "api"})

    client.extract_entities("text", ["person"], threshold=0.71, max_len=512)
    client.classify_text(
        "text", {"sentiment": ["positive", "negative"]}, threshold=0.72, max_len=513
    )
    client.extract_json("text", {"product": ["name"]}, threshold=0.73, max_len=514)
    client.extract_relations("text", ["works_for"], threshold=0.74, max_len=515)

    assert [call[2]["threshold"] for call in model.calls] == [0.71, 0.72, 0.73, 0.74]
    assert [call[2].get("max_len") for call in model.calls] == [512, 513, 514, 515]


def test_unsupported_kwargs_are_filtered_for_api_compatibility() -> None:
    captured: dict[str, Any] = {}

    def old_method(text: str, schema: list[str], threshold: float = 0.5) -> dict:
        captured.update({"text": text, "schema": schema, "threshold": threshold})
        return {"ok": True}

    result = client_module._call_with_supported_kwargs(
        old_method,
        "source",
        ["person"],
        threshold=0.8,
        include_spans=True,
        max_len=512,
    )

    assert result == {"ok": True}
    assert captured == {
        "text": "source",
        "schema": ["person"],
        "threshold": 0.8,
    }


def test_explicit_cuda_without_cuda_fails_closed(monkeypatch) -> None:
    class FakeGLiNER2:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeModel:
            raise AssertionError("Model loading must not be attempted")

    monkeypatch.setattr(
        client_module,
        "_import_gliner2",
        lambda mode="local": (FakeGLiNER2, ""),
    )
    monkeypatch.setattr(
        client_module,
        "_get_torch_runtime_status",
        lambda: {
            "installed": True,
            "version": "test",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": "",
        },
    )

    client = client_module.GLiNER2Client(
        {"gliner2_mode": "local", "gliner2_device": "cuda"}
    )

    assert client.is_available() is False
    assert "CUDA was requested" in client.last_error
