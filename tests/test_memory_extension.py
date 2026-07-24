from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


EXTENSION_PATH = (
    Path(__file__).resolve().parents[1]
    / "extensions/python/_functions/agent/Agent/call_utility_model/start"
    / "_10_gliner2_memory_utility.py"
)


def _load_extension_module():
    spec = importlib.util.spec_from_file_location("gliner2_memory_extension_test", EXTENSION_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracts_label_and_confidence_from_current_result_shape() -> None:
    module = _load_extension_module()

    label, confidence = module._extract_label_and_confidence(
        {"relevance": {"label": "relevant", "confidence": 0.91}},
        "relevance",
    )

    assert label == "relevant"
    assert confidence == 0.91


def test_missing_confidence_falls_through_instead_of_assuming_certainty(
    monkeypatch,
) -> None:
    module = _load_extension_module()

    class FakeClient:
        def is_available(self, load_model: bool = True) -> bool:
            return True

        def classify_text(self, **kwargs: Any) -> dict[str, Any]:
            return {"relevance": "relevant"}

    monkeypatch.setattr(module, "get_client", lambda config: FakeClient())
    message = (
        "## Memories and solutions:\n"
        "{0: 'Ada wrote notes about the Analytical Engine.'}\n"
        "## User message:\nTell me about Ada.\n"
        "## History for context:\n"
    )

    result = module._filter_relevant_memories(
        {
            "gliner2_post_filter_threshold": 0.5,
            "gliner2_max_text_chars": 50_000,
            "gliner2_max_len": 0,
        },
        message,
    )

    assert result is None
