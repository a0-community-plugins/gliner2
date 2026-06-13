from __future__ import annotations

import asyncio
import ast
import json
from typing import Any

from helpers.extension import Extension
from usr.plugins.gliner2 import hooks
from usr.plugins.gliner2.helpers.gliner2_client import get_client


KEYWORD_PROMPT_MARKER = "Memory Keyword Extraction System"
KEYWORD_CONTENT_MARKER = "**Memory Content:**"
RECALL_QUERY_MARKER = "provide a search query for search engine"
RECALL_USER_MARKER = "## User message:"
RECALL_HISTORY_MARKER = "## Conversation history for context:"
FILTER_PROMPT_MARKER = "array of indices of relevant memories"
FILTER_MEMORIES_MARKER = "## Memories and solutions:"
FILTER_USER_MARKER = "## User message:"
FILTER_HISTORY_MARKER = "## History for context:"
CONSOLIDATION_PROMPT_MARKER = "Memory Consolidation Analysis System"
CONSOLIDATION_MEMORY_MARKER = "**New Memory to Process**:"
CONSOLIDATION_METADATA_MARKER = "**New Memory Metadata**:"
CONSOLIDATION_SIMILAR_MARKER = "**Existing Similar Memories**:"


def _get_call_text(data: dict[str, Any], key: str, position: int) -> str:
    call_kwargs = data.get("kwargs")
    if isinstance(call_kwargs, dict) and key in call_kwargs:
        return str(call_kwargs.get(key) or "")

    args = data.get("args")
    if isinstance(args, tuple) and len(args) > position:
        return str(args[position] or "")

    return ""


def _after(text: str, marker: str) -> str:
    index = text.find(marker)
    if index < 0:
        return text.strip()
    return text[index + len(marker):].strip()


def _between(text: str, start_marker: str, end_marker: str) -> str:
    tail = _after(text, start_marker)
    index = tail.find(end_marker)
    if index < 0:
        return tail.strip()
    return tail[:index].strip()


def _is_memory_keyword_call(system: str, message: str) -> bool:
    return KEYWORD_PROMPT_MARKER in system and KEYWORD_CONTENT_MARKER in message


def _is_recall_query_call(system: str, message: str) -> bool:
    return RECALL_QUERY_MARKER in system and RECALL_USER_MARKER in message


def _is_memory_filter_call(system: str, message: str) -> bool:
    return (
        FILTER_PROMPT_MARKER in system
        and FILTER_MEMORIES_MARKER in message
        and FILTER_USER_MARKER in message
    )


def _is_consolidation_call(system: str, message: str) -> bool:
    return (
        CONSOLIDATION_PROMPT_MARKER in system
        and CONSOLIDATION_MEMORY_MARKER in message
        and CONSOLIDATION_SIMILAR_MARKER in message
    )


def _extract_memory_content(message: str) -> str:
    return _after(message, KEYWORD_CONTENT_MARKER)


def _extract_recall_content(message: str) -> str:
    user_message = _between(message, RECALL_USER_MARKER, RECALL_HISTORY_MARKER)
    history = _after(message, RECALL_HISTORY_MARKER)
    return "\n\n".join(part for part in [user_message, history] if part.strip())


def _extract_filter_context(message: str) -> tuple[str, str]:
    user_message = _between(message, FILTER_USER_MARKER, FILTER_HISTORY_MARKER)
    history = _after(message, FILTER_HISTORY_MARKER)
    return user_message, history


def _extract_filter_memories(message: str) -> dict[int, str] | None:
    raw = _between(message, FILTER_MEMORIES_MARKER, FILTER_USER_MARKER)
    if not raw:
        return None

    try:
        parsed = ast.literal_eval(raw.strip())
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    memories: dict[int, str] = {}
    for raw_key, raw_value in parsed.items():
        try:
            index = int(raw_key)
        except (TypeError, ValueError):
            continue
        memories[index] = str(raw_value or "").strip()

    return memories or None


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _confidence_from_any(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("confidence", "score", "probability", "prob"):
            if key in value:
                return _confidence_from_any(value.get(key))
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _extract_label_and_confidence(
    result: dict[str, Any] | None,
    field_name: str,
) -> tuple[str, float | None]:
    if not isinstance(result, dict):
        return "", None

    candidates = []
    if field_name in result:
        candidates.append(result.get(field_name))
    for key in ("label", "classification", "class", "value", "text"):
        if key in result:
            candidates.append(result.get(key))

    for candidate in candidates:
        if isinstance(candidate, dict):
            label = ""
            for key in ("label", "classification", "class", "value", "text"):
                if candidate.get(key) is not None:
                    label = _normalize_label(candidate.get(key))
                    break
            confidence = _confidence_from_any(candidate)
            if label:
                return label, confidence
            continue

        if isinstance(candidate, list):
            best_label = ""
            best_confidence: float | None = None
            for item in candidate:
                item_label, item_confidence = _extract_label_and_confidence(
                    {field_name: item}, field_name
                )
                if not item_label:
                    continue
                if best_confidence is None or (item_confidence or 1.0) > best_confidence:
                    best_label = item_label
                    best_confidence = item_confidence
            if best_label:
                return best_label, best_confidence
            continue

        if candidate is not None:
            return _normalize_label(candidate), _confidence_from_any(result)

    return "", _confidence_from_any(result)


def _classify(
    client: Any,
    text: str,
    field_name: str,
    labels: list[str],
) -> tuple[str, float | None]:
    result = client.classify_text(
        text=text,
        schema={field_name: labels},
        include_confidence=True,
    )
    return _extract_label_and_confidence(result, field_name)


def _extract_entities(
    agent: Any, config: dict[str, Any], text: str
) -> list[str] | None:
    if not text.strip():
        return None

    client = get_client(config)
    if not client.is_available(load_model=False):
        return None

    result = client.extract_entities(
        text=text,
        schema=config.get("gliner2_memory_entity_types", []) or [],
        threshold=float(config.get("gliner2_entity_threshold", 0.5) or 0.5),
    )
    entities = hooks._flatten_entities(result)
    return entities or None


def _get_available_client(config: dict[str, Any]) -> Any | None:
    client = get_client(config)
    if not client.is_available(load_model=False):
        return None
    return client


def _log_gliner_usage(
    agent: Any,
    config: dict[str, Any],
    feature: str,
    detail: str,
    kvps: dict[str, Any] | None = None,
) -> None:
    if not config.get("gliner2_usage_logging", True):
        return

    try:
        context = getattr(agent, "context", None)
        log = getattr(context, "log", None)
        if log is None:
            return
        log.log(
            type="util",
            heading=f"GLiNER2 used: {feature}",
            content=detail,
            kvps={
                "plugin": "gliner2",
                "mode": config.get("gliner2_mode", "local"),
                **(kvps or {}),
            },
        )
    except Exception:
        return


def _filter_relevant_memories(config: dict[str, Any], message: str) -> str | None:
    memories = _extract_filter_memories(message)
    if not memories:
        return None

    client = _get_available_client(config)
    if client is None:
        return None

    threshold = float(config.get("gliner2_post_filter_threshold", 0.5) or 0.5)
    user_message, history = _extract_filter_context(message)
    relevant_indices: list[int] = []
    saw_confident_result = False

    for index in sorted(memories):
        memory_text = memories[index]
        if not memory_text:
            continue

        classification_text = "\n\n".join(
            part
            for part in [
                f"User message:\n{user_message}",
                f"Conversation history:\n{history}",
                f"Memory or solution:\n{memory_text}",
            ]
            if part.strip()
        )
        label, confidence = _classify(
            client,
            classification_text,
            "relevance",
            ["relevant", "irrelevant"],
        )
        if not label:
            continue

        confidence = 1.0 if confidence is None else confidence
        if confidence >= threshold:
            saw_confident_result = True
            if label in {"relevant", "yes", "include", "included"}:
                relevant_indices.append(index)

    if not saw_confident_result:
        return None

    return json.dumps(relevant_indices)


def _extract_new_memory_for_consolidation(message: str) -> str:
    return _between(
        message,
        CONSOLIDATION_MEMORY_MARKER,
        CONSOLIDATION_METADATA_MARKER,
    )


def _consolidation_result(
    action: str,
    new_memory: str,
    reasoning: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "action": action,
            "memories_to_remove": [],
            "memories_to_update": [],
            "new_memory_content": new_memory,
            "metadata": metadata or {},
            "reasoning": reasoning,
        }
    )


def _triage_consolidation(config: dict[str, Any], message: str) -> str | None:
    client = _get_available_client(config)
    if client is None:
        return None

    threshold = float(
        config.get("gliner2_consolidation_triage_threshold", 0.65) or 0.65
    )
    label, confidence = _classify(
        client,
        message,
        "action",
        ["keep_separate", "skip", "merge", "replace", "update"],
    )
    if not label:
        return None

    confidence = 1.0 if confidence is None else confidence
    if confidence < threshold:
        return None

    new_memory = _extract_new_memory_for_consolidation(message)
    if label in {"keep_separate", "skip"}:
        return _consolidation_result(
            label,
            new_memory,
            "GLiNER2 classified this as safe for non-generative consolidation triage.",
            {"gliner2_triaged": True, "gliner2_confidence": confidence},
        )

    return None


async def _run_with_timeout(config: dict[str, Any], func, *args) -> Any | None:
    try:
        timeout_seconds = int(config.get("gliner2_operation_timeout_seconds", 30) or 30)
    except (TypeError, ValueError):
        timeout_seconds = 30

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args),
            timeout=max(1, min(600, timeout_seconds)),
        )
    except TimeoutError:
        return None


class GLiNER2MemoryUtility(Extension):
    async def execute(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if self.agent is None or not isinstance(data, dict):
            return

        config = hooks._get_config(agent=self.agent)
        if not config.get("gliner2_enabled", True):
            return
        if not config.get("gliner2_utility_replacement_enabled", True):
            return

        system = _get_call_text(data, "system", 1)
        message = _get_call_text(data, "message", 2)
        fallback_to_utility = bool(
            config.get("gliner2_fallback_to_utility_model", True)
        )

        if _is_memory_keyword_call(system, message):
            if not config.get("gliner2_memory_keyword_extraction", True):
                return
            entities = await _run_with_timeout(
                config,
                _extract_entities,
                self.agent, config, _extract_memory_content(message)
            )
            if entities:
                data["result"] = json.dumps(entities)
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory keyword extraction",
                    "Replaced the configured Utility model with GLiNER2 entity extraction.",
                    {
                        "entities_found": len(entities),
                        "threshold": config.get("gliner2_entity_threshold", 0.5),
                    },
                )
            elif not fallback_to_utility:
                data["result"] = "[]"
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory keyword extraction",
                    (
                        "GLiNER2 handled the keyword request with an empty result "
                        "because Utility model fallback is disabled."
                    ),
                    {"entities_found": 0},
                )
            return

        if _is_recall_query_call(system, message):
            if not config.get("gliner2_recall_query_enrichment", False):
                return
            entities = await _run_with_timeout(
                config,
                _extract_entities,
                self.agent, config, _extract_recall_content(message)
            )
            if entities:
                data["result"] = " ".join(entities)
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory recall query",
                    "Replaced the configured Utility model with GLiNER2 entity extraction.",
                    {
                        "entities_found": len(entities),
                        "threshold": config.get("gliner2_entity_threshold", 0.5),
                    },
                )
            elif not fallback_to_utility:
                data["result"] = "-"
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory recall query",
                    (
                        "GLiNER2 handled the recall query with a skip marker "
                        "because Utility model fallback is disabled."
                    ),
                    {"entities_found": 0},
                )
            return

        if _is_memory_filter_call(system, message):
            if not config.get("gliner2_memory_post_filter", True):
                return
            result = await _run_with_timeout(
                config, _filter_relevant_memories, config, message
            )
            if result is not None:
                data["result"] = result
                try:
                    selected_count = len(json.loads(result))
                except Exception:
                    selected_count = 0
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory post-filtering",
                    "Replaced the configured Utility model with GLiNER2 relevance classification.",
                    {
                        "selected_indices": selected_count,
                        "threshold": config.get("gliner2_post_filter_threshold", 0.5),
                    },
                )
            elif not fallback_to_utility:
                data["result"] = "[]"
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory post-filtering",
                    (
                        "GLiNER2 handled the post-filter with an empty result "
                        "because Utility model fallback is disabled."
                    ),
                    {"selected_indices": 0},
                )
            return

        if _is_consolidation_call(system, message):
            if not config.get("gliner2_consolidation_triage", True):
                return
            result = await _run_with_timeout(
                config, _triage_consolidation, config, message
            )
            if result is not None:
                data["result"] = result
                try:
                    parsed_result = json.loads(result)
                    action = parsed_result.get("action", "")
                    confidence = parsed_result.get("metadata", {}).get(
                        "gliner2_confidence", ""
                    )
                except Exception:
                    action = ""
                    confidence = ""
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory consolidation triage",
                    (
                        "Replaced the configured Utility model with GLiNER2 "
                        "classification for a non-generative consolidation decision."
                    ),
                    {
                        "action": action,
                        "confidence": confidence,
                        "threshold": config.get(
                            "gliner2_consolidation_triage_threshold", 0.65
                        ),
                    },
                )
            elif not fallback_to_utility:
                data["result"] = _consolidation_result(
                    "skip",
                    _extract_new_memory_for_consolidation(message),
                    (
                        "GLiNER2 could not confidently perform non-generative "
                        "consolidation triage and Utility model fallback is disabled."
                    ),
                    {"gliner2_triaged": False},
                )
                _log_gliner_usage(
                    self.agent,
                    config,
                    "memory consolidation triage",
                    (
                        "GLiNER2 did not make a confident triage decision, so the "
                        "plugin returned a conservative skip because Utility model "
                        "fallback is disabled."
                    ),
                    {
                        "action": "skip",
                        "threshold": config.get(
                            "gliner2_consolidation_triage_threshold", 0.65
                        ),
                    },
                )
