from __future__ import annotations

from usr.plugins.gliner2.helpers.config import (
    DEFAULT_ENTITY_TYPES,
    HARD_MAX_SCHEMA_ITEMS,
    HARD_MAX_TEXT_CHARS,
    normalize_config,
    normalize_schema_for_task,
    validate_text,
)


def test_normalize_config_preserves_valid_saved_values() -> None:
    config = normalize_config(
        {
            "gliner2_mode": "API",
            "gliner2_device": "CPU",
            "gliner2_enabled": "false",
            "gliner2_entity_threshold": "0.77",
            "gliner2_api_timeout_seconds": "45",
            "gliner2_operation_timeout_seconds": "42",
            "gliner2_api_max_retries": "5",
            "gliner2_max_text_chars": "75000",
            "gliner2_memory_entity_types": '["person", "Person", "event"]',
        }
    )

    assert config["gliner2_mode"] == "api"
    assert config["gliner2_device"] == "cpu"
    assert config["gliner2_enabled"] is False
    assert config["gliner2_entity_threshold"] == 0.77
    assert config["gliner2_api_timeout_seconds"] == 45.0
    assert config["gliner2_operation_timeout_seconds"] == 42
    assert config["gliner2_api_max_retries"] == 5
    assert config["gliner2_max_text_chars"] == 75_000
    assert config["gliner2_memory_entity_types"] == ["person", "event"]


def test_normalize_config_clamps_limits_and_rejects_invalid_env_name() -> None:
    config = normalize_config(
        {
            "gliner2_api_key_env": "bad env;name",
            "gliner2_max_text_chars": 999_999,
            "gliner2_max_schema_items": 999_999,
            "gliner2_max_len": -3,
        }
    )

    assert config["gliner2_api_key_env"] == "PIONEER_API_KEY"
    assert config["gliner2_max_text_chars"] == HARD_MAX_TEXT_CHARS
    assert config["gliner2_max_schema_items"] == HARD_MAX_SCHEMA_ITEMS
    assert config["gliner2_max_len"] == 0


def test_normalize_config_uses_safe_defaults() -> None:
    config = normalize_config(None)

    assert config["gliner2_mode"] == "local"
    assert config["gliner2_device"] == "auto"
    assert config["gliner2_memory_entity_types"] == DEFAULT_ENTITY_TYPES
    assert config["gliner2_fallback_to_utility_model"] is True


def test_validate_text_enforces_configured_bound() -> None:
    config = normalize_config({"gliner2_max_text_chars": 1_000})

    value, error = validate_text("x" * 1_001, config)

    assert value == ""
    assert "1,001 characters" in str(error)


def test_entities_schema_uses_configured_defaults() -> None:
    config = normalize_config({"gliner2_memory_entity_types": ["person", "event"]})

    schema, error = normalize_schema_for_task("entities", None, config)

    assert error is None
    assert schema == ["person", "event"]


def test_json_schema_requires_field_arrays() -> None:
    config = normalize_config(None)

    schema, error = normalize_schema_for_task(
        "json",
        {"product": {"name": "str"}},
        config,
    )

    assert schema is None
    assert "array" in str(error)


def test_schema_item_limit_is_enforced() -> None:
    config = normalize_config({"gliner2_max_schema_items": 3})

    schema, error = normalize_schema_for_task(
        "classify",
        {"sentiment": ["positive", "negative", "neutral"]},
        config,
    )

    assert schema is None
    assert "4 items" in str(error)
