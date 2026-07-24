from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_and_defaults_describe_v2_runtime() -> None:
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    defaults = yaml.safe_load((ROOT / "default_config.yaml").read_text())

    assert manifest["name"] == "gliner2"
    assert manifest["version"] == "2.0.0"
    assert defaults["gliner2_device"] == "auto"
    assert defaults["gliner2_max_text_chars"] == 50_000
    assert defaults["gliner2_api_key_env"] == "PIONEER_API_KEY"


def test_publication_files_and_runtime_ignores_exist() -> None:
    ignored = (ROOT / ".gitignore").read_text()

    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "README.md").is_file()
    assert "config.json" in ignored
    assert ".toggle-1" in ignored
    assert not (ROOT / "execute.py").exists()


def test_ui_avoids_unbounded_horizontal_content() -> None:
    html = (ROOT / "webui/config.html").read_text()

    assert "max-width: 100%" in html
    assert "overflow-wrap: anywhere" in html
    assert "Raw runtime diagnostics" in html
    assert "Install / repair" in html
