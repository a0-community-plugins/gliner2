# GLiNER2 for Agent Zero

GLiNER2 gives Agent Zero a purpose-built structured inference path for:

- entity extraction
- text classification
- typed JSON extraction
- relation extraction
- memory keywords, relevance filtering, and conservative consolidation triage
- knowledge-import metadata enrichment

It runs either with a local checkpoint or through the GLiNER2 hosted API. The
plugin does not modify Agent Zero core.

## Install

Install from Agent Zero's Plugin Hub. The Plugin Hub invokes `hooks.py`
automatically, and the hook installs the supported runtime into Agent Zero's
framework Python environment:

- local: `gliner2[local]>=1.3.2,<2`
- API: `gliner2>=1.3.2,<2`

The default route is local, so a normal Plugin Hub install prepares the local
runtime in one flow. The settings panel also provides an **Install / repair**
action for a changed mode or interrupted dependency install.

The hook deliberately uses `sys.executable -m pip`. In the official Docker
image that targets `/opt/venv-a0`, where plugin hooks and the Agent Zero backend
run. It does not install into the separate `/opt/venv` agent execution
environment.

## Start here

1. Open **Settings → Plugins → GLiNER2**.
2. Choose **Local runtime** or **Hosted API**.
3. Confirm that the readiness console shows no blockers.
4. Select **Load model** or **Connect API**. Local model startup runs in the
   background; the console reports `Starting` until it is ready.
5. Run the built-in test bench before enabling memory replacements broadly.

Saved global, project, and agent-scoped settings are honored. Version 2.0
removes the old config-read hook that accidentally replaced loaded settings
with defaults.

## Local mode

The default model is `fastino/gliner2-base-v1`.

Device options:

- **Auto** uses CUDA when PyTorch can see a CUDA device, otherwise CPU.
- **CPU** explicitly keeps inference on CPU.
- **CUDA** fails closed with a clear blocker when CUDA is unavailable.

Quantization and `torch.compile` are applied only on CUDA. When either option is
enabled on CPU, the runtime skips it and reports a warning instead of attempting
an invalid model load.

### Docker GPU access

The plugin can detect GPU visibility but cannot grant a running container new
host-device permissions. Recreate Agent Zero with GPU passthrough—commonly
`--gpus all`—then confirm the GLiNER2 console reports CUDA. CPU mode remains a
fully supported setup and requires no Docker socket.

No Docker socket is required by this plugin.

## API mode

API mode installs the lightweight base package and reads the credential from an
environment variable. `PIONEER_API_KEY` is the default.

The plugin passes the credential directly to the public `GLiNER2API` client. It never
copies a custom credential into `PIONEER_API_KEY`, persists the secret in
`config.json`, or displays the secret in diagnostics.

You can optionally configure:

- a custom API base URL
- request timeout
- retry count

API mode sends extraction text and schemas to the configured endpoint. Local
mode keeps extraction in the Agent Zero runtime.

## Agent tool

The discoverable tool is `gliner2_extract`.

Arguments:

- `task`: `entities`, `classify`, `json`, or `relations`
- `text`: source text
- `schema`: task-specific JSON array or object
- `include_confidence`: include model confidence where supported
- `include_spans`: include source spans where supported

Examples:

```json
{
  "task": "entities",
  "text": "Ada Lovelace worked with Charles Babbage in London.",
  "schema": ["person", "location"],
  "include_confidence": true
}
```

```json
{
  "task": "classify",
  "text": "The release is stable and dramatically faster.",
  "schema": {
    "sentiment": ["positive", "negative", "neutral"]
  }
}
```

All four task wrappers forward the configured extraction threshold. Local
methods also receive the optional model maximum length; API methods safely
ignore unsupported keyword arguments.

## Memory integration

The framework extension intercepts only utility-model calls whose current
Agent Zero prompt shape is compatible with extraction or classification:

- memory keyword extraction
- recall-query enrichment
- recalled-memory relevance filtering
- conservative consolidation triage

Generative merge, replace, and update decisions still fall through to the
configured Utility model. Missing confidence also falls through; it is never
treated as certainty.

When **Safe Utility fallback** is disabled, the extension returns conservative
empty/skip results for eligible calls it cannot answer. This is useful for
fully non-generative paths, but the default is to keep fallback enabled.

When **Usage observability** is enabled, successful replacements appear as
`util` log entries with the feature, mode, counts, thresholds, and timing-safe
metadata. Source text and API credentials are not added to those entries.

## Runtime safety

Version 2.0 adds:

- bounded input text, schema size, and optional model length
- serialized model load and inference access
- non-blocking background model startup with visible loading state
- background execution for async tools, diagnostics, and utility extensions
- a bounded operation wait before explicit extraction or Utility replacement
  falls back
- a bounded in-process client cache
- explicit CUDA readiness checks
- package compatibility reporting
- sanitized dependency-install output
- case-insensitive entity deduplication across current GLiNER2 result shapes

The default limits are 50,000 text characters and 100 nested schema items per
call. The settings UI can tune these within hard ceilings.

## Configuration reference

Core runtime:

- `gliner2_enabled`
- `gliner2_mode`
- `gliner2_model`
- `gliner2_device`
- `gliner2_api_key_env`
- `gliner2_api_base_url`
- `gliner2_api_timeout_seconds`
- `gliner2_api_max_retries`
- `gliner2_quantize`
- `gliner2_compile`
- `gliner2_entity_threshold`
- `gliner2_max_len`
- `gliner2_max_text_chars`
- `gliner2_max_schema_items`
- `gliner2_operation_timeout_seconds`

Agent Zero integrations:

- `gliner2_tool_enabled`
- `gliner2_utility_replacement_enabled`
- `gliner2_fallback_to_utility_model`
- `gliner2_usage_logging`
- `gliner2_memory_keyword_extraction`
- `gliner2_recall_query_enrichment`
- `gliner2_memory_post_filter`
- `gliner2_post_filter_threshold`
- `gliner2_consolidation_triage`
- `gliner2_consolidation_triage_threshold`
- `gliner2_knowledge_import_enrichment`
- `gliner2_memory_entity_types`
- `gliner2_import_entity_types`

## Development and verification

Run the focused suite from an Agent Zero checkout or from this standalone
repository:

```bash
pytest -q
```

The tests use fake GLiNER2 runtimes; they do not download a model, install
dependencies, or call the hosted API. GitHub Actions runs the suite on Python
3.11, 3.12, and 3.13.

For a delivery check, open the live settings panel, verify the readiness cards
at narrow and wide widths, and run the test bench against the route you intend
to use.

## Upstream

This integration tracks the public
[fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) API and currently
supports the stable 1.x line from 1.3.2 onward. The upper bound prevents a future
major API change from silently breaking an installed plugin.

Licensed under Apache License 2.0.
