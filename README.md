# GLiNER2 Plugin

`gliner2` adds local or API-backed schema-driven information extraction to Agent Zero.
The runtime plugin directory is `usr/plugins/gliner2`, and the manifest `name`
must stay `gliner2` to match Agent Zero plugin discovery rules.

What it does:
- exposes a direct `gliner2_extract` tool for entities, classification, JSON extraction, and relations
- exposes a framework extension that can replace eligible memory utility-model calls with GLiNER2 entity extraction
- exposes framework hooks for memory keyword extraction and knowledge metadata enrichment
- includes a plugin settings panel with runtime status, install actions, and a sample extraction workflow
- normalizes saved settings so booleans, thresholds, modes, and entity type lists stay runtime-safe

Installation:
- local mode: `python -m pip install 'gliner2[local]'`
- api mode: `python -m pip install gliner2`

The settings panel install action runs through `hooks.py` using
`sys.executable -m pip`, so dependencies are installed into the Agent Zero
framework Python runtime. In Docker this is `/opt/venv-a0`, not the separate
agent execution runtime at `/opt/venv`.

Docker and GPU:
1. Start Agent Zero with Docker GPU passthrough, for example add `--gpus all` to your `docker run` command.
2. In the GLiNER plugin, select `local` mode and run the install action. This installs inside the Agent Zero framework runtime in the container.
3. Refresh the GLiNER status panel and confirm `torch_cuda_available` is `true`.
4. If CUDA is still unavailable, install a CUDA-enabled Linux pip build of PyTorch inside the container using the command from PyTorch Start Locally: https://docs.pytorch.org/get-started/locally/

Notes:
- GLiNER2’s README says `pip install gliner2[local]` is the local inference package and enables quantization / `torch.compile` GPU options.
- PyTorch’s Linux install docs say to choose the CUDA version that matches your machine, then verify with `torch.cuda.is_available()`.

Configuration:
- `gliner2_enabled`: master enable switch
- `gliner2_mode`: `local` or `api`
- `gliner2_model`: local Hugging Face model id
- `gliner2_api_key_env`: environment variable that holds the Pioneer API key
- `gliner2_quantize`: request quantized local loading on CUDA
- `gliner2_compile`: request `torch.compile` on CUDA
- `gliner2_utility_replacement_enabled`: master switch for replacing eligible memory utility-model calls
- `gliner2_fallback_to_utility_model`: allow the selected Utility model to run when GLiNER2 cannot produce a replacement
- `gliner2_usage_logging`: show a util log entry whenever GLiNER2 replaces an eligible Utility-model memory call
- `gliner2_memory_keyword_extraction`: let GLiNER2 provide memory-search keywords
- `gliner2_recall_query_enrichment`: let GLiNER2 produce memory recall queries when `_memory` query prep is enabled
- `gliner2_memory_post_filter`: let GLiNER2 classify recalled memories and solutions for relevance
- `gliner2_post_filter_threshold`: minimum confidence for accepting a post-filter relevance decision
- `gliner2_consolidation_triage`: let GLiNER2 skip or keep separate obvious consolidation cases
- `gliner2_consolidation_triage_threshold`: minimum confidence for accepting a consolidation triage decision
- `gliner2_knowledge_import_enrichment`: add structured entity metadata during knowledge import
- `gliner2_tool_enabled`: allow direct agent tool use
- `gliner2_memory_entity_types`: entity labels used when the tool or hook needs a default entities schema
- `gliner2_import_entity_types`: entity labels used for knowledge import metadata enrichment

Tool prompt:
- the discoverable prompt file is `prompts/agent.system.tool.gliner2_extract.md`
- Agent Zero only loads tool prompts matching `agent.system.tool.*.md`

Framework hooks:
- `extensions/python/_functions/agent/Agent/call_utility_model/start/_10_gliner2_memory_utility.py` short-circuits only GLiNER-compatible memory utility calls
- `provide_memory_keywords(agent, text)` returns flattened entities when GLiNER2 is enabled and available
- `enrich_knowledge_metadata(agent, text, metadata, log_item)` returns GLiNER2 entity metadata when enabled and available
- `get_plugin_config` and `save_plugin_config` normalize plugin settings loaded through Agent Zero
- `install(mode, config)` installs either `gliner2[local]` or `gliner2`
- `pre_update()` is present for Plugin Hub update compatibility

Utility model replacement is intentionally scoped. GLiNER2 can replace memory
keyword extraction, memory recall query prep, memory post-filter relevance
classification, and safe consolidation triage because those calls can be
answered by extraction or classification. It does not replace general Utility
model work such as summarization, behavior merging, or consolidation actions
that need rewritten memory content. Merge, replace, and update consolidation
decisions still fall through to the configured Utility model when fallback is
enabled. If `gliner2_fallback_to_utility_model` is off, eligible memory calls
return conservative non-generative defaults instead of falling through to the
configured Utility model.

Observability:
- when `gliner2_usage_logging` is enabled, each replacement writes a `util`
  log entry headed `GLiNER2 used: ...`
- log details include the plugin id, mode, feature, thresholds, selected counts,
  entity counts, or triage action where applicable
- if GLiNER2 cannot handle a call and Utility fallback is enabled, no GLiNER2
  usage log is written because the configured Utility model handled the call

Privacy:
- local mode keeps extraction on the local machine
- API mode uses the Pioneer-hosted GLiNER2 API and requires an API key

Troubleshooting:
- if status says the package is missing, use the install action in plugin settings or run the pip command manually
- `model_loaded: false` means no GLiNER model object has been initialized in the current Agent Zero process yet; use Load Model or run Sample Extraction to initialize it
- if you are in Docker and want GPU, make sure the container was started with GPU access before troubleshooting PyTorch inside it
- if `torch_cuda_available` is `false`, install or reinstall the CUDA-enabled Linux pip build of PyTorch inside the container and refresh status
- if local model loading fails, try disabling quantize/compile first
- if API mode fails, verify the configured environment variable exists and contains a valid Pioneer API key

Community plugin notes:
- for Plugin Index publication, keep this plugin as a standalone repository with `plugin.yaml` at the repository root
- include a repository-level `LICENSE` before submitting to the Plugin Index
- do not publish local `config.json` values or secrets
