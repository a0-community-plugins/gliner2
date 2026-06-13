from __future__ import annotations

import inspect
import os
import threading
import time
from copy import deepcopy
from typing import Any


_CLIENTS: dict[tuple[Any, ...], "GLiNER2Client"] = {}
_CLIENTS_LOCK = threading.Lock()


def _import_gliner2() -> tuple[type[Any] | None, str]:
    try:
        from gliner2 import GLiNER2  # type: ignore

        return GLiNER2, ""
    except Exception as e:
        return None, str(e)


def _config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config.get("gliner2_mode", "local"),
        config.get("gliner2_model", "fastino/gliner2-base-v1"),
        config.get("gliner2_api_key_env", "PIONEER_API_KEY"),
        bool(config.get("gliner2_quantize", False)),
        bool(config.get("gliner2_compile", False)),
    )


def _operation_timeout_seconds(config: dict[str, Any]) -> int:
    try:
        timeout = int(config.get("gliner2_operation_timeout_seconds", 30) or 30)
    except (TypeError, ValueError):
        timeout = 30
    return max(1, min(600, timeout))


def _get_torch_runtime_status() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        return {
            "installed": True,
            "version": getattr(torch, "__version__", ""),
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cuda_device_name": (
                str(torch.cuda.get_device_name(0))
                if cuda_available and device_count > 0
                else ""
            ),
            "error": "",
        }
    except Exception as e:
        return {
            "installed": False,
            "version": "",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": str(e),
        }


def get_runtime_status(
    config: dict[str, Any] | None,
    load_model: bool = False,
) -> dict[str, Any]:
    cfg = config or {}
    mode = str(cfg.get("gliner2_mode", "local") or "local")
    model_name = str(cfg.get("gliner2_model", "fastino/gliner2-base-v1") or "fastino/gliner2-base-v1")
    api_key_env = str(cfg.get("gliner2_api_key_env", "PIONEER_API_KEY") or "PIONEER_API_KEY")
    gliner_cls, import_error = _import_gliner2()
    client = get_client(cfg)
    model_load_started = False
    if load_model and gliner_cls is not None:
        model_load_started = client.start_loading()
    torch_status = _get_torch_runtime_status()
    in_docker = os.path.exists("/.dockerenv")
    model_loaded = client.is_loaded()
    model_loading = client.is_loading()
    model_loading_seconds = client.loading_seconds()
    operation_timeout_seconds = _operation_timeout_seconds(cfg)
    model_load_slow = (
        model_loading_seconds is not None
        and model_loading_seconds >= operation_timeout_seconds
    )
    model_state = "loaded" if model_loaded else "not_loaded"
    if import_error:
        model_state = "package_missing"
    elif model_loading:
        model_state = "loading"
    elif client.last_error:
        model_state = "load_failed"

    return {
        "ok": True,
        "plugin": "gliner2",
        "enabled": bool(cfg.get("gliner2_enabled", True)),
        "utility_replacement_enabled": bool(
            cfg.get("gliner2_utility_replacement_enabled", True)
        ),
        "fallback_to_utility_model": bool(
            cfg.get("gliner2_fallback_to_utility_model", True)
        ),
        "usage_logging": bool(cfg.get("gliner2_usage_logging", True)),
        "memory_keyword_extraction": bool(
            cfg.get("gliner2_memory_keyword_extraction", True)
        ),
        "recall_query_enrichment": bool(
            cfg.get("gliner2_recall_query_enrichment", False)
        ),
        "memory_post_filter": bool(cfg.get("gliner2_memory_post_filter", True)),
        "consolidation_triage": bool(cfg.get("gliner2_consolidation_triage", True)),
        "package_installed": gliner_cls is not None,
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "model_loading_seconds": model_loading_seconds,
        "model_load_slow": model_load_slow,
        "operation_timeout_seconds": operation_timeout_seconds,
        "model_state": model_state,
        "model_status_note": (
            "Model object is loaded in this Agent Zero process."
            if model_loaded
            else (
                "Model load is running in the background."
                if model_loading
                else (
                    "Install the GLiNER2 package before loading a model."
                    if import_error
                    else (
                        "Model is installed but not loaded yet; click Load Model "
                        "or run an explicit extraction to initialize it."
                    )
                )
            )
        ),
        "status_loaded_model": bool(load_model),
        "model_load_started": model_load_started,
        "mode": mode,
        "model_name": model_name if mode == "local" else "api",
        "api_key_env": api_key_env,
        "api_key_configured": bool(os.environ.get(api_key_env) or os.environ.get("PIONEER_API_KEY")),
        "in_docker": in_docker,
        "torch_installed": torch_status["installed"],
        "torch_version": torch_status["version"],
        "torch_cuda_available": torch_status["cuda_available"],
        "torch_cuda_device_count": torch_status["cuda_device_count"],
        "torch_cuda_device_name": torch_status["cuda_device_name"],
        "torch_error": torch_status["error"] or None,
        "error": (client.last_error if client and client.last_error else "") or import_error or None,
    }


def get_client(config: dict[str, Any] | None) -> "GLiNER2Client":
    cfg = deepcopy(config or {})
    key = _config_key(cfg)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = GLiNER2Client(cfg)
            _CLIENTS[key] = client
        return client


def get_loaded_client_or_status(
    config: dict[str, Any] | None,
) -> tuple["GLiNER2Client | None", dict[str, Any]]:
    cfg = config or {}
    status = get_runtime_status(cfg, load_model=False)
    client = get_client(cfg)

    if not status["package_installed"]:
        return None, status

    if client.is_available(load_model=False):
        return client, status

    model_load_started = client.start_loading()
    status = get_runtime_status(cfg, load_model=False)
    if model_load_started:
        status["model_load_started"] = True
    return None, status


class GLiNER2Client:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = deepcopy(config or {})
        self._lock = threading.Lock()
        self.model: Any | None = None
        self.last_error = ""
        self._import_error = ""
        self._load_thread: threading.Thread | None = None
        self._loading_started_at: float | None = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def is_loading(self) -> bool:
        thread = self._load_thread
        if thread is None:
            return self._loading_started_at is not None
        if thread.is_alive():
            return True
        self._load_thread = None
        return self._loading_started_at is not None

    def loading_seconds(self) -> float | None:
        if self._loading_started_at is None:
            return None
        return round(max(0.0, time.monotonic() - self._loading_started_at), 1)

    def is_available(self, load_model: bool = True) -> bool:
        if load_model:
            self._ensure_model()
        return self.model is not None

    def start_loading(self) -> bool:
        if self.model is not None:
            return False
        if self.is_loading():
            return False

        self.last_error = ""
        self._loading_started_at = time.monotonic()

        def _load() -> None:
            try:
                self._ensure_model()
            finally:
                if threading.current_thread() is self._load_thread:
                    self._load_thread = None

        thread = threading.Thread(target=_load, name="gliner2-model-load", daemon=True)
        self._load_thread = thread
        thread.start()
        return True

    def _ensure_model(self) -> Any | None:
        if self.model is not None:
            return self.model

        with self._lock:
            if self.model is not None:
                return self.model

            gliner_cls, import_error = _import_gliner2()
            self._import_error = import_error
            if gliner_cls is None:
                self.last_error = import_error or "gliner2 is not installed."
                self._loading_started_at = None
                return None

            self._loading_started_at = time.monotonic()
            try:
                mode = str(self.config.get("gliner2_mode", "local") or "local").lower()
                if mode == "api":
                    self._apply_api_env_override()
                    self.model = gliner_cls.from_api()
                else:
                    kwargs: dict[str, Any] = {}
                    if self.config.get("gliner2_quantize"):
                        kwargs["quantize"] = True
                        kwargs["map_location"] = "cuda"
                    if self.config.get("gliner2_compile"):
                        kwargs["compile"] = True
                        kwargs.setdefault("map_location", "cuda")
                    self.model = gliner_cls.from_pretrained(
                        str(self.config.get("gliner2_model", "fastino/gliner2-base-v1") or "fastino/gliner2-base-v1"),
                        **kwargs,
                    )
                self.last_error = ""
            except Exception as e:
                self.model = None
                self.last_error = str(e)
            finally:
                self._loading_started_at = None

        return self.model

    def _apply_api_env_override(self) -> None:
        env_name = str(self.config.get("gliner2_api_key_env", "PIONEER_API_KEY") or "PIONEER_API_KEY")
        if env_name == "PIONEER_API_KEY":
            return
        api_key = os.environ.get(env_name, "")
        if api_key:
            os.environ["PIONEER_API_KEY"] = api_key

    def _invoke(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        model = self._ensure_model()
        if model is None:
            return None

        method = getattr(model, method_name, None)
        if method is None:
            self.last_error = f"GLiNER2 method not available: {method_name}"
            return None

        try:
            signature = inspect.signature(method)
            accepted_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
        except Exception:
            accepted_kwargs = kwargs

        try:
            result = method(*args, **accepted_kwargs)
            self.last_error = ""
            return result
        except Exception as e:
            self.last_error = str(e)
            return None

    def extract_entities(
        self,
        text: str,
        schema: list[str] | dict[str, str],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> dict[str, Any] | None:
        kwargs = {
            "threshold": threshold,
            "include_confidence": include_confidence,
            "include_spans": include_spans,
        }
        return self._invoke("extract_entities", text, schema, **kwargs)

    def batch_extract_entities(
        self,
        texts: list[str],
        schema: list[str] | dict[str, str],
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> list[dict[str, Any]] | None:
        kwargs = {
            "batch_size": batch_size,
            "threshold": threshold,
            "include_confidence": include_confidence,
            "include_spans": include_spans,
        }
        result = self._invoke("batch_extract_entities", texts, schema, **kwargs)
        return result if isinstance(result, list) else None

    def classify_text(
        self,
        text: str,
        schema: dict[str, list[str]],
        include_confidence: bool = False,
    ) -> dict[str, Any] | None:
        kwargs = {"include_confidence": include_confidence}
        return self._invoke("classify_text", text, schema, **kwargs)

    def extract_json(
        self,
        text: str,
        schema: dict[str, Any],
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> dict[str, Any] | None:
        kwargs = {
            "include_confidence": include_confidence,
            "include_spans": include_spans,
        }
        return self._invoke("extract_json", text, schema, **kwargs)

    def extract_relations(
        self,
        text: str,
        schema: list[str] | dict[str, str],
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> dict[str, Any] | None:
        kwargs = {
            "include_confidence": include_confidence,
            "include_spans": include_spans,
        }
        return self._invoke("extract_relations", text, schema, **kwargs)
