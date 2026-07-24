from __future__ import annotations

import hashlib
import inspect
import os
import re
import threading
import time
from copy import deepcopy
from importlib import metadata
from typing import Any

from usr.plugins.gliner2.helpers.config import (
    DEFAULT_MODEL,
    GLINER2_LOCAL_PACKAGE_SPEC,
    GLINER2_PACKAGE_SPEC,
    MIN_GLINER2_VERSION,
    normalize_config,
)


_CLIENTS: dict[tuple[Any, ...], "GLiNER2Client"] = {}
_CLIENTS_LOCK = threading.Lock()
_MAX_CACHED_CLIENTS = 4


def _import_gliner2(mode: str = "local") -> tuple[type[Any] | None, str]:
    try:
        if mode == "api":
            from gliner2 import GLiNER2API  # type: ignore

            return GLiNER2API, ""
        from gliner2 import GLiNER2  # type: ignore

        return GLiNER2, ""
    except Exception as exc:
        return None, str(exc)


def _package_version() -> str:
    try:
        return metadata.version("gliner2")
    except metadata.PackageNotFoundError:
        return ""
    except Exception:
        return ""


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", value or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _supported_package_version(value: str) -> bool:
    parsed = _version_tuple(value)
    minimum = _version_tuple(MIN_GLINER2_VERSION)
    return bool(parsed and parsed >= minimum and parsed[0] < 2)


def _resolve_api_key(config: dict[str, Any]) -> str:
    env_name = str(
        config.get("gliner2_api_key_env", "PIONEER_API_KEY") or "PIONEER_API_KEY"
    )
    return os.environ.get(env_name, "")


def _secret_fingerprint(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    cfg = normalize_config(config)
    return (
        cfg["gliner2_mode"],
        cfg["gliner2_model"],
        cfg["gliner2_api_key_env"],
        _secret_fingerprint(_resolve_api_key(cfg)),
        cfg["gliner2_api_base_url"],
        cfg["gliner2_api_timeout_seconds"],
        cfg["gliner2_api_max_retries"],
        cfg["gliner2_device"],
        cfg["gliner2_quantize"],
        cfg["gliner2_compile"],
    )


def _get_torch_runtime_status() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        return {
            "installed": True,
            "version": str(getattr(torch, "__version__", "") or ""),
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cuda_device_name": (
                str(torch.cuda.get_device_name(0))
                if cuda_available and device_count > 0
                else ""
            ),
            "error": "",
        }
    except Exception as exc:
        return {
            "installed": False,
            "version": "",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": str(exc),
        }


def _call_with_supported_kwargs(
    callable_object: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        signature = inspect.signature(callable_object)
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            accepted_kwargs = kwargs
        else:
            accepted_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
    except (TypeError, ValueError):
        accepted_kwargs = kwargs
    return callable_object(*args, **accepted_kwargs)


def clear_clients() -> None:
    with _CLIENTS_LOCK:
        _CLIENTS.clear()


def get_client(config: dict[str, Any] | None) -> "GLiNER2Client":
    cfg = normalize_config(deepcopy(config or {}))
    key = _config_key(cfg)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            if len(_CLIENTS) >= _MAX_CACHED_CLIENTS:
                oldest_key = next(iter(_CLIENTS))
                _CLIENTS.pop(oldest_key, None)
            client = GLiNER2Client(cfg)
            _CLIENTS[key] = client
        return client


def get_runtime_status(
    config: dict[str, Any] | None,
    load_model: bool = False,
) -> dict[str, Any]:
    cfg = normalize_config(config)
    mode = cfg["gliner2_mode"]
    package_version = _package_version()
    gliner_class, import_error = _import_gliner2(mode)
    package_installed = bool(package_version)
    runtime_importable = gliner_class is not None
    package_compatible = _supported_package_version(package_version)
    torch_status = (
        _get_torch_runtime_status()
        if mode == "local"
        else {
            "installed": False,
            "version": "",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": "",
        }
    )
    api_key_configured = bool(_resolve_api_key(cfg))

    key = _config_key(cfg)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
    setup_blockers: list[str] = []
    warnings: list[str] = []
    if not cfg["gliner2_enabled"]:
        setup_blockers.append("The plugin is disabled in this settings scope.")
    if not package_installed:
        setup_blockers.append(
            "The GLiNER2 package is not installed in Agent Zero's framework runtime."
        )
    elif not package_compatible:
        setup_blockers.append(
            f"GLiNER2 {package_version or 'unknown'} is outside the supported "
            f"range {GLINER2_PACKAGE_SPEC.removeprefix('gliner2')}."
        )
    if package_installed and not runtime_importable:
        setup_blockers.append(
            (
                "The local GLiNER2 runtime could not import. Install or repair "
                "the local extra."
                if mode == "local"
                else "The GLiNER2 API client could not import."
            )
        )

    if mode == "api":
        if not api_key_configured:
            setup_blockers.append(
                f"No API key is available in {cfg['gliner2_api_key_env']}."
            )
    else:
        if not torch_status["installed"]:
            setup_blockers.append("PyTorch is not installed for local inference.")
        requested_device = cfg["gliner2_device"]
        if requested_device == "cuda" and not torch_status["cuda_available"]:
            setup_blockers.append("CUDA was requested but is not visible to PyTorch.")
        if (
            (cfg["gliner2_quantize"] or cfg["gliner2_compile"])
            and not torch_status["cuda_available"]
        ):
            warnings.append(
                "Quantization and torch.compile are ignored until CUDA is available."
            )
        if requested_device == "auto" and not torch_status["cuda_available"]:
            warnings.append("Auto device selection will use CPU.")

    configured = not setup_blockers
    model_load_started = False
    if load_model and configured:
        client = get_client(cfg)
        model_load_started = client.start_loading()

    client_loaded = bool(client and client.is_loaded())
    client_loading = bool(client and client.is_loading())
    client_loading_seconds = client.loading_seconds() if client else None
    client_state = "ready" if client_loaded else "not_loaded"
    if client_loading:
        client_state = "loading"
    elif client and client.last_error:
        client_state = "failed"

    blockers = list(setup_blockers)
    if client and client.last_warning:
        warnings.append(client.last_warning)
    if client and client.last_error:
        blockers.append(client.last_error)
    elif import_error and package_installed:
        blockers.append(import_error)

    if client_loaded:
        readiness = "ready"
    elif client_loading:
        readiness = "loading"
    elif client and client.last_error and configured:
        readiness = "failed"
    elif configured:
        readiness = "configured"
    else:
        readiness = "blocked"

    resolved_device = "api"
    if mode == "local":
        resolved_device = (
            client.resolved_device
            if client and client.resolved_device
            else (
                "cuda"
                if cfg["gliner2_device"] == "auto"
                and torch_status["cuda_available"]
                else (
                    "cpu"
                    if cfg["gliner2_device"] == "auto"
                    else cfg["gliner2_device"]
                )
            )
        )

    return {
        "ok": True,
        "plugin": "gliner2",
        "readiness": readiness,
        "configured": configured,
        "enabled": cfg["gliner2_enabled"],
        "mode": mode,
        "model_name": cfg["gliner2_model"] if mode == "local" else "Pioneer API",
        "requested_device": cfg["gliner2_device"],
        "resolved_device": resolved_device,
        "package_installed": package_installed,
        "package_version": package_version or None,
        "package_compatible": package_compatible,
        "runtime_importable": runtime_importable,
        "package_spec": (
            GLINER2_LOCAL_PACKAGE_SPEC if mode == "local" else GLINER2_PACKAGE_SPEC
        ),
        "client_loaded": client_loaded,
        "client_loading": client_loading,
        "client_loading_seconds": client_loading_seconds,
        "client_state": client_state,
        "model_loaded": client_loaded,
        "model_loading": client_loading,
        "model_loading_seconds": client_loading_seconds,
        "model_state": client_state,
        "model_load_started": model_load_started,
        "status_loaded_client": bool(load_model),
        "operation_timeout_seconds": cfg["gliner2_operation_timeout_seconds"],
        "api_key_env": cfg["gliner2_api_key_env"],
        "api_key_configured": api_key_configured,
        "api_base_url_custom": bool(cfg["gliner2_api_base_url"]),
        "in_docker": os.path.exists("/.dockerenv"),
        "torch_installed": torch_status["installed"],
        "torch_version": torch_status["version"] or None,
        "torch_cuda_available": torch_status["cuda_available"],
        "torch_cuda_device_count": torch_status["cuda_device_count"],
        "torch_cuda_device_name": torch_status["cuda_device_name"] or None,
        "torch_error": torch_status["error"] or None,
        "utility_replacement_enabled": cfg["gliner2_utility_replacement_enabled"],
        "fallback_to_utility_model": cfg["gliner2_fallback_to_utility_model"],
        "last_operation": client.last_operation if client else None,
        "last_duration_ms": client.last_duration_ms if client else None,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "error": (client.last_error if client else "") or import_error or None,
    }


def get_loaded_client_or_status(
    config: dict[str, Any] | None,
) -> tuple["GLiNER2Client | None", dict[str, Any]]:
    """Return a ready client without making the caller wait for model startup.

    The first explicit extraction starts initialization in a daemon thread and
    returns status immediately. Callers can retry once ``client_state`` becomes
    ``ready``.
    """
    cfg = normalize_config(config)
    client = get_client(cfg)
    status = get_runtime_status(cfg, load_model=False)
    if client.is_available(load_model=False):
        return client, status
    if status["configured"]:
        started = client.start_loading()
        status = get_runtime_status(cfg, load_model=False)
        status["model_load_started"] = started
    return None, status


class GLiNER2Client:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = normalize_config(deepcopy(config or {}))
        self._load_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._load_thread: threading.Thread | None = None
        self._loading_started_at: float | None = None
        self.model: Any | None = None
        self.last_error = ""
        self.last_warning = ""
        self.last_operation: str | None = None
        self.last_duration_ms: int | None = None
        self.resolved_device = ""

    def is_loaded(self) -> bool:
        return self.model is not None

    def is_loading(self) -> bool:
        with self._state_lock:
            thread = self._load_thread
            return bool(thread and thread.is_alive()) or (
                self._loading_started_at is not None
            )

    def loading_seconds(self) -> float | None:
        with self._state_lock:
            started = self._loading_started_at
        if started is None:
            return None
        return round(max(0.0, time.monotonic() - started), 1)

    def is_available(self, load_model: bool = True) -> bool:
        if load_model:
            self._ensure_model()
        return self.model is not None

    def start_loading(self) -> bool:
        """Start model/client initialization once without blocking the caller."""
        with self._state_lock:
            if self.model is not None:
                return False
            thread = self._load_thread
            if thread and thread.is_alive():
                return False
            self.last_error = ""
            self._loading_started_at = time.monotonic()
            thread = threading.Thread(
                target=self._load_in_background,
                name="gliner2-client-load",
                daemon=True,
            )
            self._load_thread = thread
            thread.start()
        return True

    def _load_in_background(self) -> None:
        try:
            self._ensure_model()
        finally:
            with self._state_lock:
                self._load_thread = None
                self._loading_started_at = None

    def _resolve_local_device(self) -> str:
        requested = self.config["gliner2_device"]
        torch_status = _get_torch_runtime_status()
        if requested == "cuda":
            if not torch_status["cuda_available"]:
                raise RuntimeError(
                    "CUDA was requested, but PyTorch cannot see a CUDA device. "
                    "Use auto/CPU or recreate the container with GPU access."
                )
            return "cuda"
        if requested == "cpu":
            return "cpu"
        return "cuda" if torch_status["cuda_available"] else "cpu"

    def _ensure_model(self) -> Any | None:
        if self.model is not None:
            return self.model

        with self._load_lock:
            if self.model is not None:
                return self.model

            with self._state_lock:
                if self._loading_started_at is None:
                    self._loading_started_at = time.monotonic()

            mode = self.config["gliner2_mode"]
            gliner_class, import_error = _import_gliner2(mode)
            if gliner_class is None:
                self.last_error = import_error or "GLiNER2 is not installed."
                with self._state_lock:
                    self._loading_started_at = None
                return None

            try:
                if mode == "api":
                    api_key = _resolve_api_key(self.config)
                    if not api_key:
                        raise RuntimeError(
                            f"No API key found in {self.config['gliner2_api_key_env']}."
                        )
                    api_kwargs = {
                        "api_key": api_key,
                        "api_base_url": self.config["gliner2_api_base_url"] or None,
                        "timeout": self.config["gliner2_api_timeout_seconds"],
                        "max_retries": self.config["gliner2_api_max_retries"],
                    }
                    from_api = getattr(gliner_class, "from_api", None)
                    self.model = _call_with_supported_kwargs(
                        from_api or gliner_class,
                        **api_kwargs,
                    )
                    self.resolved_device = "api"
                else:
                    self.resolved_device = self._resolve_local_device()
                    kwargs: dict[str, Any] = {
                        "map_location": self.resolved_device,
                    }
                    gpu_options_requested = bool(
                        self.config["gliner2_quantize"]
                        or self.config["gliner2_compile"]
                    )
                    if self.resolved_device == "cuda":
                        kwargs["quantize"] = self.config["gliner2_quantize"]
                        kwargs["compile"] = self.config["gliner2_compile"]
                    elif gpu_options_requested:
                        self.last_warning = (
                            "Quantization and torch.compile were skipped because "
                            "the resolved device is CPU."
                        )

                    self.model = _call_with_supported_kwargs(
                        gliner_class.from_pretrained,
                        self.config["gliner2_model"] or DEFAULT_MODEL,
                        **kwargs,
                    )
                self.last_error = ""
            except Exception as exc:
                self.model = None
                self.last_error = str(exc)
            finally:
                with self._state_lock:
                    if threading.current_thread() is not self._load_thread:
                        self._loading_started_at = None

        return self.model

    def _invoke(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any | None:
        model = self._ensure_model()
        if model is None:
            return None

        method = getattr(model, method_name, None)
        if method is None:
            self.last_error = f"GLiNER2 method not available: {method_name}"
            return None

        started = time.perf_counter()
        try:
            with self._inference_lock:
                result = _call_with_supported_kwargs(method, *args, **kwargs)
            self.last_error = ""
            return result
        except Exception as exc:
            self.last_error = str(exc)
            return None
        finally:
            self.last_operation = method_name
            self.last_duration_ms = max(
                0, round((time.perf_counter() - started) * 1000)
            )

    def extract_entities(
        self,
        text: str,
        schema: list[str] | dict[str, Any],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any] | None:
        return self._invoke(
            "extract_entities",
            text,
            schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )

    def batch_extract_entities(
        self,
        texts: list[str],
        schema: list[str] | dict[str, Any],
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, Any]] | None:
        result = self._invoke(
            "batch_extract_entities",
            texts,
            schema,
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )
        return result if isinstance(result, list) else None

    def classify_text(
        self,
        text: str,
        schema: dict[str, Any],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any] | None:
        return self._invoke(
            "classify_text",
            text,
            schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )

    def extract_json(
        self,
        text: str,
        schema: dict[str, Any],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any] | None:
        return self._invoke(
            "extract_json",
            text,
            schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )

    def extract_relations(
        self,
        text: str,
        schema: list[str] | dict[str, Any],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any] | None:
        return self._invoke(
            "extract_relations",
            text,
            schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )
