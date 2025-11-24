"""Backend Realtime 服务包"""
from importlib import import_module
from typing import Any

__all__ = ["llm", "tts_realtime"]


def __getattr__(name: str) -> Any:  # pragma: no cover
	if name in __all__:
		module = import_module(f"{__name__}.{name}")
		globals()[name] = module
		return module
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
