"""
Foundation layer for specialist system.

This package provides core components for all specialist tools including:
- Configuration management
- Prompt loading and caching
- Image processing
- Bedrock client management
- Message chain building
- Response processing

Submodules are resolved lazily (PEP 562) rather than imported here. Importing
one name must not drag in the dependencies of every other module: the
orchestrator runtime needs only ``foundation.job_state``, which is stdlib-only,
while ``image_processor`` and ``specialist_foundation`` import Pillow at module
scope. Eager imports meant ``from foundation import job_state`` raised
ModuleNotFoundError in the agent container, where Pillow is deliberately absent
(deployment/runtime/requirements.txt) -- and because the caller there catches
ImportError to degrade gracefully, job tracking silently disabled itself for
every specialist call.

Adding a module to _EXPORTS therefore costs nothing at import time. Keep it that
way: do not reintroduce top-level ``from .x import y`` statements.
"""

from importlib import import_module

# Exported name -> submodule that defines it.
_EXPORTS = {
    "ConfigurationManager": "configuration_manager",
    "ConfigurationError": "configuration_manager",
    "PromptLoader": "prompt_loader",
    "PromptLoadError": "prompt_loader",
    "ImageProcessor": "image_processor",
    "ImageProcessingError": "image_processor",
    "BedrockClient": "bedrock_client",
    "BedrockError": "bedrock_client",
    "MessageChainBuilder": "message_chain_builder",
    "MessageChainError": "message_chain_builder",
    "ResponseProcessor": "response_processor",
    "ResponseProcessingError": "response_processor",
    "SpecialistFoundation": "specialist_foundation",
    "AnalysisError": "specialist_foundation",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import the owning submodule on first access to an exported name.

    Keeps ``from foundation import BedrockClient`` working for the specialist
    Lambdas while leaving the import graph untouched for callers that only want
    a different submodule.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    # Cache on the module so __getattr__ runs at most once per name.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
