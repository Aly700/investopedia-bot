"""Explicit provider-architecture helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "PROVIDER_ROLE_NAMES",
    "RESEARCH_PROVIDER_ROLE_NAMES",
    "ProviderBundle",
    "ProviderCapabilities",
    "ProviderRoleStatus",
    "build_provider_bundle",
    "configured_provider_role_assignments",
    "configured_provider_role_name",
    "optional_provider_capabilities_for_role",
    "provider_capabilities_for_role",
    "provider_role_metadata_from_config",
    "supported_provider_names_for_role",
]


_BUNDLE_EXPORTS = {
    "PROVIDER_ROLE_NAMES",
    "RESEARCH_PROVIDER_ROLE_NAMES",
    "ProviderBundle",
    "ProviderRoleStatus",
    "build_provider_bundle",
    "configured_provider_role_assignments",
    "configured_provider_role_name",
    "provider_role_metadata_from_config",
}
_CAPABILITY_EXPORTS = {
    "ProviderCapabilities",
    "optional_provider_capabilities_for_role",
    "provider_capabilities_for_role",
    "supported_provider_names_for_role",
}


def __getattr__(name: str) -> Any:
    if name in _BUNDLE_EXPORTS:
        module = import_module("bot.providers.bundle")
        return getattr(module, name)
    if name in _CAPABILITY_EXPORTS:
        module = import_module("bot.providers.capabilities")
        return getattr(module, name)
    raise AttributeError(f"module 'bot.providers' has no attribute {name!r}")
