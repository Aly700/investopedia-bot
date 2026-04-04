"""Provider bundle construction and role-level provider metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from bot.config import AppConfig
from bot.data.earnings import EarningsCalendarProvider, create_earnings_calendar_provider
from bot.data.providers import (
    DailyBarProvider,
    DataProviderConfigurationError,
    create_historical_bars_provider,
)
from bot.data.reference import ReferenceUniverseProvider, create_reference_data_provider
from bot.providers.capabilities import (
    ProviderCapabilities,
    optional_provider_capabilities_for_role,
    provider_capabilities_for_role,
)


PROVIDER_ROLE_NAMES: tuple[str, ...] = (
    "stream_market_data",
    "historical_bars",
    "reference_data",
    "earnings_calendar",
    "execution_broker",
    "broker_update_stream",
)
RESEARCH_PROVIDER_ROLE_NAMES: tuple[str, ...] = (
    "historical_bars",
    "reference_data",
    "earnings_calendar",
)


@dataclass(frozen=True)
class ProviderRoleStatus:
    """Structured provider-role truth for operator surfaces."""

    role_name: str
    provider_name: str | None
    configured: bool
    available: bool
    degraded: bool
    implemented: bool
    instantiated: bool
    reason: str | None = None
    capabilities: ProviderCapabilities | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_name": self.role_name,
            "provider": self.provider_name,
            "configured": self.configured,
            "available": self.available,
            "degraded": self.degraded,
            "implemented": self.implemented,
            "instantiated": self.instantiated,
            "reason": self.reason,
            "capabilities": (
                self.capabilities.to_dict() if self.capabilities is not None else None
            ),
        }


@dataclass(frozen=True)
class ProviderBundle:
    """Resolved research providers plus role-level provider metadata."""

    role_assignments: dict[str, str | None]
    historical_bars_provider: DailyBarProvider | None
    historical_bars_capabilities: ProviderCapabilities
    reference_data_provider: ReferenceUniverseProvider | None
    reference_data_capabilities: ProviderCapabilities
    earnings_calendar_provider: EarningsCalendarProvider | None
    earnings_calendar_capabilities: ProviderCapabilities
    role_statuses: dict[str, ProviderRoleStatus] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def resolved_role_statuses(
        self,
        *,
        stream_provider: str | None = None,
        execution_broker: str | None = None,
        broker_update_stream_provider: str | None = None,
    ) -> dict[str, ProviderRoleStatus]:
        statuses = dict(self.role_statuses)
        if stream_provider is not None:
            statuses["stream_market_data"] = _non_research_role_status(
                "stream_market_data",
                stream_provider,
            )
        if execution_broker is not None:
            statuses["execution_broker"] = _non_research_role_status(
                "execution_broker",
                execution_broker,
            )
        if broker_update_stream_provider is not None:
            statuses["broker_update_stream"] = _non_research_role_status(
                "broker_update_stream",
                broker_update_stream_provider,
            )
        return statuses

    def provider_metadata(
        self,
        *,
        include_stream: bool = False,
        include_execution: bool = False,
        stream_provider: str | None = None,
        execution_broker: str | None = None,
        broker_update_stream_provider: str | None = None,
    ) -> dict[str, Any]:
        resolved_statuses = self.resolved_role_statuses(
            stream_provider=stream_provider,
            execution_broker=execution_broker,
            broker_update_stream_provider=broker_update_stream_provider,
        )
        payload: dict[str, Any] = {
            "provider": resolved_statuses["historical_bars"].provider_name,
            "historical_provider": resolved_statuses["historical_bars"].provider_name,
            "reference_provider": resolved_statuses["reference_data"].provider_name,
            "earnings_provider": resolved_statuses["earnings_calendar"].provider_name,
            "provider_roles": {
                role_name: resolved_statuses[role_name].to_dict()
                for role_name in PROVIDER_ROLE_NAMES
            },
            "degraded_provider_roles": [
                role_name
                for role_name, status in resolved_statuses.items()
                if status.degraded
            ],
            "unavailable_provider_roles": [
                role_name
                for role_name, status in resolved_statuses.items()
                if status.configured and not status.available
            ],
        }
        if include_stream:
            payload["stream_provider"] = resolved_statuses["stream_market_data"].provider_name
        if include_execution:
            payload["execution_broker"] = resolved_statuses["execution_broker"].provider_name
            payload["broker_update_stream_provider"] = resolved_statuses[
                "broker_update_stream"
            ].provider_name
        return payload


def configured_provider_role_name(
    config: object,
    role_name: str,
    *,
    fallback_to_legacy_provider: bool = False,
) -> str | None:
    """Resolve one provider role from config, preserving legacy fallbacks."""

    data_sources = getattr(config, "data_sources", None)
    if data_sources is None:
        return None

    resolved_roles = getattr(data_sources, "resolved_roles", None)
    if callable(resolved_roles):
        resolved = resolved_roles()
        role_value = getattr(resolved, role_name, None)
        if isinstance(role_value, str) and role_value:
            return role_value

    raw_roles = getattr(data_sources, "roles", None)
    if raw_roles is not None:
        role_value = getattr(raw_roles, role_name, None)
        if isinstance(role_value, str) and role_value:
            return role_value

    if fallback_to_legacy_provider:
        provider_name = getattr(data_sources, "provider", None)
        if isinstance(provider_name, str) and provider_name:
            return provider_name
    return None


def configured_provider_role_assignments(config: object) -> dict[str, str | None]:
    """Return configured provider assignments for all known roles."""

    return {
        "stream_market_data": configured_provider_role_name(config, "stream_market_data"),
        "historical_bars": configured_provider_role_name(
            config,
            "historical_bars",
            fallback_to_legacy_provider=True,
        ),
        "reference_data": configured_provider_role_name(
            config,
            "reference_data",
            fallback_to_legacy_provider=True,
        ),
        "earnings_calendar": configured_provider_role_name(
            config,
            "earnings_calendar",
            fallback_to_legacy_provider=True,
        ),
        "execution_broker": configured_provider_role_name(config, "execution_broker"),
        "broker_update_stream": configured_provider_role_name(config, "broker_update_stream"),
    }


def provider_role_metadata_from_config(
    config: object,
    *,
    include_stream: bool = False,
    include_execution: bool = False,
    stream_provider: str | None = None,
    execution_broker: str | None = None,
    broker_update_stream_provider: str | None = None,
) -> dict[str, Any]:
    """Return role-level provider metadata without constructing provider instances."""

    assignments = configured_provider_role_assignments(config)
    statuses = {
        "stream_market_data": _non_research_role_status(
            "stream_market_data",
            stream_provider if stream_provider is not None else assignments["stream_market_data"],
        ),
        "historical_bars": _historical_role_status(
            assignments["historical_bars"],
            provider_instance=None,
            compatibility_mode=False,
        ),
        "reference_data": _reference_role_status(
            assignments["reference_data"],
            provider_instance=None,
            compatibility_mode=False,
        ),
        "earnings_calendar": _earnings_role_status(
            assignments["earnings_calendar"],
            provider_instance=None,
            compatibility_mode=False,
        ),
        "execution_broker": _non_research_role_status(
            "execution_broker",
            execution_broker if execution_broker is not None else assignments["execution_broker"],
        ),
        "broker_update_stream": _non_research_role_status(
            "broker_update_stream",
            (
                broker_update_stream_provider
                if broker_update_stream_provider is not None
                else assignments["broker_update_stream"]
            ),
        ),
    }
    bundle = ProviderBundle(
        role_assignments=assignments,
        historical_bars_provider=None,
        historical_bars_capabilities=statuses["historical_bars"].capabilities
        or ProviderCapabilities(),
        reference_data_provider=None,
        reference_data_capabilities=statuses["reference_data"].capabilities
        or ProviderCapabilities(),
        earnings_calendar_provider=None,
        earnings_calendar_capabilities=statuses["earnings_calendar"].capabilities
        or ProviderCapabilities(),
        role_statuses=statuses,
    )
    return bundle.provider_metadata(
        include_stream=include_stream,
        include_execution=include_execution,
    )


def build_provider_bundle(
    config: AppConfig,
    *,
    env_file: Path | None,
    historical_bars_factory: Callable[..., DailyBarProvider] = create_historical_bars_provider,
    reference_data_factory: Callable[..., ReferenceUniverseProvider] = create_reference_data_provider,
    earnings_calendar_factory: Callable[..., EarningsCalendarProvider] = create_earnings_calendar_provider,
) -> ProviderBundle:
    """Build the explicit provider bundle for the current runtime."""

    data_sources = getattr(config, "data_sources", None)
    role_aware_config = (
        data_sources is not None
        and callable(getattr(data_sources, "provider_name_for_role", None))
        and callable(getattr(data_sources, "provider_config", None))
    )
    role_assignments = configured_provider_role_assignments(config)
    if not role_aware_config:
        historical_bars_provider = historical_bars_factory(config, env_file=env_file)
        statuses = _build_role_statuses(
            role_assignments,
            historical_bars_provider=historical_bars_provider,
            compatibility_mode=True,
        )
        return ProviderBundle(
            role_assignments=role_assignments,
            historical_bars_provider=historical_bars_provider,
            historical_bars_capabilities=statuses["historical_bars"].capabilities
            or ProviderCapabilities(),
            reference_data_provider=None,
            reference_data_capabilities=ProviderCapabilities(),
            earnings_calendar_provider=None,
            earnings_calendar_capabilities=ProviderCapabilities(),
            role_statuses=statuses,
        )

    historical_provider_name = role_assignments["historical_bars"]
    reference_provider_name = role_assignments["reference_data"]
    earnings_provider_name = role_assignments["earnings_calendar"]
    if historical_provider_name is None:
        raise DataProviderConfigurationError(
            "Historical-bars provider role is not configured."
        )
    if reference_provider_name is None:
        raise DataProviderConfigurationError(
            "Reference-data provider role is not configured."
        )
    if earnings_provider_name is None:
        raise DataProviderConfigurationError(
            "Earnings-calendar provider role is not configured."
        )

    historical_bars_provider = historical_bars_factory(config, env_file=env_file)
    reference_data_provider = reference_data_factory(config, env_file=env_file)
    earnings_calendar_provider = earnings_calendar_factory(config, env_file=env_file)
    statuses = _build_role_statuses(
        role_assignments,
        historical_bars_provider=historical_bars_provider,
        reference_data_provider=reference_data_provider,
        earnings_calendar_provider=earnings_calendar_provider,
    )
    return ProviderBundle(
        role_assignments=role_assignments,
        historical_bars_provider=historical_bars_provider,
        historical_bars_capabilities=statuses["historical_bars"].capabilities
        or ProviderCapabilities(),
        reference_data_provider=reference_data_provider,
        reference_data_capabilities=statuses["reference_data"].capabilities
        or ProviderCapabilities(),
        earnings_calendar_provider=earnings_calendar_provider,
        earnings_calendar_capabilities=statuses["earnings_calendar"].capabilities
        or ProviderCapabilities(),
        role_statuses=statuses,
    )


def _build_role_statuses(
    role_assignments: dict[str, str | None],
    *,
    historical_bars_provider: DailyBarProvider | None = None,
    reference_data_provider: ReferenceUniverseProvider | None = None,
    earnings_calendar_provider: EarningsCalendarProvider | None = None,
    compatibility_mode: bool = False,
) -> dict[str, ProviderRoleStatus]:
    return {
        "stream_market_data": _non_research_role_status(
            "stream_market_data",
            role_assignments["stream_market_data"],
        ),
        "historical_bars": _historical_role_status(
            role_assignments["historical_bars"],
            provider_instance=historical_bars_provider,
            compatibility_mode=compatibility_mode,
        ),
        "reference_data": _reference_role_status(
            role_assignments["reference_data"],
            provider_instance=reference_data_provider,
            compatibility_mode=compatibility_mode,
        ),
        "earnings_calendar": _earnings_role_status(
            role_assignments["earnings_calendar"],
            provider_instance=earnings_calendar_provider,
            compatibility_mode=compatibility_mode,
        ),
        "execution_broker": _non_research_role_status(
            "execution_broker",
            role_assignments["execution_broker"],
        ),
        "broker_update_stream": _non_research_role_status(
            "broker_update_stream",
            role_assignments["broker_update_stream"],
        ),
    }


def _historical_role_status(
    provider_name: str | None,
    *,
    provider_instance: DailyBarProvider | None,
    compatibility_mode: bool,
) -> ProviderRoleStatus:
    capabilities = optional_provider_capabilities_for_role("historical_bars", provider_name)
    if provider_name is None:
        return _empty_role_status("historical_bars")
    if capabilities is None:
        return _unsupported_role_status("historical_bars", provider_name)
    if provider_instance is None:
        if compatibility_mode:
            return replace(
                _base_role_status(
                    "historical_bars",
                    provider_name,
                    capabilities=capabilities,
                    available=True,
                    instantiated=False,
                ),
            )
        return _base_role_status(
            "historical_bars",
            provider_name,
            capabilities=capabilities,
            available=False,
            instantiated=False,
            degraded=True,
            reason="historical-bars provider is configured but not instantiated in this context.",
        )
    if not capabilities.supports_vix_symbols:
        return _base_role_status(
            "historical_bars",
            provider_name,
            capabilities=capabilities,
            available=True,
            instantiated=True,
            degraded=True,
            reason=(
                "historical-bars provider does not support VIX/index-style symbols; "
                "volatility context will be unavailable without a separate provider."
            ),
        )
    return _base_role_status(
        "historical_bars",
        provider_name,
        capabilities=capabilities,
        available=True,
        instantiated=True,
    )


def _reference_role_status(
    provider_name: str | None,
    *,
    provider_instance: ReferenceUniverseProvider | None,
    compatibility_mode: bool,
) -> ProviderRoleStatus:
    return _research_role_status(
        "reference_data",
        provider_name,
        provider_instance=provider_instance,
        compatibility_mode=compatibility_mode,
    )


def _earnings_role_status(
    provider_name: str | None,
    *,
    provider_instance: EarningsCalendarProvider | None,
    compatibility_mode: bool,
) -> ProviderRoleStatus:
    return _research_role_status(
        "earnings_calendar",
        provider_name,
        provider_instance=provider_instance,
        compatibility_mode=compatibility_mode,
    )


def _research_role_status(
    role_name: str,
    provider_name: str | None,
    *,
    provider_instance: object | None,
    compatibility_mode: bool,
) -> ProviderRoleStatus:
    capabilities = optional_provider_capabilities_for_role(role_name, provider_name)
    if provider_name is None:
        return _empty_role_status(role_name)
    if capabilities is None:
        return _unsupported_role_status(role_name, provider_name)
    if provider_instance is not None:
        return _base_role_status(
            role_name,
            provider_name,
            capabilities=capabilities,
            available=True,
            instantiated=True,
        )
    if compatibility_mode:
        return _base_role_status(
            role_name,
            provider_name,
            capabilities=capabilities,
            available=False,
            instantiated=False,
            degraded=True,
            reason=(
                f"{role_name.replace('_', '-')} provider is unavailable in compatibility mode."
            ),
        )
    return _base_role_status(
        role_name,
        provider_name,
        capabilities=capabilities,
        available=False,
        instantiated=False,
        degraded=True,
        reason=f"{role_name.replace('_', '-')} provider is configured but not instantiated.",
    )


def _non_research_role_status(
    role_name: str,
    provider_name: str | None,
) -> ProviderRoleStatus:
    if provider_name is None:
        return _empty_role_status(role_name)
    capabilities = optional_provider_capabilities_for_role(role_name, provider_name)
    if capabilities is None:
        return _unsupported_role_status(role_name, provider_name)
    if role_name == "broker_update_stream":
        return _base_role_status(
            role_name,
            provider_name,
            capabilities=capabilities,
            available=False,
            instantiated=False,
            degraded=True,
            implemented=False,
            reason=(
                "broker-update stream provider is configured but no broker-update stream "
                "adapter is wired into this runtime yet."
            ),
        )
    return _base_role_status(
        role_name,
        provider_name,
        capabilities=capabilities,
        available=True,
        instantiated=False,
    )


def _base_role_status(
    role_name: str,
    provider_name: str,
    *,
    capabilities: ProviderCapabilities,
    available: bool,
    instantiated: bool,
    degraded: bool = False,
    implemented: bool = True,
    reason: str | None = None,
) -> ProviderRoleStatus:
    return ProviderRoleStatus(
        role_name=role_name,
        provider_name=provider_name,
        configured=True,
        available=available,
        degraded=degraded,
        implemented=implemented,
        instantiated=instantiated,
        reason=reason,
        capabilities=capabilities,
    )


def _empty_role_status(role_name: str) -> ProviderRoleStatus:
    return ProviderRoleStatus(
        role_name=role_name,
        provider_name=None,
        configured=False,
        available=False,
        degraded=False,
        implemented=False,
        instantiated=False,
        reason=None,
        capabilities=None,
    )


def _unsupported_role_status(
    role_name: str,
    provider_name: str,
) -> ProviderRoleStatus:
    try:
        provider_capabilities_for_role(role_name, provider_name)
    except ValueError as exc:
        reason = str(exc)
    else:  # pragma: no cover - defensive only
        reason = None
    return ProviderRoleStatus(
        role_name=role_name,
        provider_name=provider_name,
        configured=True,
        available=False,
        degraded=True,
        implemented=False,
        instantiated=False,
        reason=reason,
        capabilities=None,
    )

