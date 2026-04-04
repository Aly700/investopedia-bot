"""Provider role capabilities and supported role/provider combinations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    """Small, explicit capability surface for one role/provider pair."""

    supports_daily_bars: bool = False
    supports_intraday_bars: bool = False
    supports_benchmark_symbols: bool = False
    supports_vix_symbols: bool = False
    supports_reference_universe: bool = False
    supports_ticker_details: bool = False
    supports_earnings_calendar: bool = False
    supports_sector_context_inputs: bool = False
    supports_batch_symbol_fetch: bool = False
    supports_realtime_bars: bool = False
    supports_realtime_quotes: bool = False
    supports_realtime_trades: bool = False
    supports_trade_updates: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


_ROLE_PROVIDER_CAPABILITIES: dict[str, dict[str, ProviderCapabilities]] = {
    "stream_market_data": {
        "alpaca": ProviderCapabilities(
            supports_realtime_bars=True,
            supports_realtime_quotes=True,
            supports_realtime_trades=True,
        ),
        "polygon": ProviderCapabilities(
            supports_realtime_bars=True,
            supports_realtime_quotes=True,
            supports_realtime_trades=True,
        ),
    },
    "historical_bars": {
        "alphavantage": ProviderCapabilities(
            supports_daily_bars=True,
            supports_benchmark_symbols=True,
            supports_sector_context_inputs=True,
        ),
        "tiingo": ProviderCapabilities(
            supports_daily_bars=True,
            supports_benchmark_symbols=True,
            supports_sector_context_inputs=True,
        ),
        "polygon": ProviderCapabilities(
            supports_daily_bars=True,
            supports_intraday_bars=True,
            supports_benchmark_symbols=True,
            supports_vix_symbols=True,
            supports_sector_context_inputs=True,
        ),
        "alpaca": ProviderCapabilities(
            supports_daily_bars=True,
            supports_intraday_bars=True,
            supports_benchmark_symbols=True,
            supports_sector_context_inputs=True,
        ),
    },
    "reference_data": {
        "polygon": ProviderCapabilities(
            supports_reference_universe=True,
            supports_ticker_details=True,
            supports_sector_context_inputs=True,
            supports_batch_symbol_fetch=True,
        ),
    },
    "earnings_calendar": {
        "polygon": ProviderCapabilities(
            supports_earnings_calendar=True,
            supports_batch_symbol_fetch=True,
        ),
    },
    "execution_broker": {
        "alpaca": ProviderCapabilities(),
    },
    "broker_update_stream": {
        "alpaca": ProviderCapabilities(
            supports_trade_updates=True,
        ),
    },
}


def supported_provider_names_for_role(role_name: str) -> tuple[str, ...]:
    """Return supported provider names for one provider role."""

    normalized_role_name = role_name.strip().lower()
    try:
        role_capabilities = _ROLE_PROVIDER_CAPABILITIES[normalized_role_name]
    except KeyError as exc:
        valid_role_names = ", ".join(sorted(_ROLE_PROVIDER_CAPABILITIES))
        raise ValueError(
            f"Unsupported provider role '{role_name}'. Expected one of: {valid_role_names}."
        ) from exc
    return tuple(sorted(role_capabilities))


def provider_capabilities_for_role(
    role_name: str,
    provider_name: str,
) -> ProviderCapabilities:
    """Return capabilities for one role/provider pair."""

    normalized_role_name = role_name.strip().lower()
    normalized_provider_name = provider_name.strip().lower()
    supported_names = supported_provider_names_for_role(normalized_role_name)
    role_capabilities = _ROLE_PROVIDER_CAPABILITIES[normalized_role_name]
    try:
        return role_capabilities[normalized_provider_name]
    except KeyError as exc:
        raise ValueError(
            "Unsupported provider assignment for role "
            f"'{normalized_role_name}': requested '{normalized_provider_name}', "
            f"supported providers are {list(supported_names)}."
        ) from exc


def optional_provider_capabilities_for_role(
    role_name: str,
    provider_name: str | None,
) -> ProviderCapabilities | None:
    """Return capabilities when available, otherwise ``None``."""

    if provider_name is None:
        return None
    try:
        return provider_capabilities_for_role(role_name, provider_name)
    except ValueError:
        return None

