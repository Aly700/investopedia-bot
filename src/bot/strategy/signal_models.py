"""Strategy signal data structures shared across research workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal


SignalSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class StrategySignal:
    """A normalized strategy signal ready for later risk and backtest layers."""

    strategy_name: str
    symbol: str
    date: date
    side: SignalSide
    entry_reason: str
    entry_price_hint: float
    stop_hint: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the signal."""

        payload = asdict(self)
        payload["date"] = self.date.isoformat()
        return payload
