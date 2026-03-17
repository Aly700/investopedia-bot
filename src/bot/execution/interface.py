"""Execution-layer interfaces shared by manual and future automated executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Sequence

from bot.risk.portfolio_rules import RiskAssessedCandidate


OrderAction = Literal["BUY", "SELL"]
IntendedOrderType = Literal["MARKET", "LIMIT", "STOP_LIMIT"]


@dataclass(frozen=True)
class ExecutionOrder:
    """A human- or machine-consumable order intent derived from an approved candidate."""

    date: date
    symbol: str
    action: OrderAction
    quantity: int
    intended_order_type: IntendedOrderType
    entry_price_hint: float | None = None
    stop_level: float | None = None
    rationale: str = ""
    time_in_force: str = "DAY"
    strategy_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")
        if self.action not in ("BUY", "SELL"):
            raise ValueError(f"Unsupported action: {self.action}")
        if self.intended_order_type not in ("MARKET", "LIMIT", "STOP_LIMIT"):
            raise ValueError(f"Unsupported intended_order_type: {self.intended_order_type}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the order."""

        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "intended_order_type": self.intended_order_type,
            "entry_price_hint": self.entry_price_hint,
            "stop_level": self.stop_level,
            "rationale": self.rationale,
            "time_in_force": self.time_in_force,
            "strategy_name": self.strategy_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionBatch:
    """A batch of order intents produced by one executor implementation."""

    executor_name: str
    as_of_date: date
    generated_at_utc: str
    orders: tuple[ExecutionOrder, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the batch."""

        return {
            "executor_name": self.executor_name,
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at_utc": self.generated_at_utc,
            "order_count": len(self.orders),
            "orders": [order.to_dict() for order in self.orders],
        }


class CandidateExecutor(ABC):
    """Interface for turning approved risk candidates into execution intents."""

    executor_name: str

    @abstractmethod
    def build_execution_batch(
        self,
        candidates: Sequence[RiskAssessedCandidate],
        *,
        as_of_date: date,
    ) -> ExecutionBatch:
        """Convert approved candidates into order intents for an execution workflow."""
