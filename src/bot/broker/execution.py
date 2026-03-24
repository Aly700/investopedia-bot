"""Broker execution adapters and normalized broker order models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from bot.config import _read_env_file
from bot.data.pending_orders import PendingOrderRecord


DEFAULT_BROKER_TIMEOUT_SECONDS = 30
DEFAULT_ALPACA_BROKER_BASE_URL = "https://paper-api.alpaca.markets"
BROKER_ORDER_TYPES = ("MARKET", "LIMIT", "STOP_LIMIT")
BROKER_ORDER_SIDES = ("BUY", "SELL")
BROKER_ORDER_STATUSES = (
    "accepted",
    "pending_cancel",
    "pending_replace",
    "partially_filled",
    "filled",
    "cancelled",
    "expired",
    "replaced",
    "rejected",
    "done_for_day",
    "stopped",
    "suspended",
    "unknown",
)


class BrokerExecutionError(RuntimeError):
    """Base exception for broker execution failures."""


class BrokerConfigurationError(BrokerExecutionError):
    """Raised when broker credentials or configuration are missing."""


class BrokerRequestError(BrokerExecutionError):
    """Raised when a broker request fails or returns malformed data."""


@dataclass(frozen=True)
class BrokerOrderRequest:
    """Normalized request used to place a broker order."""

    symbol: str
    side: str
    order_type: str
    quantity: int
    time_in_force: str = "DAY"
    client_order_id: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.side not in BROKER_ORDER_SIDES:
            raise ValueError(
                f"side must be one of {BROKER_ORDER_SIDES}, got '{self.side}'."
            )
        if self.order_type not in BROKER_ORDER_TYPES:
            raise ValueError(
                f"order_type must be one of {BROKER_ORDER_TYPES}, got '{self.order_type}'."
            )
        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")
        if not self.time_in_force.strip():
            raise ValueError("time_in_force cannot be empty.")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive when provided.")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided.")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "time_in_force", self.time_in_force.strip().upper())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "time_in_force": self.time_in_force,
        }
        if self.client_order_id is not None:
            payload["client_order_id"] = self.client_order_id
        if self.limit_price is not None:
            payload["limit_price"] = self.limit_price
        if self.stop_price is not None:
            payload["stop_price"] = self.stop_price
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class BrokerReplaceOrderRequest:
    """Normalized request used to amend an existing broker order."""

    quantity: int | None = None
    time_in_force: str | None = None
    client_order_id: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity is None and self.limit_price is None and self.stop_price is None:
            raise ValueError(
                "At least one of quantity, limit_price, or stop_price must be provided."
            )
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("quantity must be greater than zero when provided.")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive when provided.")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided.")
        if self.time_in_force is not None and not self.time_in_force.strip():
            raise ValueError("time_in_force cannot be empty when provided.")
        if self.time_in_force is not None:
            object.__setattr__(
                self,
                "time_in_force",
                self.time_in_force.strip().upper(),
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.quantity is not None:
            payload["quantity"] = self.quantity
        if self.time_in_force is not None:
            payload["time_in_force"] = self.time_in_force
        if self.client_order_id is not None:
            payload["client_order_id"] = self.client_order_id
        if self.limit_price is not None:
            payload["limit_price"] = self.limit_price
        if self.stop_price is not None:
            payload["stop_price"] = self.stop_price
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class BrokerOrderUpdate:
    """Normalized broker truth for one order."""

    broker_name: str
    broker_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    filled_quantity: int
    status: str
    client_order_id: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    average_fill_price: float | None = None
    submitted_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.broker_name.strip():
            raise ValueError("broker_name cannot be empty.")
        if not self.broker_order_id.strip():
            raise ValueError("broker_order_id cannot be empty.")
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.side not in BROKER_ORDER_SIDES:
            raise ValueError(
                f"side must be one of {BROKER_ORDER_SIDES}, got '{self.side}'."
            )
        if self.order_type not in BROKER_ORDER_TYPES:
            raise ValueError(
                f"order_type must be one of {BROKER_ORDER_TYPES}, got '{self.order_type}'."
            )
        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative.")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed quantity.")
        if self.status not in BROKER_ORDER_STATUSES:
            raise ValueError(
                f"status must be one of {BROKER_ORDER_STATUSES}, got '{self.status}'."
            )
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive when provided.")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided.")
        if self.average_fill_price is not None and self.average_fill_price <= 0:
            raise ValueError("average_fill_price must be positive when provided.")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "broker_name": self.broker_name,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "status": self.status,
        }
        if self.client_order_id is not None:
            payload["client_order_id"] = self.client_order_id
        if self.limit_price is not None:
            payload["limit_price"] = self.limit_price
        if self.stop_price is not None:
            payload["stop_price"] = self.stop_price
        if self.average_fill_price is not None:
            payload["average_fill_price"] = self.average_fill_price
        if self.submitted_at is not None:
            payload["submitted_at"] = self.submitted_at
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @property
    def active(self) -> bool:
        return self.status in {
            "accepted",
            "pending_cancel",
            "pending_replace",
            "partially_filled",
            "done_for_day",
            "stopped",
            "suspended",
        }

    @property
    def final(self) -> bool:
        return self.status in {
            "filled",
            "cancelled",
            "expired",
            "replaced",
            "rejected",
        }


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    """Normalized broker position snapshot."""

    symbol: str
    quantity: int
    average_entry_price: float
    broker_position_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")
        if self.average_entry_price <= 0:
            raise ValueError("average_entry_price must be greater than zero.")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
        }
        if self.broker_position_id is not None:
            payload["broker_position_id"] = self.broker_position_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class BrokerExecutionAdapter(ABC):
    """Interface for normalized broker execution operations."""

    broker_name: str

    @abstractmethod
    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderUpdate:
        """Place one broker order."""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> BrokerOrderUpdate | None:
        """Cancel one existing broker order."""

    @abstractmethod
    def replace_order(
        self,
        broker_order_id: str,
        request: BrokerReplaceOrderRequest,
    ) -> BrokerOrderUpdate:
        """Replace one broker order and return the replacement snapshot."""

    @abstractmethod
    def list_open_orders(self) -> tuple[BrokerOrderUpdate, ...]:
        """Return all currently open broker orders."""

    @abstractmethod
    def get_order(
        self,
        broker_order_id: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> BrokerOrderUpdate:
        """Return one broker order by broker or client identity."""

    @abstractmethod
    def list_positions(self) -> tuple[BrokerPositionSnapshot, ...]:
        """Return current broker positions."""


def broker_order_request_from_pending_order(
    record: PendingOrderRecord,
    *,
    client_order_id: str | None = None,
) -> BrokerOrderRequest:
    """Normalize one pending order into a broker placement request."""

    limit_price = record.requested_price if record.order_type in {"LIMIT", "STOP_LIMIT"} else None
    stop_price = record.stop_price if record.order_type == "STOP_LIMIT" else None
    return BrokerOrderRequest(
        symbol=record.symbol,
        side=record.side,
        order_type=record.order_type,
        quantity=record.remaining_quantity,
        time_in_force="DAY",
        client_order_id=client_order_id or record.client_order_id or record.order_id,
        limit_price=limit_price,
        stop_price=stop_price,
        metadata={
            "local_order_id": record.order_id,
            "trade_id": record.trade_id,
            "linked_decision_id": record.linked_decision_id,
        },
    )


def create_broker_execution_adapter(
    broker_name: str,
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    timeout_seconds: int = DEFAULT_BROKER_TIMEOUT_SECONDS,
) -> BrokerExecutionAdapter:
    """Return a configured broker adapter."""

    normalized_name = broker_name.strip().lower()
    if normalized_name != "alpaca":
        raise BrokerConfigurationError(
            "Unsupported broker adapter. Expected: alpaca."
        )
    return AlpacaBrokerExecutionAdapter.from_env(
        env_file=env_file,
        environ=environ,
        timeout_seconds=timeout_seconds,
    )


@dataclass(frozen=True)
class AlpacaBrokerCredentials:
    """Credential and endpoint settings for the Alpaca broker adapter."""

    api_key: str
    secret_key: str
    base_url: str = DEFAULT_ALPACA_BROKER_BASE_URL


class AlpacaBrokerExecutionAdapter(BrokerExecutionAdapter):
    """Alpaca REST broker adapter."""

    broker_name = "alpaca"

    def __init__(
        self,
        *,
        credentials: AlpacaBrokerCredentials,
        timeout_seconds: int = DEFAULT_BROKER_TIMEOUT_SECONDS,
    ) -> None:
        self.credentials = credentials
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Path | None = None,
        environ: Mapping[str, str] | None = None,
        timeout_seconds: int = DEFAULT_BROKER_TIMEOUT_SECONDS,
    ) -> "AlpacaBrokerExecutionAdapter":
        merged = _read_env_file(env_file)
        merged.update(dict(environ or os.environ))
        api_key = (
            _clean_optional_text(merged.get("ALPACA_BROKER_API_KEY"))
            or _clean_optional_text(merged.get("ALPACA_API_KEY_ID"))
        )
        secret_key = (
            _clean_optional_text(merged.get("ALPACA_BROKER_SECRET_KEY"))
            or _clean_optional_text(merged.get("ALPACA_SECRET_KEY"))
        )
        base_url = (
            _clean_optional_text(merged.get("ALPACA_BROKER_BASE_URL"))
            or _clean_optional_text(merged.get("ALPACA_BASE_URL"))
            or DEFAULT_ALPACA_BROKER_BASE_URL
        )
        if api_key is None or secret_key is None:
            raise BrokerConfigurationError(
                "Alpaca broker credentials are missing. Set ALPACA_API_KEY_ID and "
                "ALPACA_SECRET_KEY, or the ALPACA_BROKER_* equivalents."
            )
        return cls(
            credentials=AlpacaBrokerCredentials(
                api_key=api_key,
                secret_key=secret_key,
                base_url=base_url.rstrip("/"),
            ),
            timeout_seconds=timeout_seconds,
        )

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderUpdate:
        payload = {
            "symbol": request.symbol,
            "side": request.side.lower(),
            "type": request.order_type.lower(),
            "qty": str(request.quantity),
            "time_in_force": request.time_in_force.lower(),
        }
        if request.client_order_id is not None:
            payload["client_order_id"] = request.client_order_id
        if request.limit_price is not None:
            payload["limit_price"] = _format_price(request.limit_price)
        if request.stop_price is not None:
            payload["stop_price"] = _format_price(request.stop_price)
        return _merge_broker_update_with_request_context(
            self._normalize_order_update(
                self._request_json("/v2/orders", method="POST", payload=payload)
            ),
            client_order_id=request.client_order_id,
            metadata=request.metadata,
        )

    def cancel_order(self, broker_order_id: str) -> BrokerOrderUpdate | None:
        self._request_empty(f"/v2/orders/{quote(broker_order_id)}", method="DELETE")
        try:
            return self.get_order(broker_order_id)
        except BrokerRequestError:
            return None

    def replace_order(
        self,
        broker_order_id: str,
        request: BrokerReplaceOrderRequest,
    ) -> BrokerOrderUpdate:
        payload: dict[str, str] = {}
        if request.quantity is not None:
            payload["qty"] = str(request.quantity)
        if request.time_in_force is not None:
            payload["time_in_force"] = request.time_in_force.lower()
        if request.client_order_id is not None:
            payload["client_order_id"] = request.client_order_id
        if request.limit_price is not None:
            payload["limit_price"] = _format_price(request.limit_price)
        if request.stop_price is not None:
            payload["stop_price"] = _format_price(request.stop_price)
        return _merge_broker_update_with_request_context(
            self._normalize_order_update(
                self._request_json(
                    f"/v2/orders/{quote(broker_order_id)}",
                    method="PATCH",
                    payload=payload,
                )
            ),
            client_order_id=request.client_order_id,
            metadata=request.metadata,
        )

    def list_open_orders(self) -> tuple[BrokerOrderUpdate, ...]:
        payload = self._request_json(
            "/v2/orders",
            params={"status": "open", "nested": "false", "limit": "500"},
        )
        if not isinstance(payload, list):
            raise BrokerRequestError("Alpaca returned an unexpected open-orders payload.")
        return tuple(
            sorted(
                (
                    self._normalize_order_update(item)
                    for item in payload
                    if isinstance(item, Mapping)
                ),
                key=lambda update: (
                    update.updated_at or update.submitted_at or "",
                    update.broker_order_id,
                ),
            )
        )

    def get_order(
        self,
        broker_order_id: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> BrokerOrderUpdate:
        if broker_order_id is None and client_order_id is None:
            raise ValueError("broker_order_id or client_order_id is required.")
        if client_order_id is not None:
            payload = self._request_json(
                "/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
            )
        else:
            payload = self._request_json(f"/v2/orders/{quote(broker_order_id or '')}")
        return self._normalize_order_update(payload)

    def list_positions(self) -> tuple[BrokerPositionSnapshot, ...]:
        payload = self._request_json("/v2/positions")
        if not isinstance(payload, list):
            raise BrokerRequestError("Alpaca returned an unexpected positions payload.")
        positions = [
            self._normalize_position_snapshot(item)
            for item in payload
            if isinstance(item, Mapping)
        ]
        return tuple(sorted(positions, key=lambda position: position.symbol))

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        params: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        body = None
        url = f"{self.credentials.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(dict(params))}"
        headers = {
            "APCA-API-KEY-ID": self.credentials.api_key,
            "APCA-API-SECRET-KEY": self.credentials.secret_key,
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(dict(payload), sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore").strip()
            raise BrokerRequestError(
                f"Alpaca broker request failed with HTTP {exc.code}: {message or exc.reason}"
            ) from exc
        except URLError as exc:
            raise BrokerRequestError(
                f"Alpaca broker request failed: {exc.reason}"
            ) from exc

        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise BrokerRequestError("Alpaca returned invalid JSON.") from exc

    def _request_empty(
        self,
        path: str,
        *,
        method: str = "DELETE",
    ) -> None:
        url = f"{self.credentials.base_url}{path}"
        headers = {
            "APCA-API-KEY-ID": self.credentials.api_key,
            "APCA-API-SECRET-KEY": self.credentials.secret_key,
            "Accept": "application/json",
        }
        request = Request(url, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds):
                return
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore").strip()
            raise BrokerRequestError(
                f"Alpaca broker request failed with HTTP {exc.code}: {message or exc.reason}"
            ) from exc
        except URLError as exc:
            raise BrokerRequestError(
                f"Alpaca broker request failed: {exc.reason}"
            ) from exc

    def _normalize_order_update(self, payload: Any) -> BrokerOrderUpdate:
        if not isinstance(payload, Mapping):
            raise BrokerRequestError("Alpaca returned an unexpected order payload.")
        broker_order_id = _required_text(payload.get("id"), "id")
        raw_status = _clean_optional_text(payload.get("status")) or "unknown"
        metadata: dict[str, Any] = {"raw_status": raw_status}
        replaced_by = _clean_optional_text(payload.get("replaced_by"))
        if replaced_by is not None:
            metadata["replaced_by"] = replaced_by
        replaces = _clean_optional_text(payload.get("replaces"))
        if replaces is not None:
            metadata["replaces"] = replaces
        return BrokerOrderUpdate(
            broker_name=self.broker_name,
            broker_order_id=broker_order_id,
            client_order_id=_clean_optional_text(payload.get("client_order_id")),
            symbol=_required_text(payload.get("symbol"), "symbol").upper(),
            side=_normalize_side(payload.get("side")),
            order_type=_normalize_order_type(payload.get("type")),
            quantity=_normalize_positive_int(payload.get("qty"), "qty"),
            filled_quantity=_normalize_non_negative_int(
                payload.get("filled_qty"),
                "filled_qty",
            ),
            status=_normalize_alpaca_status(raw_status),
            limit_price=_normalize_optional_positive_float(payload.get("limit_price")),
            stop_price=_normalize_optional_positive_float(payload.get("stop_price")),
            average_fill_price=_normalize_optional_positive_float(
                payload.get("filled_avg_price")
            ),
            submitted_at=_clean_optional_text(payload.get("submitted_at")),
            updated_at=_clean_optional_text(payload.get("updated_at")),
            metadata=metadata,
        )

    def _normalize_position_snapshot(self, payload: Mapping[str, Any]) -> BrokerPositionSnapshot:
        position_id = _clean_optional_text(payload.get("asset_id"))
        return BrokerPositionSnapshot(
            symbol=_required_text(payload.get("symbol"), "symbol").upper(),
            quantity=_normalize_positive_int(payload.get("qty"), "qty"),
            average_entry_price=_normalize_positive_float(
                payload.get("avg_entry_price"),
                "avg_entry_price",
            ),
            broker_position_id=position_id,
            metadata={},
        )


def _normalize_alpaca_status(raw_status: str) -> str:
    normalized = raw_status.strip().lower()
    mapping = {
        "new": "accepted",
        "accepted": "accepted",
        "pending_new": "accepted",
        "accepted_for_bidding": "accepted",
        "calculated": "accepted",
        "partially_filled": "partially_filled",
        "filled": "filled",
        "canceled": "cancelled",
        "cancelled": "cancelled",
        "expired": "expired",
        "replaced": "replaced",
        "rejected": "rejected",
        "pending_cancel": "pending_cancel",
        "pending_replace": "pending_replace",
        "done_for_day": "done_for_day",
        "stopped": "stopped",
        "suspended": "suspended",
    }
    return mapping.get(normalized, "unknown")


def _merge_broker_update_with_request_context(
    update: BrokerOrderUpdate,
    *,
    client_order_id: str | None,
    metadata: Mapping[str, Any] | None,
) -> BrokerOrderUpdate:
    merged_metadata = dict(metadata or {})
    merged_metadata.update(update.metadata)
    resolved_client_order_id = update.client_order_id or client_order_id
    if (
        resolved_client_order_id == update.client_order_id
        and merged_metadata == update.metadata
    ):
        return update
    return replace(
        update,
        client_order_id=resolved_client_order_id,
        metadata=merged_metadata,
    )


def _normalize_side(value: object) -> str:
    normalized = _required_text(value, "side").upper()
    if normalized not in BROKER_ORDER_SIDES:
        raise BrokerRequestError(
            f"Broker side must be one of {BROKER_ORDER_SIDES}, got '{normalized}'."
        )
    return normalized


def _normalize_order_type(value: object) -> str:
    normalized = _required_text(value, "type").replace("-", "_").upper()
    if normalized not in BROKER_ORDER_TYPES:
        raise BrokerRequestError(
            f"Broker order type must be one of {BROKER_ORDER_TYPES}, got '{normalized}'."
        )
    return normalized


def _normalize_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise BrokerRequestError(f"{field_name} must be an integer.")
    try:
        normalized = int(float(value))
    except (TypeError, ValueError) as exc:
        raise BrokerRequestError(f"{field_name} must be an integer.") from exc
    if normalized <= 0:
        raise BrokerRequestError(f"{field_name} must be greater than zero.")
    return normalized


def _normalize_non_negative_int(value: object, field_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise BrokerRequestError(f"{field_name} must be an integer.")
    try:
        normalized = int(float(value))
    except (TypeError, ValueError) as exc:
        raise BrokerRequestError(f"{field_name} must be an integer.") from exc
    if normalized < 0:
        raise BrokerRequestError(f"{field_name} cannot be negative.")
    return normalized


def _normalize_positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise BrokerRequestError(f"{field_name} must be a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise BrokerRequestError(f"{field_name} must be a number.") from exc
    if normalized <= 0:
        raise BrokerRequestError(f"{field_name} must be greater than zero.")
    return normalized


def _normalize_optional_positive_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if normalized <= 0:
        return None
    return normalized


def _required_text(value: object, field_name: str) -> str:
    cleaned = _clean_optional_text(value)
    if cleaned is None:
        raise BrokerRequestError(f"{field_name} must be a non-empty string.")
    return cleaned


def _clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _format_price(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")
