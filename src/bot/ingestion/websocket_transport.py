"""Concrete JSON websocket transport implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Callable, Mapping, Protocol, Sequence

from bot.logging_utils import get_logger


LOGGER = get_logger(__name__)


class WebsocketConnection(Protocol):
    """Minimal websocket client connection contract."""

    def send(self, payload: str) -> None:
        """Send one message."""

    def recv(self) -> str:
        """Receive one raw message."""

    def close(self) -> None:
        """Close the connection."""

    def settimeout(self, timeout: float | None) -> None:
        """Update the socket read timeout."""


def default_websocket_connection_factory(
    url: str,
    *,
    timeout_seconds: float | None,
) -> WebsocketConnection:
    """Create a websocket-client connection lazily if the dependency is available."""

    try:
        from websocket import create_connection  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError(
            "websocket-client is required for live websocket transport support."
        ) from exc
    connection = create_connection(url, timeout=timeout_seconds)
    return connection


def parse_json_websocket_message(
    raw_message: str,
) -> tuple[dict[str, Any], ...]:
    """Parse one raw websocket payload into normalized message mappings."""

    payload = json.loads(raw_message)
    if payload is None:
        return ()
    if isinstance(payload, Mapping):
        return (dict(payload),)
    if isinstance(payload, list):
        messages: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"websocket list payload item {index} must be a mapping."
                )
            messages.append(dict(item))
        return tuple(messages)
    raise TypeError("websocket payload must be a JSON object or list of JSON objects.")


@dataclass
class JsonWebsocketTransport:
    """Concrete websocket transport that parses raw JSON messages."""

    url: str
    connection_factory: Callable[[str], WebsocketConnection] | None = None
    message_parser: Callable[[str], Sequence[Mapping[str, Any]]] = parse_json_websocket_message
    subscription_messages: Sequence[Mapping[str, Any] | str] = ()
    receive_timeout_seconds: float = 0.25
    max_messages_per_read: int = 100
    transport_name: str = "json-websocket"
    _connection: WebsocketConnection | None = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    _last_message_at_utc: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("url cannot be empty.")
        if not self.transport_name.strip():
            raise ValueError("transport_name cannot be empty.")
        if self.receive_timeout_seconds <= 0:
            raise ValueError("receive_timeout_seconds must be greater than zero.")
        if self.max_messages_per_read <= 0:
            raise ValueError("max_messages_per_read must be greater than zero.")
        if self.connection_factory is None:
            self.connection_factory = lambda url: default_websocket_connection_factory(
                url,
                timeout_seconds=self.receive_timeout_seconds,
            )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_message_at_utc(self) -> datetime | None:
        return self._last_message_at_utc

    def connect(self) -> None:
        if self._connected and self._connection is not None:
            return
        if self.connection_factory is None:
            raise RuntimeError("connection_factory must be configured.")
        connection: WebsocketConnection | None = None
        try:
            connection = self.connection_factory(self.url)
            try:
                connection.settimeout(self.receive_timeout_seconds)
            except AttributeError:
                pass
            self._connection = connection
            self._connected = True
            self._send_subscriptions()
        except Exception as exc:
            if connection is not None:
                self._close_connection_best_effort(connection)
            self._connection = None
            self._connected = False
            if isinstance(exc, OSError):
                raise
            raise OSError(str(exc)) from exc

    def disconnect(self) -> None:
        connection = self._connection
        self._connection = None
        self._connected = False
        if connection is None:
            return
        self._close_connection_best_effort(connection)

    def read_messages(
        self,
        *,
        max_messages: int | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        if not self._connected or self._connection is None:
            raise OSError("websocket transport is not connected.")
        limit = self.max_messages_per_read if max_messages is None else max_messages
        messages: list[dict[str, Any]] = []
        while len(messages) < limit:
            try:
                raw_message = self._connection.recv()
            except Exception as exc:
                if _is_timeout_exception(exc):
                    break
                self.disconnect()
                if isinstance(exc, OSError):
                    raise
                raise OSError(str(exc)) from exc
            if raw_message is None:
                break
            try:
                parsed_messages = self.message_parser(raw_message)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning("Skipping malformed websocket payload: %s", exc)
                continue
            if not parsed_messages:
                continue
            self._last_message_at_utc = _utcnow()
            for parsed_message in parsed_messages:
                messages.append(dict(parsed_message))
                if len(messages) >= limit:
                    break
        return tuple(messages)

    def _send_subscriptions(self) -> None:
        if self._connection is None:
            return
        for subscription in self.subscription_messages:
            payload = subscription
            if not isinstance(subscription, str):
                payload = json.dumps(subscription, sort_keys=True)
            self._connection.send(str(payload))

    def _close_connection_best_effort(self, connection: WebsocketConnection) -> None:
        try:
            connection.close()
        except Exception:
            LOGGER.warning("Ignoring websocket close error during disconnect.", exc_info=True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return exc.__class__.__name__ == "WebSocketTimeoutException"
