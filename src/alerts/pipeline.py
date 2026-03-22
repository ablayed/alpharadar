from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from src.alerts.models import Alert, AlertSeverity

logger = logging.getLogger(__name__)

LONDON = ZoneInfo("Europe/London")


class AlertPipeline:
    """
    Process real-time ticks into Telegram-worthy alerts.

    The pipeline applies per-asset cooldowns, global hourly limits, quiet hours,
    and short-horizon fingerprint de-duplication to prevent bursty alert spam.
    """

    COOLDOWNS = {
        AlertSeverity.INFO: 4 * 3600,
        AlertSeverity.WARNING: 30 * 60,
        AlertSeverity.CRITICAL: 5 * 60,
    }

    THRESHOLDS = {
        "stock": {
            AlertSeverity.CRITICAL: 5.0,
            AlertSeverity.WARNING: 3.0,
            AlertSeverity.INFO: 2.0,
        },
        "crypto": {
            AlertSeverity.CRITICAL: 12.0,
            AlertSeverity.WARNING: 8.0,
            AlertSeverity.INFO: 5.0,
        },
    }

    GLOBAL_MAX_PER_HOUR = 12
    QUIET_START_HOUR = 23
    QUIET_END_HOUR = 6
    FINGERPRINT_TTL_SECONDS = 90

    def __init__(
        self,
        *,
        time_func: Callable[[], float] | None = None,
        london_now_func: Callable[[], datetime] | None = None,
    ) -> None:
        self._time_func = time_func or time.time
        self._london_now_func = london_now_func or (lambda: datetime.now(LONDON))
        self._last_alert_time: dict[str, float] = {}
        self._alert_history: list[float] = []
        self._reference_prices: dict[str, float] = {}
        self._fingerprints: dict[str, float] = {}
        self._recent_alerts: deque[Alert] = deque(maxlen=20)

    def set_reference_price(self, symbol: str, price: float) -> None:
        if price <= 0:
            return
        self._reference_prices[symbol.upper()] = float(price)

    def classify_severity(self, symbol: str, current_price: float, asset_type: str) -> Alert | None:
        symbol_upper = symbol.upper()
        reference_price = self._reference_prices.get(symbol_upper)
        if reference_price is None or reference_price <= 0 or current_price <= 0:
            return None

        thresholds = self.THRESHOLDS.get(asset_type)
        if thresholds is None:
            return None

        change_pct = ((current_price - reference_price) / reference_price) * 100
        abs_change = abs(change_pct)

        severity: AlertSeverity | None = None
        for level in (AlertSeverity.CRITICAL, AlertSeverity.WARNING, AlertSeverity.INFO):
            if abs_change >= thresholds[level]:
                severity = level
                break

        if severity is None:
            return None

        now = self._time_func()
        self._prune(now)

        if not self._passes_cooldown(symbol_upper, severity, now):
            return None
        if not self._passes_global_limit(severity):
            return None
        if severity is not AlertSeverity.CRITICAL and self._is_quiet_hours():
            return None
        if self._is_duplicate(symbol_upper, change_pct, severity, now):
            return None

        direction = "📈" if change_pct >= 0 else "📉"
        message = (
            f"{direction} <b>{symbol_upper}</b> {self._intensity_word(change_pct, asset_type)} "
            f"{change_pct:+.1f}%\n"
            f"${current_price:,.2f} (from ${reference_price:,.2f})"
        )
        return Alert(
            symbol=symbol_upper,
            asset_type=asset_type,
            severity=severity,
            current_price=float(current_price),
            change_pct=float(change_pct),
            reference_price=float(reference_price),
            message=message,
            timestamp=now,
        )

    def record_sent(self, alert: Alert) -> None:
        symbol_upper = alert.symbol.upper()
        self._last_alert_time[symbol_upper] = alert.timestamp
        self._alert_history.append(alert.timestamp)
        fingerprint = self._fingerprint(symbol_upper, alert.direction, alert.severity)
        self._fingerprints[fingerprint] = alert.timestamp
        self._recent_alerts.appendleft(alert)
        self._prune(alert.timestamp)

    def build_manual_alert(
        self,
        *,
        symbol: str,
        asset_type: str,
        severity: AlertSeverity,
        message: str,
        signed_change: float,
        current_value: float = 0.0,
        reference_value: float = 0.0,
    ) -> Alert | None:
        symbol_upper = symbol.upper()
        now = self._time_func()
        self._prune(now)

        if not self._passes_cooldown(symbol_upper, severity, now):
            return None
        if not self._passes_global_limit(severity):
            return None
        if severity is not AlertSeverity.CRITICAL and self._is_quiet_hours():
            return None
        direction = "up" if signed_change >= 0 else "down"
        if self._is_duplicate(symbol_upper, signed_change, severity, now):
            return None

        del direction
        return Alert(
            symbol=symbol_upper,
            asset_type=asset_type,
            severity=severity,
            current_price=float(current_value),
            change_pct=float(signed_change),
            reference_price=float(reference_value),
            message=message,
            timestamp=now,
        )

    def recent_alert_count(self) -> int:
        now = self._time_func()
        self._prune(now)
        return len(self._alert_history)

    def active_cooldowns(self) -> dict[str, int]:
        now = self._time_func()
        cooldowns: dict[str, int] = {}
        for symbol, last_timestamp in self._last_alert_time.items():
            remaining = max(self.COOLDOWNS[AlertSeverity.INFO] - int(now - last_timestamp), 0)
            if remaining > 0:
                cooldowns[symbol] = remaining
        return cooldowns

    def recent_alerts(self, limit: int = 5) -> list[Alert]:
        return list(self._recent_alerts)[:limit]

    def status(self) -> dict[str, object]:
        return {
            "alerts_this_hour": self.recent_alert_count(),
            "global_limit": self.GLOBAL_MAX_PER_HOUR,
            "quiet_hours": self._is_quiet_hours(),
            "assets_on_cooldown": len(self.active_cooldowns()),
            "recent_alerts": self.recent_alerts(),
        }

    def _passes_cooldown(self, symbol: str, severity: AlertSeverity, now: float) -> bool:
        last_timestamp = self._last_alert_time.get(symbol)
        if last_timestamp is None:
            return True
        return (now - last_timestamp) >= self.COOLDOWNS[severity]

    def _passes_global_limit(self, severity: AlertSeverity) -> bool:
        if severity is AlertSeverity.CRITICAL:
            return True
        return len(self._alert_history) < self.GLOBAL_MAX_PER_HOUR

    def _is_duplicate(
        self,
        symbol: str,
        change_pct: float,
        severity: AlertSeverity,
        now: float,
    ) -> bool:
        direction = "up" if change_pct >= 0 else "down"
        fingerprint = self._fingerprint(symbol, direction, severity)
        previous = self._fingerprints.get(fingerprint)
        return previous is not None and (now - previous) < self.FINGERPRINT_TTL_SECONDS

    def _prune(self, now: float) -> None:
        self._alert_history = [timestamp for timestamp in self._alert_history if (now - timestamp) < 3600]
        self._fingerprints = {
            fingerprint: timestamp
            for fingerprint, timestamp in self._fingerprints.items()
            if (now - timestamp) < self.FINGERPRINT_TTL_SECONDS
        }

    def _is_quiet_hours(self) -> bool:
        now = self._london_now_func()
        return now.hour >= self.QUIET_START_HOUR or now.hour < self.QUIET_END_HOUR

    @staticmethod
    def _fingerprint(symbol: str, direction: str, severity: AlertSeverity) -> str:
        return f"{symbol}:{direction}:{severity.value}"

    @staticmethod
    def _intensity_word(change_pct: float, asset_type: str) -> str:
        abs_change = abs(change_pct)
        if asset_type == "stock":
            if abs_change > 7:
                return "surging" if change_pct > 0 else "crashing"
            if abs_change > 5:
                return "jumping" if change_pct > 0 else "plunging"
            if abs_change > 3:
                return "rallying" if change_pct > 0 else "sliding"
            return "climbing" if change_pct > 0 else "dipping"

        if abs_change > 15:
            return "surging" if change_pct > 0 else "crashing"
        if abs_change > 10:
            return "jumping" if change_pct > 0 else "plunging"
        if abs_change > 5:
            return "rallying" if change_pct > 0 else "sliding"
        return "climbing" if change_pct > 0 else "dipping"
