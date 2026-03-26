from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class OpenPosition:
    symbol: str
    asset_type: str
    direction: str
    entry_price: float
    stop_loss: float
    tp1: float | None
    risk_pct: float
    opened_at: float
    tp1_hit: bool = False
    dead_money_alerted: bool = False


class PortfolioTracker:
    MONTHLY_CAP_PCT = 6.0
    CORRELATION_THRESHOLD = 0.7

    def __init__(self) -> None:
        self._positions: dict[str, OpenPosition] = {}
        self._monthly_realized_loss_pct: float = 0.0
        self._month_start_balance: float = 0.0

    @property
    def positions(self) -> dict[str, OpenPosition]:
        return self._positions

    def add_position(self, position: OpenPosition) -> None:
        self._positions[position.symbol.upper()] = position

    def remove_position(self, symbol: str) -> None:
        self._positions.pop(symbol.upper(), None)

    def mark_tp1_hit(self, symbol: str) -> None:
        position = self._positions.get(symbol.upper())
        if position is not None:
            position.tp1_hit = True

    def mark_dead_money_alerted(self, symbol: str) -> None:
        position = self._positions.get(symbol.upper())
        if position is not None:
            position.dead_money_alerted = True

    def get_total_risk(self) -> float:
        return sum(position.risk_pct for position in self._positions.values())

    def check_risk_budget(
        self,
        new_risk_pct: float,
        *,
        replacing_symbol: str | None = None,
    ) -> tuple[bool, str]:
        current_risk = self.get_total_risk()
        if replacing_symbol is not None:
            existing = self._positions.get(replacing_symbol.upper())
            if existing is not None:
                current_risk -= existing.risk_pct

        total_after = current_risk + new_risk_pct + self._monthly_realized_loss_pct
        if total_after > self.MONTHLY_CAP_PCT:
            return False, (
                f"Adding {new_risk_pct:.1f}% would exceed the 6% monthly risk cap. "
                f"Current open risk: {max(current_risk, 0.0):.1f}%, "
                f"realized losses: {self._monthly_realized_loss_pct:.1f}%. "
                "Close an existing trade first."
            )

        return True, ""

    def check_correlation_overlap(
        self,
        new_symbol: str,
        correlation_with_existing: dict[str, float],
    ) -> str | None:
        warnings: list[str] = []
        new_symbol_upper = new_symbol.upper()
        for existing_symbol, position in self._positions.items():
            if existing_symbol == new_symbol_upper:
                continue
            correlation = correlation_with_existing.get(existing_symbol, 0.0)
            if abs(correlation) > self.CORRELATION_THRESHOLD:
                warnings.append(
                    f"{new_symbol_upper} is highly correlated ({correlation:.2f}) with your open "
                    f"{position.symbol} position. Combined risk should count as about "
                    f"{1 + abs(correlation):.1f} positions, not 2."
                )
        return "\n".join(warnings) if warnings else None

    def get_dead_money_trades(self, threshold_days: int = 7) -> list[str]:
        now = time.time()
        stale: list[str] = []
        for symbol, position in self._positions.items():
            age_days = (now - position.opened_at) / 86400
            if age_days > threshold_days and not position.tp1_hit:
                stale.append(symbol)
        return stale

    def format_status(self) -> str:
        total_risk = self.get_total_risk()
        position_count = len(self._positions)
        budget_used = total_risk + self._monthly_realized_loss_pct
        budget_remaining = max(0.0, self.MONTHLY_CAP_PCT - budget_used)

        lines = [
            "<b>PORTFOLIO RISK</b>",
            "",
            f"Open positions: {position_count}",
            f"Total risk: {total_risk:.1f}% of portfolio",
            f"Monthly losses: {self._monthly_realized_loss_pct:.1f}%",
            f"Risk budget remaining: {budget_remaining:.1f}% (of 6% cap)",
        ]

        if self._positions:
            lines.extend(["", "<b>Open:</b>"])
            now = time.time()
            for symbol in sorted(self._positions):
                position = self._positions[symbol]
                age_days = (now - position.opened_at) / 86400
                status = "TP1 hit" if position.tp1_hit else f"{age_days:.0f}d open"
                lines.append(
                    f"{position.symbol} {position.direction} | "
                    f"{position.asset_type} | risk: {position.risk_pct:.1f}% | {status}"
                )

        stale = self.get_dead_money_trades()
        if stale:
            lines.extend(
                [
                    "",
                    f"<b>Dead money (7+ days, no TP1):</b> {', '.join(stale)}",
                    "Consider closing these to free up risk budget.",
                ]
            )

        return "\n".join(lines)
