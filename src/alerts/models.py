from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(slots=True)
class Alert:
    symbol: str
    asset_type: str
    severity: AlertSeverity
    current_price: float
    change_pct: float
    reference_price: float
    message: str
    timestamp: float = field(default_factory=time.time)

    @property
    def direction(self) -> str:
        return "up" if self.change_pct >= 0 else "down"
