"""Bounded, audit-only Binance Spot archive feasibility probe for P2.

This module is deliberately standalone: it imports no application package and never
opens a database.  Network use is anonymous and capped by ``MAX_DOWNLOAD_BYTES``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
KLINE_COLUMNS = 12
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
ARCHIVE_ROOT = "https://data.binance.vision/data/spot/monthly/klines"
ACCESS_DATE = "2026-07-20"


class AuditError(ValueError):
    """A strict validation failure; official files are never repaired."""


@dataclass(frozen=True)
class Probe:
    symbol: str
    interval: str
    period: str
    purpose: str
    download: bool = False

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.interval}-{self.period}.zip"

    @property
    def url(self) -> str:
        return f"{ARCHIVE_ROOT}/{self.symbol}/{self.interval}/{self.filename}"


# Representative, deliberately small set.  The event evidence is documented in
# the report; archive absence is never classified as a delisting by this script.
PROBES = (
    Probe("BTCUSDT", "1h", "2024-12", "pre-2025 timestamp sample", True),
    Probe("BTCUSDT", "1h", "2025-01", "timestamp transition sample", True),
    Probe("ETHUSDT", "4h", "2025-01", "4h schema sample", True),
    Probe("SOLUSDT", "1d", "2026-06", "latest completed monthly archive", True),
    Probe("ADAUSDT", "1h", "2021-05", "long-lived liquid altcoin", False),
    Probe("KAITOUSDT", "1h", "2025-02", "recent listing first-month candidate", True),
    Probe("MANTAUSDT", "1d", "2024-01", "recent listing first-month candidate", False),
    Probe("DREPUSDT", "1d", "2025-04", "pre-delisting continuity", True),
    Probe("RNDRUSDT", "1h", "2024-07", "rebrand/migration predecessor", True),
    Probe("AMBUSDT", "1h", "2025-02", "AirDAO full-Spot-delisting predecessor", True),
    Probe("AMBUSDT", "1h", "2025-03", "AirDAO post-cessation archive observation", False),
    Probe("ALPACAUSDT", "1h", "2025-05", "Alpaca full-Spot-delisting predecessor", True),
    Probe("ALPACAUSDT", "1h", "2025-06", "Alpaca post-cessation archive observation", False),
    Probe("RENDERUSDT", "1h", "2024-07", "Render migration successor", True),
    Probe("FRONTUSDT", "1h", "2024-08", "Frontier migration predecessor", True),
    Probe("SLFUSDT", "1h", "2024-08", "Self Chain migration successor", True),
)


def normalize_timestamp(raw: int, *, interval: str) -> tuple[int, str]:
    """Return UTC milliseconds plus detected archive unit, rejecting ambiguity."""
    if interval not in INTERVAL_MS:
        raise AuditError(f"unsupported interval: {interval}")
    # Unix milliseconds for Binance's supported archive era are 13 digits;
    # microseconds are 16 digits.  A fixed epoch window prevents magnitude-only
    # acceptance of arbitrary large integers.
    if 1_500_000_000_000 <= raw <= 2_100_000_000_000:
        milliseconds, unit = raw, "milliseconds"
    elif 1_500_000_000_000_000 <= raw <= 2_100_000_000_000_000:
        if raw % 1000:
            # Microsecond precision is valid, but kline *open* times must land
            # exactly on millisecond boundaries after conversion.
            raise AuditError(f"microsecond kline open time is not millisecond-aligned: {raw}")
        milliseconds, unit = raw // 1000, "microseconds"
    else:
        raise AuditError(f"ambiguous or out-of-range archive timestamp: {raw}")
    try:
        datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise AuditError(f"invalid UTC timestamp: {raw}") from exc
    return milliseconds, unit


def normalize_close_timestamp(raw: int, *, interval: str) -> tuple[int, str]:
    """Normalize a kline close time, whose microsecond value ends in 999999."""
    if 1_500_000_000_000 <= raw <= 2_100_000_000_000:
        return raw, "milliseconds"
    if 1_500_000_000_000_000 <= raw <= 2_100_000_000_000_000:
        return raw // 1000, "microseconds"
    raise AuditError(f"ambiguous or out-of-range archive timestamp: {raw}")


def lifecycle_gap(
    *,
    predecessor_last_open: datetime,
    successor_first_open: datetime,
    announced_stop: datetime,
    announced_start: datetime,
    interval: str = "1h",
) -> dict[str, int]:
    """Calculate unambiguous migration gap fields from timezone-aware UTC times."""
    if interval not in INTERVAL_MS:
        raise AuditError(f"unsupported interval: {interval}")
    values = (predecessor_last_open, successor_first_open, announced_stop, announced_start)
    if any(value.tzinfo is None or value.utcoffset() is None for value in values):
        raise AuditError("lifecycle timestamps must be timezone-aware")
    step_seconds = INTERVAL_MS[interval] // 1000
    open_delta_seconds = int((successor_first_open - predecessor_last_open).total_seconds())
    interruption_seconds = int((announced_start - announced_stop).total_seconds())
    if open_delta_seconds < 0 or interruption_seconds < 0:
        raise AuditError("lifecycle timestamps are not ordered")
    if open_delta_seconds % step_seconds or interruption_seconds % step_seconds:
        raise AuditError("lifecycle timestamps are not aligned to the interval")
    return {
        "open_time_delta_hours": open_delta_seconds // 3600,
        "missing_bar_count": max(0, open_delta_seconds // step_seconds - 1),
        "announced_interruption_hours": interruption_seconds // 3600,
    }


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


LIFECYCLE_GAPS = {
    "RNDRUSDT_TO_RENDERUSDT": lifecycle_gap(
        predecessor_last_open=_utc("2024-07-22T02:00:00+00:00"),
        successor_first_open=_utc("2024-07-26T08:00:00+00:00"),
        announced_stop=_utc("2024-07-22T03:00:00+00:00"),
        announced_start=_utc("2024-07-26T08:00:00+00:00"),
    ),
    "FRONTUSDT_TO_SLFUSDT": lifecycle_gap(
        predecessor_last_open=_utc("2024-08-27T02:00:00+00:00"),
        successor_first_open=_utc("2024-08-30T08:00:00+00:00"),
        announced_stop=_utc("2024-08-27T03:00:00+00:00"),
        announced_start=_utc("2024-08-30T08:00:00+00:00"),
    ),
}


def verify_checksum(payload: bytes, checksum_text: str) -> bool:
    """Verify a Binance sha256sum-style companion file."""
    expected = checksum_text.strip().split()[0].lower() if checksum_text.strip() else ""
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise AuditError("checksum companion does not start with a SHA-256 digest")
    return hashlib.sha256(payload).hexdigest() == expected


def _safe_csv_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    members = archive.infolist()
    if len(members) != 1:
        raise AuditError(f"expected exactly one ZIP member, got {len(members)}")
    member = members[0]
    path = PurePosixPath(member.filename)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".csv":
        raise AuditError(f"unsafe or non-CSV ZIP member: {member.filename}")
    return member


def parse_kline_csv(payload: bytes, *, interval: str) -> dict[str, Any]:
    """Strictly validate headerless Spot kline CSV data and return observations."""
    expected_step = INTERVAL_MS[interval]
    rows = list(csv.reader(io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8", newline="")))
    if not rows:
        raise AuditError("empty kline CSV")
    open_times: list[int] = []
    units: set[str] = set()
    anomalies: list[str] = []
    for number, row in enumerate(rows, start=1):
        if len(row) != KLINE_COLUMNS:
            raise AuditError(f"row {number}: expected {KLINE_COLUMNS} columns, got {len(row)}")
        try:
            raw_open, raw_close = int(row[0]), int(row[6])
            open_, high, low, close, volume, quote_volume = map(float, (row[1], row[2], row[3], row[4], row[5], row[7]))
            trades = int(row[8])
            taker_base, taker_quote = float(row[9]), float(row[10])
        except ValueError as exc:
            raise AuditError(f"row {number}: non-numeric kline field") from exc
        open_ms, unit = normalize_timestamp(raw_open, interval=interval)
        close_ms, close_unit = normalize_close_timestamp(raw_close, interval=interval)
        units.update((unit, close_unit))
        if close_ms - open_ms + 1 != expected_step:
            raise AuditError(f"row {number}: candle duration does not match {interval}")
        if high < max(open_, close, low) or low > min(open_, close, high):
            raise AuditError(f"row {number}: OHLC invariant failed")
        if min(volume, quote_volume, taker_base, taker_quote) < 0 or trades < 0:
            raise AuditError(f"row {number}: negative volume or trade count")
        open_times.append(open_ms)
    duplicates = len(open_times) - len(set(open_times))
    if duplicates:
        anomalies.append(f"duplicate_open_times={duplicates}")
    if open_times != sorted(open_times):
        anomalies.append("open_times_not_monotonic")
    missing = sum(max(0, (b - a) // expected_step - 1) for a, b in zip(open_times, open_times[1:]))
    return {
        "row_count": len(rows),
        "first_open_time_utc": datetime.fromtimestamp(open_times[0] / 1000, tz=UTC).isoformat(),
        "last_open_time_utc": datetime.fromtimestamp(open_times[-1] / 1000, tz=UTC).isoformat(),
        "timestamp_units": sorted(units),
        "duplicate_open_times": duplicates,
        "missing_expected_bars": missing,
        "anomalies": anomalies,
    }


def validate_zip(payload: bytes, *, interval: str) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = _safe_csv_member(archive)
        return {"zip_member": member.filename, **parse_kline_csv(archive.read(member), interval=interval)}


class BudgetClient:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.downloaded = 0

    def request(self, client: httpx.Client, method: str, url: str) -> tuple[httpx.Response, bytes]:
        with client.stream(method, url, follow_redirects=True) as response:
            declared = int(response.headers.get("content-length", "0") or 0)
            if declared and self.downloaded + declared > self.limit:
                raise AuditError(f"budget would exceed {self.limit} bytes before downloading {url}")
            chunks: list[bytes] = []
            for chunk in response.iter_bytes():
                if self.downloaded + len(chunk) > self.limit:
                    raise AuditError(f"budget would exceed {self.limit} while downloading {url}")
                self.downloaded += len(chunk)
                chunks.append(chunk)
            return response, b"".join(chunks)


def run_probe(*, output_dir: Path, cache_dir: Path, max_bytes: int = MAX_DOWNLOAD_BYTES) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    budget = BudgetClient(max_bytes)
    availability: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(20.0), headers={"User-Agent": "AlphaRadar-P2-audit/1.0"}) as client:
        for probe in PROBES:
            try:
                response, body = budget.request(client, "HEAD", probe.url)
                status, size = response.status_code, int(response.headers.get("content-length", "0") or 0)
                if response.status_code in (405, 501):
                    response, body = budget.request(client, "GET", probe.url)
                    status, size = response.status_code, len(body)
            except Exception as exc:
                availability.append({**asdict(probe), "url": probe.url, "status": "ERROR", "error": str(exc), "bytes": 0})
                continue
            availability.append({**asdict(probe), "url": probe.url, "status": status, "bytes": size})
            if not probe.download or status != 200:
                continue
            response, zip_bytes = budget.request(client, "GET", probe.url)
            checksum_url = probe.url + ".CHECKSUM"
            checksum_response, checksum_bytes = budget.request(client, "GET", checksum_url)
            checksum_ok = checksum_response.status_code == 200 and verify_checksum(zip_bytes, checksum_bytes.decode("utf-8"))
            cached = cache_dir / probe.filename
            cached.write_bytes(zip_bytes)
            try:
                detail = validate_zip(zip_bytes, interval=probe.interval)
                error: str | None = None
            except (AuditError, zipfile.BadZipFile) as exc:
                detail, error = {}, str(exc)
            validations.append({**asdict(probe), "url": probe.url, "checksum_url": checksum_url,
                                "zip_bytes": len(zip_bytes), "checksum_bytes": len(checksum_bytes),
                                "checksum_ok": checksum_ok, "validation_error": error, **detail})
    availability.sort(key=lambda item: (item["symbol"], item["interval"], item["period"]))
    validations.sort(key=lambda item: (item["symbol"], item["interval"], item["period"]))
    _write_csv(output_dir / "availability_matrix.csv", availability)
    _write_json(output_dir / "sample_validation.json", {
        "access_date": ACCESS_DATE,
        "lifecycle_gaps": LIFECYCLE_GAPS,
        "validations": validations,
    })
    return {"actual_bytes_downloaded": budget.downloaded, "availability": availability, "validations": validations}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 bounded Binance archive feasibility probe")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/p2"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/p2-cache"))
    parser.add_argument("--max-bytes", type=int, default=MAX_DOWNLOAD_BYTES)
    args = parser.parse_args(argv)
    result = run_probe(output_dir=args.output_dir, cache_dir=args.cache_dir, max_bytes=args.max_bytes)
    print(json.dumps({"actual_bytes_downloaded": result["actual_bytes_downloaded"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
