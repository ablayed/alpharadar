from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
from datetime import datetime
from pathlib import Path
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("p2", Path("scripts/research/p2_data_feasibility.py"))
assert SPEC and SPEC.loader
p2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p2
SPEC.loader.exec_module(p2)


def _row(open_time: int, close_time: int) -> list[str]:
    return [str(open_time), "10", "12", "9", "11", "2", str(close_time), "21", "3", "1", "10", "0"]


def _csv(*rows: list[str]) -> bytes:
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    return output.getvalue().encode()


def test_normalizes_pre_2025_milliseconds() -> None:
    assert p2.normalize_timestamp(1735603200000, interval="1h") == (1735603200000, "milliseconds")


def test_normalizes_post_2025_microseconds() -> None:
    assert p2.normalize_timestamp(1735689600000000, interval="1h") == (1735689600000, "microseconds")


def test_rejects_ambiguous_timestamp() -> None:
    with pytest.raises(p2.AuditError, match="ambiguous"):
        p2.normalize_timestamp(17_356_896_000_000, interval="1h")


def test_boundary_2025_transition_is_explicit() -> None:
    assert p2.normalize_timestamp(1735689599000, interval="1h")[1] == "milliseconds"
    assert p2.normalize_timestamp(1735689600000000, interval="1h")[1] == "microseconds"


def test_checksum_accepts_valid_and_rejects_synthetic_corruption() -> None:
    payload = b"official-fixture"
    checksum = hashlib.sha256(payload).hexdigest() + "  sample.zip\n"
    assert p2.verify_checksum(payload, checksum)
    assert not p2.verify_checksum(payload + b"x", checksum)


def test_parses_valid_headerless_kline() -> None:
    result = p2.parse_kline_csv(_csv(_row(1735603200000, 1735606799999)), interval="1h")
    assert result["row_count"] == 1
    assert result["missing_expected_bars"] == 0


def test_rejects_bad_column_count() -> None:
    with pytest.raises(p2.AuditError, match="columns"):
        p2.parse_kline_csv(b"1,2,3\n", interval="1h")


def test_detects_duplicate_and_missing_bars() -> None:
    start = 1735603200000
    result = p2.parse_kline_csv(_csv(
        _row(start, start + 3_600_000 - 1),
        _row(start, start + 3_600_000 - 1),
        _row(start + 2 * 3_600_000, start + 3 * 3_600_000 - 1),
    ), interval="1h")
    assert result["duplicate_open_times"] == 1
    assert result["missing_expected_bars"] == 1


def test_artifact_writers_are_deterministic(tmp_path: Path) -> None:
    value = {"b": 1, "a": [2, 1]}
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    p2._write_json(first, value)
    p2._write_json(second, value)
    assert first.read_bytes() == second.read_bytes()


def test_audit_module_has_no_production_runtime_or_database_dependency() -> None:
    source = Path("scripts/research/p2_data_feasibility.py").read_text(encoding="utf-8")
    assert "from src" not in source
    assert "import src" not in source
    assert "sqlite" not in source.lower()


def test_lifecycle_gap_fields_for_both_migrations_are_deterministic() -> None:
    def utc(value: str) -> datetime:
        return datetime.fromisoformat(value)

    rndr = p2.lifecycle_gap(
        predecessor_last_open=utc("2024-07-22T02:00:00+00:00"),
        successor_first_open=utc("2024-07-26T08:00:00+00:00"),
        announced_stop=utc("2024-07-22T03:00:00+00:00"),
        announced_start=utc("2024-07-26T08:00:00+00:00"),
    )
    front = p2.lifecycle_gap(
        predecessor_last_open=utc("2024-08-27T02:00:00+00:00"),
        successor_first_open=utc("2024-08-30T08:00:00+00:00"),
        announced_stop=utc("2024-08-27T03:00:00+00:00"),
        announced_start=utc("2024-08-30T08:00:00+00:00"),
    )
    assert rndr == {
        "open_time_delta_hours": 102,
        "missing_bar_count": 101,
        "announced_interruption_hours": 101,
    }
    assert front == {
        "open_time_delta_hours": 78,
        "missing_bar_count": 77,
        "announced_interruption_hours": 77,
    }
