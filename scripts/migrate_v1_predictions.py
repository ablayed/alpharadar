from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.audit.cost_model import resolve_round_trip_cost_bps

MIGRATION_SQL_PATH = Path(__file__).resolve().parents[1] / "migrations" / "000_phase0_event_tables.sql"
SUPPORTED_HORIZONS = ("1d", "3d", "7d")
HORIZON_COLUMNS: dict[str, tuple[str, str, str]] = {
    "1d": ("actual_price_1d", "actual_change_1d_pct", "is_correct_1d"),
    "3d": ("actual_price_3d", "actual_change_3d_pct", "is_correct_3d"),
    "7d": ("actual_price_7d", "actual_change_7d_pct", "is_correct_7d"),
}


@dataclass(slots=True)
class MigrationResult:
    predictions_seen: int = 0
    inserted_rows: int = 0
    skipped_existing_rows: int = 0
    skipped_missing_horizon_columns: dict[str, int] = field(default_factory=dict)
    skipped_row_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _apply_migration_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(MIGRATION_SQL_PATH.read_text(encoding="utf-8"))
    connection.commit()


def _normalize_direction(value: object) -> str:
    normalized = str(value or "").strip().lower()
    mapping = {
        "bullish": "BULLISH",
        "long": "BULLISH",
        "buy": "BULLISH",
        "bearish": "BEARISH",
        "short": "BEARISH",
        "sell": "BEARISH",
        "neutral": "NEUTRAL",
    }
    return mapping.get(normalized, "NEUTRAL")


def _confidence_bucket(raw_probability: object) -> str | None:
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError):
        return None

    if probability < 0 or probability > 1:
        return None

    lower = min(int(probability * 20) / 20, 0.95)
    upper = min(lower + 0.05, 1.0)
    return f"{lower:.2f}_{upper:.2f}"


def _metadata_dict(raw_metadata: object) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not isinstance(raw_metadata, str) or not raw_metadata.strip():
        return {}

    try:
        parsed = json.loads(raw_metadata)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _feature_snapshot_id(metadata: dict[str, Any]) -> int | None:
    for key in ("feature_snapshot_id", "feature_snapshot", "snapshot_id"):
        value = _optional_int(metadata.get(key))
        if value is not None:
            return value
    return None


def _model_version(metadata: dict[str, Any]) -> str | None:
    for key in ("model_version", "model", "ml_model_version"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolved_flag(row: sqlite3.Row, price_column: str, change_column: str, correct_column: str) -> bool:
    return any(row[column] is not None for column in (price_column, change_column, correct_column) if column in row.keys())


def _gross_return_pct(direction: str, actual_change_pct: float | None) -> float | None:
    if actual_change_pct is None:
        return None
    if direction == "BULLISH":
        return actual_change_pct
    if direction == "BEARISH":
        return -actual_change_pct
    return None


def _build_prediction_event_row(
    row: sqlite3.Row,
    *,
    horizon: str,
    available_columns: set[str],
) -> tuple[Any, ...] | None:
    price_column, change_column, correct_column = HORIZON_COLUMNS[horizon]
    required_row_columns = {"id", "created_at", "symbol", "source", "predicted_direction", "confidence", "price_at_prediction"}
    if not required_row_columns.issubset(row.keys()):
        missing = sorted(required_row_columns.difference(row.keys()))
        raise ValueError(f"missing required prediction columns: {missing}")

    metadata = _metadata_dict(row["metadata"]) if "metadata" in row.keys() else {}
    direction = _normalize_direction(row["predicted_direction"])
    actual_change_pct = _optional_float(row[change_column]) if change_column in available_columns else None
    gross_return_pct = _gross_return_pct(direction, actual_change_pct)
    cost_bps = None
    net_return_pct = None
    if gross_return_pct is not None:
        cost_bps = resolve_round_trip_cost_bps()
        net_return_pct = gross_return_pct - (cost_bps / 100.0)

    return (
        row["created_at"],
        str(row["symbol"]),
        str(row["source"]),
        direction,
        _optional_float(row["confidence"]),
        _confidence_bucket(row["confidence"]),
        horizon,
        _optional_float(row["price_at_prediction"]),
        _optional_float(row[price_column]) if price_column in available_columns else None,
        int(_resolved_flag(row, price_column, change_column, correct_column)),
        _optional_int(row[correct_column]) if correct_column in available_columns else None,
        gross_return_pct,
        cost_bps,
        net_return_pct,
        _model_version(metadata),
        _feature_snapshot_id(metadata),
        int(row["id"]),
    )


def migrate_v1_predictions(db_path: str) -> MigrationResult:
    result = MigrationResult()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        _apply_migration_schema(connection)
        if not _table_exists(connection, "predictions"):
            result.notes.append("No legacy predictions table found; schema only migration applied.")
            return result

        legacy_columns = _table_columns(connection, "predictions")
        missing_horizons = {
            horizon: columns
            for horizon, columns in HORIZON_COLUMNS.items()
            if not set(columns).issubset(legacy_columns)
        }
        for horizon, columns in missing_horizons.items():
            result.skipped_missing_horizon_columns[horizon] = int(
                connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            )
            result.notes.append(
                f"Skipped horizon {horizon}: missing legacy columns {', '.join(columns)}"
            )

        predictions = connection.execute("SELECT * FROM predictions ORDER BY id").fetchall()
        result.predictions_seen = len(predictions)
        insert_sql = """
            INSERT OR IGNORE INTO prediction_events (
                ts,
                symbol,
                source,
                direction,
                raw_probability,
                confidence_bucket,
                horizon,
                entry_price,
                resolution_price,
                resolved,
                outcome_label,
                gross_return_pct,
                cost_bps,
                net_return_pct,
                model_version,
                feature_snapshot_id,
                migrated_from_v1_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for prediction in predictions:
            for horizon in SUPPORTED_HORIZONS:
                required_columns = set(HORIZON_COLUMNS[horizon])
                if not required_columns.issubset(legacy_columns):
                    continue

                try:
                    event_row = _build_prediction_event_row(
                        prediction,
                        horizon=horizon,
                        available_columns=legacy_columns,
                    )
                except Exception as exc:
                    result.skipped_row_errors.append(
                        f"prediction_id={prediction['id']} horizon={horizon}: {type(exc).__name__}: {exc}"
                    )
                    continue

                if event_row is None:
                    continue

                cursor = connection.execute(insert_sql, event_row)
                if cursor.rowcount == 1:
                    result.inserted_rows += 1
                else:
                    result.skipped_existing_rows += 1

        connection.commit()
        return result
    finally:
        connection.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Best-effort backfill from v1 predictions into prediction_events.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file.")
    return parser


def main(argv: list[str] | None = None, *, db_path: str | None = None) -> int:
    if db_path is None:
        args = _build_arg_parser().parse_args(argv)
        db_path = args.db_path

    result = migrate_v1_predictions(db_path)
    print(f"predictions_seen={result.predictions_seen}")
    print(f"inserted_rows={result.inserted_rows}")
    print(f"skipped_existing_rows={result.skipped_existing_rows}")
    if result.skipped_missing_horizon_columns:
        print(f"skipped_missing_horizon_columns={json.dumps(result.skipped_missing_horizon_columns, sort_keys=True)}")
    for note in result.notes:
        print(note)
    for error in result.skipped_row_errors:
        print(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
