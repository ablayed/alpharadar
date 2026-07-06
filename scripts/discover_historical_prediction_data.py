from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

CANDIDATE_COLUMN_TOKENS = (
    "symbol",
    "source",
    "direction",
    "prediction_ts",
    "actual_price",
    "actual_change",
    "is_correct",
    "outcome_label",
    "resolved",
    "horizon",
    "confidence",
    "probability",
)
HIGH_SIGNAL_TERMS = ("37.3", "33.3", "29.6", "99 resolved")
GENERIC_TERMS = (
    "accuracy",
    "actual_price",
    "actual_change",
    "is_correct",
    "outcome_label",
    "resolved",
    "briefing",
    "auto_trade_scan",
)
TEXT_SUFFIXES = {".py", ".json", ".csv", ".md", ".log", ".sql"}
DB_SUFFIXES = {".db", ".sqlite"}
IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".serena",
    "__pycache__",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover likely historical prediction data sources.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file to inspect.")
    parser.add_argument(
        "--search-dir",
        default=".",
        help="Root directory to search for candidate files and databases.",
    )
    return parser


def _candidate_overlap(columns: list[str]) -> list[str]:
    lowered_columns = [column.lower() for column in columns]
    overlap = [
        token
        for token in CANDIDATE_COLUMN_TOKENS
        if any(token in column for column in lowered_columns)
    ]
    return overlap


def _quoted_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _db_table_report(db_path: Path) -> tuple[list[str], list[str]]:
    connection = sqlite3.connect(db_path)
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()

        table_lines = ["DB inspection:"]
        candidate_lines = ["Candidate tables:"]
        candidate_found = False

        for (table_name,) in table_rows:
            quoted_name = _quoted_identifier(str(table_name))
            columns = [str(row[1]) for row in connection.execute(f"PRAGMA table_info({quoted_name})")]
            row_count = int(connection.execute(f"SELECT COUNT(*) FROM {quoted_name}").fetchone()[0])
            overlap = _candidate_overlap(columns)
            likely_candidate = len(overlap) >= 3
            table_lines.append(
                f"- {table_name}: rows={row_count}; likely_candidate={'yes' if likely_candidate else 'no'}; columns={', '.join(columns)}"
            )
            if likely_candidate:
                candidate_found = True
                candidate_lines.append(
                    f"- {table_name}: overlap={', '.join(overlap)}; rows={row_count}"
                )

        if not candidate_found:
            candidate_lines.append("- none found")
        return table_lines, candidate_lines
    finally:
        connection.close()


def _docker_volume_roots(search_dir: Path) -> list[Path]:
    docker_compose_path = search_dir / "docker-compose.yml"
    if not docker_compose_path.exists():
        return []

    roots: list[Path] = []
    for raw_line in docker_compose_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        mapping = line[2:].strip().strip('"').strip("'")
        if ":" not in mapping:
            continue
        host_path = mapping.split(":", 1)[0]
        if host_path.startswith("./"):
            candidate = (search_dir / host_path[2:]).resolve()
        else:
            candidate = (search_dir / host_path).resolve()
        if candidate.exists():
            roots.append(candidate)
    return roots


def _iter_files(search_dir: Path) -> Iterable[Path]:
    for path in search_dir.rglob("*"):
        if any(part in IGNORED_DIR_NAMES or part.startswith("pytest-cache-files-") for part in path.parts):
            continue
        if path.is_file():
            yield path


def _db_files(search_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in _iter_files(search_dir)
        if path.suffix.lower() in DB_SUFFIXES
    )


def _generic_scan_roots(search_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for dirname in ("data", "logs", "exports", "reports", "backups"):
        candidate = (search_dir / dirname).resolve()
        if candidate.exists():
            roots.append(candidate)
    roots.extend(_docker_volume_roots(search_dir))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            deduped.append(root)
    return deduped


def _is_under_any_root(path: Path, roots: list[Path]) -> bool:
    resolved_path = path.resolve()
    for root in roots:
        try:
            resolved_path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _filename_matches(path: Path) -> bool:
    lowered = path.name.lower()
    filename_terms = [*HIGH_SIGNAL_TERMS, *GENERIC_TERMS]
    return any(term.lower() in lowered for term in filename_terms)


def _search_text_files(search_dir: Path) -> list[str]:
    high_signal_pattern = re.compile("|".join(re.escape(term) for term in HIGH_SIGNAL_TERMS), re.IGNORECASE)
    generic_pattern = re.compile("|".join(re.escape(term) for term in GENERIC_TERMS), re.IGNORECASE)
    generic_roots = _generic_scan_roots(search_dir)

    hits: list[str] = []
    for path in _iter_files(search_dir):
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            continue

        if _filename_matches(path):
            hits.append(f"- filename: {path}")

        should_generic_scan = _is_under_any_root(path, generic_roots) or path.parent.resolve() == search_dir.resolve()
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for line_number, line in enumerate(lines, start=1):
            if high_signal_pattern.search(line):
                hits.append(f"- {path}:{line_number}: {line.strip()}")
                continue
            if should_generic_scan and generic_pattern.search(line):
                hits.append(f"- {path}:{line_number}: {line.strip()}")
    return hits


def discover(db_path: str, search_dir: str) -> str:
    db_file = Path(db_path).resolve()
    search_root = Path(search_dir).resolve()

    table_lines, candidate_lines = _db_table_report(db_file)
    db_files = _db_files(search_root)
    text_hits = _search_text_files(search_root)
    scan_roots = _generic_scan_roots(search_root)

    lines: list[str] = []
    lines.extend(table_lines)
    lines.append("")
    lines.extend(candidate_lines)
    lines.append("")
    lines.append("Filesystem scan roots:")
    for root in scan_roots:
        lines.append(f"- {root}")
    lines.append("")
    lines.append("Other *.db/*.sqlite files found:")
    if db_files:
        for db_candidate in db_files:
            lines.append(f"- {db_candidate}")
    else:
        lines.append("- none found")
    lines.append("")
    lines.append("Filesystem search hits:")
    if text_hits:
        lines.extend(text_hits)
    else:
        lines.append("- none found")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_arg_parser().parse_args(argv)
    print(discover(args.db_path, args.search_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
