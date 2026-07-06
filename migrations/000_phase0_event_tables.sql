CREATE TABLE IF NOT EXISTS prediction_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    direction TEXT NOT NULL,
    raw_probability REAL,
    confidence_bucket TEXT,
    horizon TEXT NOT NULL,
    entry_price REAL,
    resolution_price REAL,
    resolved BOOLEAN DEFAULT FALSE,
    outcome_label INTEGER,
    gross_return_pct REAL,
    cost_bps REAL,
    net_return_pct REAL,
    model_version TEXT,
    feature_snapshot_id INTEGER,
    migrated_from_v1_id INTEGER,
    UNIQUE(migrated_from_v1_id, horizon)
);

CREATE TABLE IF NOT EXISTS trade_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES prediction_events(id),
    ts TIMESTAMP NOT NULL,
    strategy_id TEXT,
    calibrated_probability REAL,
    p_trade REAL,
    expected_value REAL,
    confidence_bucket TEXT,
    regime TEXT,
    candidate_status TEXT
);

CREATE TABLE IF NOT EXISTS executed_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES trade_candidates(id),
    mode TEXT,
    symbol TEXT,
    status TEXT,
    entry_price REAL,
    stop_price REAL,
    take_profit_price REAL,
    quantity REAL,
    fees REAL,
    slippage REAL,
    gross_return REAL,
    net_return REAL,
    pnl REAL,
    exit_reason TEXT,
    strategy_id TEXT,
    model_version TEXT,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
