"""
Lightweight ML ensemble for next-day directional prediction.
"""
from __future__ import annotations

import logging
import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data") / "ml_models"
PERFORMANCE_FILE = MODEL_DIR / "performance_log.json"
LOOKBACK_WINDOW = 14
UP_VOTE_THRESHOLD = 0.5
DOWN_VOTE_THRESHOLD = 0.5
BULLISH_CONFIRMATION_THRESHOLD = 0.60
BEARISH_CONFIRMATION_THRESHOLD = 0.40


@dataclass(frozen=True)
class MLDependencies:
    XGBClassifier: type[Any]
    RandomForestClassifier: type[Any]
    LogisticRegression: type[Any]
    StandardScaler: type[Any]


@dataclass(slots=True)
class EnsemblePrediction:
    symbol: str
    direction: str
    confidence: float
    xgb_prob: float
    rf_prob: float
    lr_prob: float
    agreement: int
    description: str
    last_trained: float


@dataclass(slots=True)
class ModelPerformance:
    """Tracks rolling profitability for each model in the ensemble."""

    history: list[dict[str, Any]] = field(default_factory=list)
    performance_file: Path = PERFORMANCE_FILE

    def add_outcome(
        self,
        symbol: str,
        model_name: str,
        predicted_up: bool,
        actual_up: bool,
        prob: float,
        price_change_pct: float,
    ) -> None:
        """Record a resolved prediction outcome for one model."""
        pnl = price_change_pct if predicted_up else -price_change_pct
        now = time.time()
        self.history.append(
            {
                "timestamp": now,
                "symbol": symbol.upper(),
                "model": model_name,
                "predicted_up": predicted_up,
                "actual_up": actual_up,
                "prob": round(float(prob), 4),
                "pnl_pct": round(float(pnl), 4),
                "correct": predicted_up == actual_up,
            }
        )

        cutoff = now - 90 * 86400
        self.history = [
            outcome
            for outcome in self.history
            if float(outcome.get("timestamp", 0.0) or 0.0) > cutoff
        ]

    def get_model_weight(self, model_name: str, lookback_days: int = LOOKBACK_WINDOW) -> float:
        """Calculate a model's raw weight from recent exponentially-decayed PnL."""
        now = time.time()
        cutoff = now - lookback_days * 86400
        recent = [
            outcome
            for outcome in self.history
            if outcome.get("model") == model_name
            and float(outcome.get("timestamp", 0.0) or 0.0) > cutoff
        ]

        if not recent:
            return 1.0 / 3.0

        half_life = lookback_days * 86400 / 2
        weighted_pnl = 0.0
        total_weight = 0.0
        for outcome in recent:
            age = max(now - float(outcome.get("timestamp", 0.0) or 0.0), 0.0)
            decay = 2.0 ** (-age / half_life)
            weighted_pnl += float(outcome.get("pnl_pct", 0.0) or 0.0) * decay
            total_weight += decay

        if total_weight == 0:
            return 1.0 / 3.0

        avg_pnl = max(-20.0, min(20.0, weighted_pnl / total_weight))
        weight = 1.0 / (1.0 + math.exp(-avg_pnl * 0.7))
        return max(0.1, min(1.0, weight))

    def get_all_weights(self, lookback_days: int = LOOKBACK_WINDOW) -> dict[str, float]:
        """Get normalized profitability weights for all ensemble members."""
        raw = {
            "xgb": self.get_model_weight("xgb", lookback_days),
            "rf": self.get_model_weight("rf", lookback_days),
            "lr": self.get_model_weight("lr", lookback_days),
        }
        total = sum(raw.values())
        if total <= 0:
            return {"xgb": 0.3333, "rf": 0.3333, "lr": 0.3333}
        return {model_name: round(value / total, 4) for model_name, value in raw.items()}

    def save(self) -> None:
        """Persist performance history to disk."""
        try:
            self.performance_file.parent.mkdir(parents=True, exist_ok=True)
            with self.performance_file.open("w", encoding="utf-8") as handle:
                json.dump(self.history, handle)
        except Exception:
            logger.exception("Failed to save ML performance history")

    @classmethod
    def load(cls, performance_file: Path = PERFORMANCE_FILE) -> "ModelPerformance":
        """Load performance history from disk."""
        perf = cls(performance_file=performance_file)
        try:
            if performance_file.exists():
                with performance_file.open(encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, list):
                    perf.history = [item for item in loaded if isinstance(item, dict)]
        except Exception:
            logger.exception("Failed to load ML performance history")
        return perf


class MLEnsemble:
    """Train and serve a compact three-model ensemble per symbol."""

    RETRAIN_INTERVAL = 7 * 86400
    MIN_TRAINING_SAMPLES = 60

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        min_training_samples: int | None = None,
        performance_file: Path | None = None,
    ) -> None:
        self.model_dir = model_dir or MODEL_DIR
        self._models: dict[str, dict[str, Any]] = {}
        self._performance = ModelPerformance.load(performance_file or PERFORMANCE_FILE)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if min_training_samples is not None:
            self.MIN_TRAINING_SAMPLES = min_training_samples

    def train(
        self,
        symbol: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: list[str] | None = None,
    ) -> bool:
        """Train all models for one symbol and persist them to disk."""
        symbol_upper = symbol.upper()
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float64)

        valid_mask = ~np.any(np.isnan(X_array), axis=1) & ~np.isnan(y_array)
        X_clean = np.nan_to_num(
            X_array[valid_mask],
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        y_clean = y_array[valid_mask].astype(np.int32)

        if len(X_clean) < self.MIN_TRAINING_SAMPLES:
            logger.warning(
                "Not enough clean ML samples for %s: %d < %d",
                symbol_upper,
                len(X_clean),
                self.MIN_TRAINING_SAMPLES,
            )
            return False
        if len(np.unique(y_clean)) < 2:
            logger.warning("Skipping ML training for %s: labels only contain one class", symbol_upper)
            return False

        try:
            deps = _load_ml_dependencies()
        except ImportError:
            logger.warning(
                "Skipping ML training for %s because xgboost/scikit-learn are unavailable",
                symbol_upper,
            )
            return False

        try:
            scaler = deps.StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            xgb = deps.XGBClassifier(
                n_estimators=64,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                reg_lambda=1.0,
                tree_method="hist",
                n_jobs=1,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
            xgb.fit(X_clean, y_clean)

            rf = deps.RandomForestClassifier(
                n_estimators=64,
                max_depth=5,
                min_samples_leaf=4,
                n_jobs=1,
                random_state=42,
            )
            rf.fit(X_clean, y_clean)

            lr = deps.LogisticRegression(
                max_iter=500,
                C=0.25,
                random_state=42,
            )
            lr.fit(X_scaled, y_clean)

            trained_at = time.time()
            payload: dict[str, Any] = {
                "xgb": xgb,
                "rf": rf,
                "lr": lr,
                "scaler": scaler,
                "trained_at": trained_at,
                "n_samples": len(X_clean),
                "feature_names": list(feature_names or []),
            }
            self._models[symbol_upper] = payload

            with (self.model_dir / f"{symbol_upper}_ensemble.pkl").open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(
                "Trained ML ensemble for %s on %d samples and %d features",
                symbol_upper,
                len(X_clean),
                X_clean.shape[1],
            )
            return True
        except Exception:
            logger.exception("ML training failed for %s", symbol_upper)
            return False

    def predict(self, symbol: str, X_latest: np.ndarray) -> EnsemblePrediction | None:
        """Get the ensemble vote for the latest feature row."""
        symbol_upper = symbol.upper()
        models = self._get_models(symbol_upper)
        if models is None:
            return None

        try:
            X = np.asarray(X_latest, dtype=np.float32)
            latest_row = X.reshape(1, -1) if X.ndim == 1 else X[-1:, :]
            latest_row = np.nan_to_num(latest_row, nan=0.0, posinf=1e6, neginf=-1e6)
            scaled_row = models["scaler"].transform(latest_row)

            xgb_prob = float(models["xgb"].predict_proba(latest_row)[0, 1])
            rf_prob = float(models["rf"].predict_proba(latest_row)[0, 1])
            lr_prob = float(models["lr"].predict_proba(scaled_row)[0, 1])

            return _vote_prediction(
                symbol=symbol_upper,
                probs={"xgb": xgb_prob, "rf": rf_prob, "lr": lr_prob},
                weights=self._performance.get_all_weights(),
                trained_at=float(models["trained_at"]),
            )
        except Exception:
            logger.exception("ML prediction failed for %s", symbol_upper)
            return None

    def predict_from_ohlcv(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        open_: np.ndarray | None = None,
    ) -> EnsemblePrediction | None:
        """Engineer features on demand and run the latest-row prediction."""
        from src.ml.features import engineer_features

        try:
            features, _feature_names = engineer_features(close, high, low, volume, open_)
        except Exception:
            logger.exception("Failed to engineer ML features for %s", symbol.upper())
            return None
        return self.predict(symbol, features)

    def needs_retrain(self, symbol: str) -> bool:
        """Whether a symbol is missing models or the existing models are stale."""
        models = self._get_models(symbol.upper())
        if models is None:
            return True
        return (time.time() - float(models["trained_at"])) > self.RETRAIN_INTERVAL

    def get_feature_importance(self, symbol: str) -> dict[str, float] | None:
        """Return XGBoost feature importance mapped to feature names when available."""
        models = self._get_models(symbol.upper())
        if models is None:
            return None

        feature_names = list(models.get("feature_names", []))
        importances = getattr(models.get("xgb"), "feature_importances_", None)
        if not isinstance(importances, np.ndarray):
            return None

        if feature_names and len(feature_names) == len(importances):
            pairs = zip(feature_names, importances, strict=False)
        else:
            pairs = ((str(index), value) for index, value in enumerate(importances))

        return {
            name: float(score)
            for name, score in sorted(pairs, key=lambda item: float(item[1]), reverse=True)
        }

    def _get_models(self, symbol: str) -> dict[str, Any] | None:
        symbol_upper = symbol.upper()
        if symbol_upper in self._models:
            return self._models[symbol_upper]

        model_path = self.model_dir / f"{symbol_upper}_ensemble.pkl"
        if not model_path.exists():
            return None

        try:
            with model_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            logger.exception("Failed to load ML ensemble for %s", symbol_upper)
            return None

        if not isinstance(payload, dict):
            logger.warning("Ignoring malformed ML payload for %s", symbol_upper)
            return None

        self._models[symbol_upper] = payload
        return payload


def _load_ml_dependencies() -> MLDependencies:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return MLDependencies(
        XGBClassifier=XGBClassifier,
        RandomForestClassifier=RandomForestClassifier,
        LogisticRegression=LogisticRegression,
        StandardScaler=StandardScaler,
    )


def _vote_prediction(
    *,
    symbol: str,
    probs: dict[str, float],
    weights: dict[str, float],
    trained_at: float,
) -> EnsemblePrediction:
    weights = {
        "xgb": float(weights.get("xgb", 1.0 / 3.0)),
        "rf": float(weights.get("rf", 1.0 / 3.0)),
        "lr": float(weights.get("lr", 1.0 / 3.0)),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {"xgb": 1.0 / 3.0, "rf": 1.0 / 3.0, "lr": 1.0 / 3.0}
    else:
        weights = {model_name: weight / total_weight for model_name, weight in weights.items()}

    weighted_up_prob = sum(weights[model_name] * probs[model_name] for model_name in ("xgb", "rf", "lr"))
    model_up_votes = {
        model_name: prob > UP_VOTE_THRESHOLD
        for model_name, prob in probs.items()
    }
    votes_up = sum(model_up_votes.values())
    votes_down = 3 - votes_up
    weight_text = _format_weight_text(weights)

    if votes_up == 3:
        return EnsemblePrediction(
            symbol=symbol,
            direction="bullish",
            confidence=round(weighted_up_prob, 3),
            xgb_prob=round(probs["xgb"], 3),
            rf_prob=round(probs["rf"], 3),
            lr_prob=round(probs["lr"], 3),
            agreement=3,
            description=(
                f"ML ensemble: 3/3 unanimous bullish "
                f"(weighted prob: {weighted_up_prob:.0%}) {weight_text}"
            ),
            last_trained=trained_at,
        )

    if votes_down == 3:
        confidence = 1.0 - weighted_up_prob
        return EnsemblePrediction(
            symbol=symbol,
            direction="bearish",
            confidence=round(confidence, 3),
            xgb_prob=round(probs["xgb"], 3),
            rf_prob=round(probs["rf"], 3),
            lr_prob=round(probs["lr"], 3),
            agreement=3,
            description=(
                f"ML ensemble: 3/3 unanimous bearish "
                f"(weighted prob: {confidence:.0%}) {weight_text}"
            ),
            last_trained=trained_at,
        )

    if votes_up >= 2 and weighted_up_prob > BULLISH_CONFIRMATION_THRESHOLD:
        return EnsemblePrediction(
            symbol=symbol,
            direction="bullish",
            confidence=round(weighted_up_prob, 3),
            xgb_prob=round(probs["xgb"], 3),
            rf_prob=round(probs["rf"], 3),
            lr_prob=round(probs["lr"], 3),
            agreement=votes_up,
            description=(
                f"ML ensemble: {votes_up}/3 bullish, weighted confirms "
                f"(prob: {weighted_up_prob:.0%}) {weight_text}"
            ),
            last_trained=trained_at,
        )

    if votes_down >= 2 and weighted_up_prob < BEARISH_CONFIRMATION_THRESHOLD:
        confidence = 1.0 - weighted_up_prob
        return EnsemblePrediction(
            symbol=symbol,
            direction="bearish",
            confidence=round(confidence, 3),
            xgb_prob=round(probs["xgb"], 3),
            rf_prob=round(probs["rf"], 3),
            lr_prob=round(probs["lr"], 3),
            agreement=votes_down,
            description=(
                f"ML ensemble: {votes_down}/3 bearish, weighted confirms "
                f"(prob: {confidence:.0%}) {weight_text}"
            ),
            last_trained=trained_at,
        )

    return EnsemblePrediction(
        symbol=symbol,
        direction="neutral",
        confidence=0.5,
        xgb_prob=round(probs["xgb"], 3),
        rf_prob=round(probs["rf"], 3),
        lr_prob=round(probs["lr"], 3),
        agreement=0,
        description=(
            f"ML ensemble: no consensus "
            f"(weighted prob: {weighted_up_prob:.0%}, votes: {votes_up}/3 up) {weight_text}"
        ),
        last_trained=trained_at,
    )


def _format_weight_text(weights: dict[str, float]) -> str:
    return f"[XGB:{weights['xgb']:.0%} RF:{weights['rf']:.0%} LR:{weights['lr']:.0%}]"
