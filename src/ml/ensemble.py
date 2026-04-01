"""
Lightweight ML ensemble for next-day directional prediction.
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data") / "ml_models"
UP_VOTE_THRESHOLD = 0.55
DOWN_VOTE_THRESHOLD = 0.45


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


class MLEnsemble:
    """Train and serve a compact three-model ensemble per symbol."""

    RETRAIN_INTERVAL = 7 * 86400
    MIN_TRAINING_SAMPLES = 60

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        min_training_samples: int | None = None,
    ) -> None:
        self.model_dir = model_dir or MODEL_DIR
        self._models: dict[str, dict[str, Any]] = {}
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
    trained_at: float,
) -> EnsemblePrediction:
    bullish_votes = [prob for prob in probs.values() if prob >= UP_VOTE_THRESHOLD]
    bearish_votes = [1.0 - prob for prob in probs.values() if prob <= DOWN_VOTE_THRESHOLD]

    if len(bullish_votes) >= 2:
        agreement = len(bullish_votes)
        avg_prob = float(np.mean(bullish_votes))
        return EnsemblePrediction(
            symbol=symbol,
            direction="bullish",
            confidence=round(avg_prob, 3),
            xgb_prob=round(probs["xgb"], 3),
            rf_prob=round(probs["rf"], 3),
            lr_prob=round(probs["lr"], 3),
            agreement=agreement,
            description=f"ML ensemble: {agreement}/3 models predict upside (avg prob: {avg_prob:.0%})",
            last_trained=trained_at,
        )

    if len(bearish_votes) >= 2:
        agreement = len(bearish_votes)
        avg_prob = float(np.mean(bearish_votes))
        return EnsemblePrediction(
            symbol=symbol,
            direction="bearish",
            confidence=round(avg_prob, 3),
            xgb_prob=round(probs["xgb"], 3),
            rf_prob=round(probs["rf"], 3),
            lr_prob=round(probs["lr"], 3),
            agreement=agreement,
            description=f"ML ensemble: {agreement}/3 models predict downside (avg prob: {avg_prob:.0%})",
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
        description="ML ensemble: no 2/3 consensus yet",
        last_trained=trained_at,
    )
