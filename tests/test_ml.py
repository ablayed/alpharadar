from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from src.ml.ensemble import MLDependencies, MLEnsemble
from src.ml.features import engineer_features


def _sample_ohlcv(length: int = 140) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(length, dtype=np.float64)
    close = 100.0 + (index * 0.35) + np.sin(index / 5.0) * 2.0
    open_ = close - 0.4 + np.cos(index / 6.0) * 0.15
    high = np.maximum(open_, close) + 1.2
    low = np.minimum(open_, close) - 1.1
    volume = 1_000.0 + (index * 8.0) + np.sin(index / 4.0) * 75.0
    return close, high, low, volume, open_


def _model_dir() -> Path:
    model_dir = Path(".test_tmp") / f"ml-{uuid4().hex}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def test_engineer_features_returns_dense_matrix_after_warmup() -> None:
    close, high, low, volume, open_ = _sample_ohlcv()

    X, feature_names = engineer_features(close, high, low, volume, open_)

    assert X.shape[0] == len(close)
    assert X.shape[1] >= 50
    assert len(feature_names) == X.shape[1]
    assert "rsi_14" in feature_names
    assert "macd_histogram" in feature_names
    assert "volume_ratio" in feature_names
    assert "return_14d" in feature_names
    assert np.isnan(X[:50]).all()
    assert np.isfinite(X[80:]).all()


class _FakeScaler:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class _BaseFakeModel:
    default_prob = 0.5

    def __init__(self, **_kwargs: object) -> None:
        self.n_features_in_ = 0
        self.feature_importances_ = np.array([], dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseFakeModel":
        del y
        self.n_features_in_ = int(X.shape[1])
        self.feature_importances_ = np.linspace(1.0, 0.1, self.n_features_in_, dtype=np.float32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rows = int(X.shape[0])
        up = np.full((rows, 1), self.default_prob, dtype=np.float32)
        down = 1.0 - up
        return np.hstack([down, up])


class _FakeXGB(_BaseFakeModel):
    default_prob = 0.72


class _FakeRF(_BaseFakeModel):
    default_prob = 0.67


class _FakeLR(_BaseFakeModel):
    default_prob = 0.43


def test_ml_ensemble_trains_predicts_and_persists(monkeypatch) -> None:
    close, high, low, volume, open_ = _sample_ohlcv()
    X, feature_names = engineer_features(close, high, low, volume, open_)
    y = np.full(len(close), np.nan, dtype=np.float64)
    y[:-1] = (close[1:] > close[:-1]).astype(np.float64)
    model_dir = _model_dir()

    monkeypatch.setattr(
        "src.ml.ensemble._load_ml_dependencies",
        lambda: MLDependencies(
            XGBClassifier=_FakeXGB,
            RandomForestClassifier=_FakeRF,
            LogisticRegression=_FakeLR,
            StandardScaler=_FakeScaler,
        ),
    )

    ensemble = MLEnsemble(model_dir=model_dir, min_training_samples=30)

    assert ensemble.train("BTC", X, y, feature_names=feature_names) is True

    prediction = ensemble.predict("BTC", X[-1])
    assert prediction is not None
    assert prediction.direction == "bullish"
    assert prediction.agreement == 2
    assert prediction.confidence > 0.6
    assert "2/3 models predict upside" in prediction.description
    assert (model_dir / "BTC_ensemble.pkl").exists()

    importance = ensemble.get_feature_importance("BTC")
    assert importance is not None
    assert list(importance.keys())[0] == feature_names[0]


def test_ml_ensemble_degrades_gracefully_without_dependencies(monkeypatch) -> None:
    def _missing_dependencies() -> MLDependencies:
        raise ImportError("missing")

    monkeypatch.setattr("src.ml.ensemble._load_ml_dependencies", _missing_dependencies)
    ensemble = MLEnsemble(model_dir=_model_dir(), min_training_samples=10)
    X = np.ones((20, 5), dtype=np.float64)
    y = np.array([0, 1] * 10, dtype=np.float64)

    assert ensemble.train("ETH", X, y, feature_names=["a", "b", "c", "d", "e"]) is False
