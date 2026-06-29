"""
yoga_classifier.py
==================
Random Forest classifier: 8-D EEG feature vector -> Chitta Bhumi label.

Why Random Forest?
  - ~5 ms inference latency (suitable for real-time use at 1 Hz cadence)
  - Interpretable feature importances (which band drives the classification?)
  - Excellent on tabular features with 2000-4000 training rows
  - No GPU required (runs on Render free tier)
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from neuro_yogic.data_generator import (
    CHITTA_BHUMIS,
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
)


class YogaClassifier:
    """
    Trains a Random Forest on EEG feature vectors and maps them to
    the four Chitta Bhumis of the Yoga Sutras.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self._model = RandomForestClassifier(
            n_estimators  = n_estimators,
            max_depth     = None,
            random_state  = random_state,
            n_jobs        = -1,
            class_weight  = "balanced",
        )
        self._label_enc  = LabelEncoder()
        self._is_trained = False

    def train_model(self, csv_path: str) -> dict:
        """Load CSV, train classifier, return performance metrics."""
        print(f"\n[YogaClassifier] Loading dataset from '{csv_path}' ...")
        df = pd.read_csv(csv_path)

        missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        X = df[FEATURE_COLUMNS].values.astype(np.float64)
        self._label_enc.fit(CHITTA_BHUMIS)
        y = self._label_enc.transform(df["label"].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"[YogaClassifier] Training on {len(X_train)} samples ...")
        self._model.fit(X_train, y_train)
        self._is_trained = True

        cv_scores = cross_val_score(self._model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        y_pred    = self._model.predict(X_test)

        importances = dict(zip(FEATURE_COLUMNS, self._model.feature_importances_.tolist()))
        metrics = {
            "accuracy":            float(self._model.score(X_test, y_test)),
            "cv_mean":             float(cv_scores.mean()),
            "cv_std":              float(cv_scores.std()),
            "feature_importances": importances,
            "n_samples":           len(df),
        }

        print(f"[YogaClassifier] Test accuracy : {metrics['accuracy']:.3f}")
        print(f"[YogaClassifier] CV accuracy   : {metrics['cv_mean']:.3f} +/- {metrics['cv_std']:.3f}")
        print("[YogaClassifier] Feature importances:")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 40)
            print(f"  {feat:<18} {bar} {imp:.4f}")
        return metrics

    def predict(self, features: np.ndarray) -> str:
        """Return the Chitta Bhumi label for a single feature vector."""
        self._require_trained()
        pred_int = self._model.predict(np.atleast_2d(features))[0]
        return str(self._label_enc.inverse_transform([pred_int])[0])

    def predict_proba(self, features: np.ndarray) -> dict:
        """Return {label: probability} dict for a single feature vector."""
        self._require_trained()
        probs = self._model.predict_proba(np.atleast_2d(features))[0]
        return {str(name): round(float(p), 4)
                for name, p in zip(self._label_enc.classes_, probs)}

    def save(self, model_path: str = DEFAULT_MODEL_PATH, labels_path: str = DEFAULT_LABELS_PATH) -> None:
        self._require_trained()
        joblib.dump(self._model,     model_path)
        joblib.dump(self._label_enc, labels_path)
        print(f"[YogaClassifier] Saved -> '{model_path}' + '{labels_path}'")

    def load(self, model_path: str = DEFAULT_MODEL_PATH, labels_path: str = DEFAULT_LABELS_PATH) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model at '{model_path}' -- run with --retrain first.")
        self._model      = joblib.load(model_path)
        self._label_enc  = joblib.load(labels_path)
        self._is_trained = True
        print(f"[YogaClassifier] Loaded from '{model_path}'")

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Not trained. Call train_model() or load() first.")

    @property
    def is_trained(self) -> bool:
        return self._is_trained
