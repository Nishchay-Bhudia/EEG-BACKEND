"""
yoga_classifier.py — ML Training Module (YogaClassifier)
=========================================================
Trains a Random Forest classifier on EEG feature vectors and maps them
to the four Chitta Bhumis (Patanjali's Yoga Sutras, 1.1–1.51):

  Kshipta   — "Scattered": mind in constant restless motion (Vikshepa).
               High Beta power; Alpha weak. The default human condition.

  Vikshipta — "Oscillating": moments of clarity interrupted by distraction.
               Alpha rising but Beta still present. Common in beginners.

  Ekagra    — "One-pointed": sustained, effortless single-pointed focus.
               Strong Alpha/Theta, subdued Beta. The entry to Dharana.

  Niruddha  — "Restrained": all fluctuations (Vrittis) cease.
               Dominant Theta/Delta, near-zero Beta. Samadhi territory.

Feature vector (8 dimensions):
  [delta, theta, alpha, beta, gamma,
   alpha_left, alpha_right, alpha_asymmetry]

All powers are relative (0–1) and normalized to sum ≈ 1 across bands.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# Ordered list of feature columns expected by the model
FEATURE_COLUMNS = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "alpha_left",
    "alpha_right",
    "alpha_asymmetry",
]

# Canonical label ordering (from most scattered to deepest absorption)
CHITTA_BHUMIS = ["Kshipta", "Vikshipta", "Ekagra", "Niruddha"]

# Default paths for persisting the trained model artefacts
DEFAULT_MODEL_PATH  = "neuro_yogic/yoga_classifier.joblib"
DEFAULT_LABELS_PATH = "neuro_yogic/label_encoder.joblib"


class YogaClassifier:
    """
    Trains, persists, and serves a Random Forest classifier that maps
    8-dimensional EEG feature vectors to Chitta Bhumi states.

    Usage
    -----
    # --- Training phase ---
    clf = YogaClassifier()
    metrics = clf.train_model("neuro_yogic/mock_eeg_data.csv")
    clf.save()

    # --- Inference phase (load from disk) ---
    clf = YogaClassifier()
    clf.load()
    state = clf.predict(feature_vector)
    probabilities = clf.predict_proba(feature_vector)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the Random Forest. 200 balances accuracy
            with real-time inference latency (~5 ms on modern hardware).
        max_depth : int or None
            Maximum depth of each tree. None → trees grow until leaves
            are pure (may overfit; use with enough training data).
        random_state : int
            Seed for reproducibility.
        """
        # Random Forest is an ensemble of decorrelated decision trees.
        # Each tree is trained on a bootstrap sample of the training data
        # (bagging) and considers only a random subset of features at each
        # split (feature randomisation). The majority vote across all trees
        # produces the final prediction, reducing variance considerably.
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,       # use all CPU cores during training
            class_weight="balanced",  # handle class imbalance gracefully
        )
        # LabelEncoder maps string labels ↔ integer class indices
        self._label_enc = LabelEncoder()
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self, csv_path: str) -> dict:
        """
        Load a labelled EEG feature CSV, train the classifier, and return
        a dictionary of performance metrics.

        The CSV must contain the columns listed in FEATURE_COLUMNS plus
        a "label" column with Chitta Bhumi names.

        Parameters
        ----------
        csv_path : str
            Path to the training CSV (real dataset or synthetic mock).

        Returns
        -------
        dict
            Keys: accuracy, cv_mean, cv_std, classification_report,
                  confusion_matrix, feature_importances, n_samples.
        """
        print(f"\n[YogaClassifier] Loading dataset from '{csv_path}' …")
        df = pd.read_csv(csv_path)

        # Validate columns
        missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Expected: {FEATURE_COLUMNS + ['label']}"
            )

        X = df[FEATURE_COLUMNS].values.astype(np.float64)
        y_raw = df["label"].values

        # Encode string labels to integers (required by sklearn)
        # Fit the encoder on all known Chitta Bhumis so class indices are
        # stable even if a class is absent from this particular dataset.
        self._label_enc.fit(CHITTA_BHUMIS)
        y = self._label_enc.transform(y_raw)

        # Hold-out split: 80% train, 20% test — stratified to preserve
        # class proportions in both splits.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"[YogaClassifier] Training on {len(X_train)} samples …")
        self._model.fit(X_train, y_train)
        self._is_trained = True

        # --- Evaluation -----------------------------------------------

        # 5-fold stratified cross-validation on the full dataset gives a
        # less biased estimate of generalisation performance than a single
        # train/test split.
        cv_scores = cross_val_score(
            self._model, X, y, cv=5, scoring="accuracy", n_jobs=-1
        )

        y_pred = self._model.predict(X_test)
        class_names = self._label_enc.classes_

        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importance: how much each feature reduces impurity
        # (Gini importance, averaged across all trees).
        importances = dict(
            zip(FEATURE_COLUMNS, self._model.feature_importances_.tolist())
        )

        metrics = {
            "accuracy":               float(self._model.score(X_test, y_test)),
            "cv_mean":                float(cv_scores.mean()),
            "cv_std":                 float(cv_scores.std()),
            "classification_report":  report,
            "confusion_matrix":       cm,
            "feature_importances":    importances,
            "n_samples":              len(df),
        }

        print(f"[YogaClassifier] Test accuracy : {metrics['accuracy']:.3f}")
        print(f"[YogaClassifier] CV accuracy   : {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        print(f"\n[YogaClassifier] Feature importances (top-to-bottom):")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"  {feat:<18} {bar} {imp:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> str:
        """
        Predict the Chitta Bhumi from a single feature vector.

        Parameters
        ----------
        features : np.ndarray, shape (8,) or (1, 8)
            [delta, theta, alpha, beta, gamma,
             alpha_left, alpha_right, alpha_asymmetry]

        Returns
        -------
        str
            One of: 'Kshipta', 'Vikshipta', 'Ekagra', 'Niruddha'
        """
        self._require_trained()
        features = np.atleast_2d(features)
        pred_int = self._model.predict(features)[0]
        return str(self._label_enc.inverse_transform([pred_int])[0])

    def predict_proba(self, features: np.ndarray) -> dict:
        """
        Return class probabilities (confidence scores) for a feature vector.

        Returns
        -------
        dict  {chitta_bhumi_name: probability, …}
        """
        self._require_trained()
        features = np.atleast_2d(features)
        probs = self._model.predict_proba(features)[0]
        class_names = self._label_enc.classes_
        return {name: round(float(p), 4) for name, p in zip(class_names, probs)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        labels_path: str = DEFAULT_LABELS_PATH,
    ) -> None:
        """Persist the trained model and label encoder to disk."""
        self._require_trained()
        joblib.dump(self._model,      model_path)
        joblib.dump(self._label_enc,  labels_path)
        print(f"[YogaClassifier] Model saved  → '{model_path}'")
        print(f"[YogaClassifier] Labels saved → '{labels_path}'")

    def load(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        labels_path: str = DEFAULT_LABELS_PATH,
    ) -> None:
        """Load a previously trained model and label encoder from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at '{model_path}'. "
                "Run train_model() and save() first."
            )
        self._model      = joblib.load(model_path)
        self._label_enc  = joblib.load(labels_path)
        self._is_trained = True
        print(f"[YogaClassifier] Model loaded from '{model_path}'")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Classifier is not trained. "
                "Call train_model() or load() first."
            )

    @property
    def is_trained(self) -> bool:
        return self._is_trained


if __name__ == "__main__":
    from neuro_yogic.data_generator import save_dataset

    # 1. Generate mock training data
    csv_path = save_dataset(n_samples_per_class=600)

    # 2. Train the classifier
    clf = YogaClassifier(n_estimators=200)
    metrics = clf.train_model(csv_path)

    # 3. Persist to disk
    clf.save()

    # 4. Demonstrate inference with a hand-crafted "Ekagra" feature vector
    ekagra_features = np.array([0.15, 0.22, 0.42, 0.16, 0.05, 0.18, 0.24, 0.06])
    state = clf.predict(ekagra_features)
    probs = clf.predict_proba(ekagra_features)
    print(f"\n[Demo] Predicted state: {state}")
    print(f"[Demo] Probabilities  : {json.dumps(probs, indent=2)}")
