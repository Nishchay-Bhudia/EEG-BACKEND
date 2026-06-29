"""
data_generator.py
=================
Generates a synthetic EEG feature dataset mimicking the Kaggle
"Meditation EEG Data" and Hugging Face "EEGMeditation" datasets.

Chitta Bhumis (Patanjali Yoga Sutras, 1.1-1.51):
  Kshipta   -- Scattered mind. High Beta, low Alpha.
  Vikshipta -- Oscillating. Alpha rising, Beta moderating.
  Ekagra    -- One-pointed focus. Alpha/Theta dominant, low Beta.
  Niruddha  -- Deep absorption. Theta/Delta dominant, near-zero Beta.

Each row = one 2-second EEG epoch with 8 features + label.
"""

import numpy as np
import pandas as pd

# Band power profiles per state: (mean, std) for relative power.
_STATE_PROFILES = {
    "Kshipta": {
        "delta": (0.10, 0.03), "theta": (0.12, 0.03),
        "alpha": (0.18, 0.04), "beta":  (0.50, 0.06),
        "gamma": (0.10, 0.03), "alpha_left": (0.10, 0.03), "alpha_right": (0.08, 0.03),
    },
    "Vikshipta": {
        "delta": (0.12, 0.03), "theta": (0.15, 0.04),
        "alpha": (0.30, 0.05), "beta":  (0.33, 0.05),
        "gamma": (0.10, 0.03), "alpha_left": (0.15, 0.04), "alpha_right": (0.15, 0.04),
    },
    "Ekagra": {
        "delta": (0.15, 0.04), "theta": (0.22, 0.05),
        "alpha": (0.42, 0.06), "beta":  (0.16, 0.04),
        "gamma": (0.05, 0.02), "alpha_left": (0.18, 0.04), "alpha_right": (0.24, 0.05),
    },
    "Niruddha": {
        "delta": (0.30, 0.06), "theta": (0.35, 0.06),
        "alpha": (0.20, 0.05), "beta":  (0.08, 0.03),
        "gamma": (0.07, 0.04), "alpha_left": (0.10, 0.03), "alpha_right": (0.10, 0.03),
    },
}

FEATURE_COLUMNS = [
    "delta", "theta", "alpha", "beta", "gamma",
    "alpha_left", "alpha_right", "alpha_asymmetry",
]
CHITTA_BHUMIS = ["Kshipta", "Vikshipta", "Ekagra", "Niruddha"]

DEFAULT_MODEL_PATH  = "yoga_classifier.joblib"
DEFAULT_LABELS_PATH = "label_encoder.joblib"
DEFAULT_CSV_PATH    = "mock_eeg_data.csv"


def generate_dataset(n_samples_per_class: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic EEG feature dataset with 8 features and 4 class labels."""
    np.random.seed(random_seed)
    rows = []

    for state, profile in _STATE_PROFILES.items():
        n = n_samples_per_class

        def samp(band, _profile=profile, _n=n):
            m, s = _profile[band]
            return np.clip(np.random.normal(m, s, _n), 0.0, 1.0)

        delta, theta, alpha = samp("delta"), samp("theta"), samp("alpha")
        beta,  gamma        = samp("beta"),  samp("gamma")
        al, ar              = samp("alpha_left"), samp("alpha_right")

        # Normalise band powers so each row sums to 1 (relative power)
        totals = delta + theta + alpha + beta + gamma
        delta /= totals; theta /= totals; alpha /= totals
        beta  /= totals; gamma /= totals

        for i in range(n):
            rows.append({
                "delta":           round(float(delta[i]), 6),
                "theta":           round(float(theta[i]), 6),
                "alpha":           round(float(alpha[i]), 6),
                "beta":            round(float(beta[i]),  6),
                "gamma":           round(float(gamma[i]), 6),
                "alpha_left":      round(float(al[i]),    6),
                "alpha_right":     round(float(ar[i]),    6),
                "alpha_asymmetry": round(float(ar[i] - al[i]), 6),
                "label":           state,
            })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=random_seed).reset_index(drop=True)


def save_dataset(
    path: str = DEFAULT_CSV_PATH,
    n_samples_per_class: int = 500,
) -> str:
    """Generate and save the dataset to a CSV file. Returns the path."""
    df = generate_dataset(n_samples_per_class=n_samples_per_class)
    df.to_csv(path, index=False)
    print(f"[DataGenerator] Saved {len(df)} rows -> '{path}'")
    return path
