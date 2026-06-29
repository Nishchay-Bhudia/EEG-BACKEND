"""
data_generator.py — Synthetic EEG Dataset Generator
=====================================================
Generates a realistic mock dataset that mimics the structure of the
Kaggle "Meditation EEG Data" and Hugging Face "EEGMeditation" datasets.

Each row represents one 2-second EEG epoch and contains:
  - Relative band powers for Delta, Theta, Alpha, Beta, Gamma
  - Left and Right hemisphere Alpha power
  - Hemispheric Alpha Asymmetry (Right − Left)
  - Label: Chitta Bhumi state

Chitta Bhumis (cognitive states per Yoga Sutras of Patanjali):
  - Kshipta    : Scattered / restless — dominant Beta, low Alpha
  - Vikshipta  : Oscillating / partially stable — moderate Alpha/Beta mix
  - Ekagra     : One-pointed concentration — dominant Alpha/Theta
  - Niruddha   : Deep absorption (Samadhi onset) — dominant Theta/Delta,
                  low Beta, possible Gamma bursts
"""

import numpy as np
import pandas as pd


# Frequency band power profiles per Chitta Bhumi state.
# Each tuple is (mean, std) for the RELATIVE power of that band (0–1 scale).
# Values are loosely grounded in published EEG meditation literature:
#   - Alpha increases with focused attention and relaxation.
#   - Theta increases in deep meditative absorption.
#   - Delta increases in near-sleep / deepest states.
#   - Beta decreases as mental chatter subsides.
#   - Gamma can transiently spike during insight/binding events.
STATE_PROFILES = {
    "Kshipta": {
        # High Beta (mind-wandering, default mode network active),
        # low Alpha (attention diffuse).
        "delta": (0.10, 0.03),
        "theta": (0.12, 0.03),
        "alpha": (0.18, 0.04),
        "beta":  (0.50, 0.06),
        "gamma": (0.10, 0.03),
        # Asymmetry skewed: left frontal Beta dominance (approach/agitation)
        "alpha_left":  (0.10, 0.03),
        "alpha_right": (0.08, 0.03),
    },
    "Vikshipta": {
        # Transitional: Alpha rising, Beta moderating.
        "delta": (0.12, 0.03),
        "theta": (0.15, 0.04),
        "alpha": (0.30, 0.05),
        "beta":  (0.33, 0.05),
        "gamma": (0.10, 0.03),
        "alpha_left":  (0.15, 0.04),
        "alpha_right": (0.15, 0.04),
    },
    "Ekagra": {
        # Strong Alpha, elevated Theta, subdued Beta — classic "flow" signature.
        "delta": (0.15, 0.04),
        "theta": (0.22, 0.05),
        "alpha": (0.42, 0.06),
        "beta":  (0.16, 0.04),
        "gamma": (0.05, 0.02),
        # Right-hemisphere Alpha dominance (Ida / parasympathetic tendency).
        "alpha_left":  (0.18, 0.04),
        "alpha_right": (0.24, 0.05),
    },
    "Niruddha": {
        # Deep absorption: Theta/Delta dominant, near-zero Beta,
        # occasional transient Gamma bursts (binding / insight events).
        "delta": (0.30, 0.06),
        "theta": (0.35, 0.06),
        "alpha": (0.20, 0.05),
        "beta":  (0.08, 0.03),
        "gamma": (0.07, 0.04),  # higher std → transient spikes possible
        "alpha_left":  (0.10, 0.03),
        "alpha_right": (0.10, 0.03),
    },
}


def _sample_band(profile: dict, band: str, n: int) -> np.ndarray:
    """Draw n samples from a clipped normal for a single band."""
    mean, std = profile[band]
    samples = np.random.normal(mean, std, n)
    return np.clip(samples, 0.0, 1.0)


def generate_dataset(n_samples_per_class: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic EEG feature dataset.

    Parameters
    ----------
    n_samples_per_class : int
        Number of 2-second epochs to simulate per Chitta Bhumi class.
    random_seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: delta, theta, alpha, beta, gamma,
                 alpha_left, alpha_right, alpha_asymmetry, label
    """
    np.random.seed(random_seed)
    rows = []

    for state, profile in STATE_PROFILES.items():
        n = n_samples_per_class
        delta       = _sample_band(profile, "delta", n)
        theta       = _sample_band(profile, "theta", n)
        alpha       = _sample_band(profile, "alpha", n)
        beta        = _sample_band(profile, "beta",  n)
        gamma       = _sample_band(profile, "gamma", n)
        alpha_left  = _sample_band(profile, "alpha_left",  n)
        alpha_right = _sample_band(profile, "alpha_right", n)

        # Normalize each row so band powers sum to 1.0
        # (relative power must be proportional within a row)
        totals = delta + theta + alpha + beta + gamma
        delta  /= totals
        theta  /= totals
        alpha  /= totals
        beta   /= totals
        gamma  /= totals

        # Hemispheric Alpha Asymmetry: positive → right dominant (Ida tendency),
        # negative → left dominant (Pingala tendency).
        asymmetry = alpha_right - alpha_left

        for i in range(n):
            rows.append({
                "delta":           round(delta[i], 6),
                "theta":           round(theta[i], 6),
                "alpha":           round(alpha[i], 6),
                "beta":            round(beta[i], 6),
                "gamma":           round(gamma[i], 6),
                "alpha_left":      round(alpha_left[i], 6),
                "alpha_right":     round(alpha_right[i], 6),
                "alpha_asymmetry": round(asymmetry[i], 6),
                "label":           state,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df


def save_dataset(path: str = "neuro_yogic/mock_eeg_data.csv",
                 n_samples_per_class: int = 500) -> str:
    """
    Generate and save the synthetic dataset to a CSV file.

    Returns the path where the file was saved.
    """
    df = generate_dataset(n_samples_per_class=n_samples_per_class)
    df.to_csv(path, index=False)
    print(f"[DataGenerator] Saved {len(df)} rows to '{path}'")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")
    return path


if __name__ == "__main__":
    save_dataset()
