"""
yoga_classifier.py  (v2 — rule-based, paper-aligned)
======================================================
Classifies each EEG epoch into the five Chitta Bhumis of Patanjali's
Yoga Sutras, using rule-based fuzzy scoring derived directly from the paper:

"Electroencephalographic Mapping of Yogic Physiology"

Why rule-based instead of Random Forest?
─────────────────────────────────────────
1. The paper provides EXACT EEG thresholds for each state. Synthetic training
   data can never capture the true distribution of real Muse S signals.
2. Rule-based is transparent: if the classification is wrong, you can see
   exactly which rule to adjust.
3. No startup training latency — the backend responds instantly.
4. Adds MUDHA (the 5th Chitta Bhumi) which was missing from v1.

The five Chitta Bhumis (progression from lowest to highest):
─────────────────────────────────────────────────────────────
Mudha    — Dull/Torpid.   Tamas-dominant.  Waking delta surge, absent alpha/gamma.
Kshipta  — Scattered.     Rajas-dominant.  High beta (18-30 Hz), suppressed alpha.
Vikshipta— Oscillating.   Sattva-emerging. Moderate alpha alternating with beta.
Ekagra   — One-Pointed.   Pure Sattva.     Sustained Fm-θ + high alpha synchrony.
Niruddha — Mastered.      Gunatita.        Global gamma coherence (PLV > 0.80).

Scoring logic:
──────────────
Each state receives a fuzzy score based on EEG feature thresholds from the
paper. Scores are normalised to probabilities. The highest-scoring state wins.

Personal baseline z-scoring (root-cause fix for the trait/state confound)
───────────────────────────────────────────────────────────────────────────
Fixed thresholds like "gamma > 0.15 = Niruddha" assume every subject's
resting band power sits at the same population baseline. It doesn't — Cahn &
Polich (2006) specifically flag that elevated gamma in experienced
meditators is often a *trait* effect of long-term practice, not a
within-session *state* change. Someone whose resting gamma naturally sits
at 15% shouldn't register as "Niruddha" just for existing.
`classify_from_info` accepts an optional `baseline` dict (per-band
{"mean": .., "std": ..}, computed by the caller from a short resting
calibration period) and re-centers each band's power on the *population*
reference before scoring: a value that is 2 std above someone's own
baseline reads the same as 2 std above the population baseline, regardless
of where that person's resting level actually sits. No baseline supplied →
identical behaviour to before (fully backward compatible).

Temporal smoothing
───────────────────
Each 2 s epoch was previously classified in total isolation — no memory of
the previous epoch — so single noisy epochs could flip the displayed state
every couple of seconds. `classify_from_info` accepts an optional
`recent_probs` list (the caller's last few epochs' probability dicts,
oldest→newest) and blends them into the current epoch via an exponential
moving average before picking a winner, damping single-epoch noise while
still tracking genuine state changes within a few epochs.
"""

import os
import numpy as np
from typing import List, Optional, Tuple

# ── Class definitions ─────────────────────────────────────────────────────────
CHITTA_BHUMIS = ["Mudha", "Kshipta", "Vikshipta", "Ekagra", "Niruddha"]

DEFAULT_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "yoga_rf.joblib")
DEFAULT_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yoga_labels.npy")
FEATURE_COLUMNS     = [
    "delta", "theta", "alpha", "low_beta", "high_beta", "gamma",
    "alpha_left", "alpha_right", "faa", "plv",
]

# ── Population reference (heuristic priors, not empirically fit) ─────────────
# These match the existing fallback defaults already used below (br.get(..,
# default)) — reused here as the "population mean" that a personal baseline
# gets re-centered onto. Std is a rough 35% coefficient of variation, a
# commonly cited ballpark for relative EEG band-power variability; there is
# no dataset behind these specific numbers, so treat them as a documented
# heuristic, not a calibrated constant.
_POPULATION_BAND_MEANS = {
    "delta": 0.15, "theta": 0.18, "alpha": 0.28,
    "low_beta": 0.18, "high_beta": 0.13, "gamma": 0.08,
}
_POPULATION_BAND_STDS = {k: max(v * 0.35, 0.02) for k, v in _POPULATION_BAND_MEANS.items()}

# ── Temporal smoothing ────────────────────────────────────────────────────────
SMOOTHING_DECAY = 0.6   # per-step-back discount applied to older epochs


def _apply_baseline(band_rel: dict, baseline: Optional[dict]) -> dict:
    """
    Re-center each band's relative power on the population mean, scaled by
    this person's own deviation from their baseline (z-score). See module
    docstring. Returns band_rel unchanged when no baseline is supplied.
    """
    if not baseline:
        return dict(band_rel)

    adjusted = dict(band_rel)
    for band, mean_pop in _POPULATION_BAND_MEANS.items():
        stats = baseline.get(band)
        if not stats or band not in band_rel:
            continue
        try:
            personal_mean = float(stats.get("mean", band_rel[band]))
            personal_std  = float(stats.get("std", 0.0)) or 1e-6
        except (TypeError, ValueError):
            continue
        z = (float(band_rel[band]) - personal_mean) / personal_std
        std_pop = _POPULATION_BAND_STDS.get(band, personal_std)
        adjusted[band] = max(0.0, mean_pop + z * std_pop)
    return adjusted


def _smooth_probs(current: dict, recent: Optional[List[dict]]) -> dict:
    """
    Exponential moving average across recent epochs' probability
    distributions (oldest→newest), so a single noisy epoch can't flip the
    displayed Chitta Bhumi on its own. Returns `current` unchanged when no
    history is supplied.
    """
    if not recent:
        return dict(current)

    blended = dict(current)
    weight = 1.0
    weight_total = 1.0
    for probs in reversed(recent):
        weight *= SMOOTHING_DECAY
        if weight < 1e-3:
            break
        for k, v in probs.items():
            try:
                blended[k] = blended.get(k, 0.0) + weight * float(v)
            except (TypeError, ValueError):
                continue
        weight_total += weight
    return {k: v / weight_total for k, v in blended.items()}


# ── Core rule-based classifier ────────────────────────────────────────────────

def _rule_classify(
    info: dict,
    baseline: Optional[dict] = None,
    recent_probs: Optional[List[dict]] = None,
) -> Tuple[str, dict]:
    """
    Paper-derived fuzzy rule classifier.

    Takes the full `info` dict from FeatureExtractor (or a compatible dict
    from the /analyze/bands endpoint) and returns (label, probability_dict).

    baseline     : optional per-band {"mean", "std"} from the caller's own
                   resting calibration — see _apply_baseline.
    recent_probs : optional list of this session's last few probability
                   dicts (oldest→newest) — see _smooth_probs.
    """
    br = _apply_baseline(info.get("band_relative", {}), baseline)

    delta    = float(br.get("delta",     0.15))
    theta    = float(br.get("theta",     0.18))
    alpha    = float(br.get("alpha",     0.28))
    gamma    = float(br.get("gamma",     0.08))
    high_beta= float(br.get("high_beta", 0.13))
    low_beta = float(br.get("low_beta",  0.18))

    # Combine into total beta for any legacy callers
    beta_total = high_beta + low_beta

    faa = float(info.get("faa", 0.0))
    plv = float(info.get("plv", 0.50))

    scores = {}

    # ── NIRUDDHA (Mastered) ────────────────────────────────────────────────
    # Paper: "high-amplitude, globally coherent Gamma (30-100 Hz) with massive
    #         bilateral interhemispheric phase coherence. Slower frequencies
    #         (delta, alpha) drop to baseline idling amplitudes."
    # Key markers: gamma > 0.15, PLV > 0.80, minimal high-beta, minimal delta.
    # Thresholds exact from paper: gamma > 0.15, PLV > 0.80.
    scores["Niruddha"] = (
        max(0.0, (gamma    - 0.15) * 6.0) +       # gamma > 0.15 = paper threshold
        max(0.0, (plv      - 0.80) * 5.0) +       # PLV > 0.80 = paper threshold
        max(0.0, (0.08 - high_beta) * 2.0) +      # Suppressed high beta
        max(0.0, (0.08 - delta)    * 2.0) +       # Minimal delta
        max(0.0, (0.05 - abs(faa)) * 1.5) +       # Perfectly balanced FAA
        max(0.0, (alpha    - 0.20) * 0.5)         # Some alpha (not absent)
    )

    # ── EKAGRA (One-Pointed) ───────────────────────────────────────────────
    # Paper: "sustained, exceptionally high-amplitude Frontal Midline Theta
    #         (Fm-θ, 4-8 Hz) coupled with massive, synchronized, global
    #         Alpha (8-12 Hz). Beta and Delta suppressed to minimum."
    # Key markers: theta > 0.25, alpha > 0.25, high_beta < 0.10.
    # Thresholds from paper: Fm-θ > 0.25, alpha > 0.25.
    scores["Ekagra"] = (
        max(0.0, (theta    - 0.22) * 5.0) +       # theta > 0.22 (paper: ~0.25 target)
        max(0.0, (alpha    - 0.25) * 5.0) +       # alpha > 0.25 = paper threshold
        max(0.0, (0.10 - high_beta) * 3.0) +      # Suppressed high beta
        max(0.0, (0.10 - delta)    * 1.5) +       # Suppressed delta
        max(0.0, (plv      - 0.60) * 1.5) +       # Moderate-high coherence
        max(0.0, (0.15 - abs(faa)) * 1.0)         # Balanced FAA
    )

    # ── VIKSHIPTA (Oscillating) ────────────────────────────────────────────
    # Paper: "rapid, unstable oscillations between different frequency bands.
    #         Short bursts of high-amplitude Alpha suddenly interrupted by
    #         High Beta desynchronization when the mind wanders."
    # Key markers: moderate alpha (0.12-0.30), some beta present, transitional.
    # This is the DEFAULT state for most people — it has a baseline score.
    alpha_moderate = 1.0 - abs(alpha - 0.22) / 0.22         # peaks at alpha=0.22
    scores["Vikshipta"] = (
        max(0.0, alpha_moderate * 3.0) +                     # Moderate alpha
        max(0.0, alpha * 1.5) +                              # Some alpha present
        max(0.0, (low_beta - 0.10) * 1.0) +                 # Some beta active
        max(0.0, (0.25 - high_beta) * 1.0) +                # Not dominated by high beta
        0.8                                                  # Baseline (common state)
    )

    # ── KSHIPTA (Scattered) ────────────────────────────────────────────────
    # Paper: "persistent, high-amplitude, desynchronized High Beta (18-30 Hz)
    #         across both frontal sensors. Alpha-band power is significantly
    #         diminished, reflecting inability to self-regulate."
    # Key markers: high_beta > 0.20, alpha < 0.15.
    scores["Kshipta"] = (
        max(0.0, (high_beta - 0.15) * 5.0) +      # High beta primary signal
        max(0.0, (0.15 - alpha)     * 3.0) +      # Suppressed alpha
        max(0.0, faa * 1.5) +                     # Positive FAA (left PFC arousal)
        max(0.0, (0.40 - plv)       * 1.0) +      # Low coherence (scattered)
        max(0.0, (beta_total - 0.25) * 1.0)       # Total beta elevated
    )

    # ── MUDHA (Dull/Torpid) ────────────────────────────────────────────────
    # Paper: "waking EEG reveals pathological dominance of slow Delta waves
    #         (0.5-4 Hz) and sluggish, low-frequency Theta (4-6 Hz) over
    #         frontal/central areas, with COMPLETE ABSENCE of Alpha or Gamma."
    # Key markers: delta > 0.40, alpha < 0.10, gamma < 0.05 (paper-exact).
    # Threshold anchor at 0.40 as cited in paper for waking delta dominance.
    scores["Mudha"] = (
        max(0.0, (delta    - 0.40) * 7.0) +       # delta > 0.40 = paper threshold
        max(0.0, (delta    - 0.30) * 2.0) +       # Softer ramp below 0.40 for graceful scoring
        max(0.0, (0.10 - alpha)    * 3.5) +       # Absent alpha (paper: "complete absence")
        max(0.0, (0.05 - gamma)    * 2.5) +       # Absent gamma (paper: "complete absence")
        max(0.0, (0.45 - plv)      * 1.0) +       # Disorganized (low PLV)
        max(0.0, (0.10 - high_beta) * 0.5)        # Not agitated (Tamas ≠ Rajas)
    )

    # ── Normalise to probabilities ─────────────────────────────────────────
    total = sum(scores.values()) or 1e-10
    probs = {k: round(v / total, 4) for k, v in scores.items()}

    # ── Temporal smoothing across recent epochs ────────────────────────────
    # Blend with the caller's recent-epoch history so one noisy 2s epoch
    # can't flip the displayed state on its own. Winner is picked from the
    # smoothed distribution, not the raw single-epoch one.
    smoothed = _smooth_probs(probs, recent_probs)
    smoothed = {k: round(v, 4) for k, v in smoothed.items()}

    label = max(smoothed, key=smoothed.get)
    return label, smoothed


# ── Public classifier class (API-compatible with v1) ─────────────────────────

class YogaClassifier:
    """
    Rule-based Chitta Bhumi classifier.

    Keeps the same public API as v1 (predict, predict_proba, save, load,
    train_model) so main.py requires minimal changes. The classify_from_info
    method is the recommended interface — it uses the full info dict.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        # Rule-based: always "trained"
        self._is_trained = True

    # ── Recommended interface ──────────────────────────────────────────────────
    def classify_from_info(
        self,
        info: dict,
        baseline: Optional[dict] = None,
        recent_probs: Optional[List[dict]] = None,
    ) -> Tuple[str, dict]:
        """
        Classify using the full info dict from FeatureExtractor.
        This is the primary interface — no feature array gymnastics required.

        baseline     : optional personal calibration — see _apply_baseline.
        recent_probs : optional recent-epoch history — see _smooth_probs.

        Returns (chitta_bhumi_label, probability_dict).
        """
        return _rule_classify(info, baseline=baseline, recent_probs=recent_probs)

    # ── Legacy interface (backward-compatible with main.py v1) ───────────────
    def predict(self, features: np.ndarray) -> str:
        """Accept a 10-D (or 8-D legacy) feature vector. Returns label."""
        info = _features_to_info(features)
        return _rule_classify(info)[0]

    def predict_proba(self, features: np.ndarray) -> dict:
        """Return {label: probability} dict for a feature vector."""
        info = _features_to_info(features)
        _, probs = _rule_classify(info)
        return probs

    # ── Training / persistence stubs (no-op for rule-based) ──────────────────
    def train_model(self, csv_path: str) -> dict:
        """No-op: rule-based classifier doesn't train. Returns mock metrics."""
        return {
            "accuracy": 1.0,
            "method":   "rule-based (paper-derived thresholds)",
            "classes":  CHITTA_BHUMIS,
            "note":     "No training required. Classification uses paper-derived fuzzy rules.",
        }

    def save(self, model_path: str = DEFAULT_MODEL_PATH,
             labels_path: str = DEFAULT_LABELS_PATH) -> None:
        """No-op: nothing to save."""
        pass

    def load(self, model_path: str = DEFAULT_MODEL_PATH,
             labels_path: str = DEFAULT_LABELS_PATH) -> None:
        """No-op: nothing to load."""
        pass

    def is_trained(self) -> bool:
        return True


def _features_to_info(features: np.ndarray) -> dict:
    """
    Convert a raw feature array to an info dict for _rule_classify.
    Handles both 10-D (new) and 8-D (legacy) feature vectors.
    """
    f = list(features)
    if len(f) >= 10:
        # New 10-D: [delta, theta, alpha, low_beta, high_beta, gamma, al, ar, faa, plv]
        return {
            "band_relative": {
                "delta":     float(f[0]),
                "theta":     float(f[1]),
                "alpha":     float(f[2]),
                "low_beta":  float(f[3]),
                "high_beta": float(f[4]),
                "gamma":     float(f[5]),
            },
            "faa": float(f[8]),
            "plv": float(f[9]),
        }
    else:
        # Legacy 8-D: [delta, theta, alpha, beta, gamma, al_rel, ar_rel, asym]
        beta = float(f[3]) if len(f) > 3 else 0.0
        return {
            "band_relative": {
                "delta":     float(f[0]) if len(f) > 0 else 0.15,
                "theta":     float(f[1]) if len(f) > 1 else 0.18,
                "alpha":     float(f[2]) if len(f) > 2 else 0.28,
                "low_beta":  beta * 0.55,
                "high_beta": beta * 0.45,
                "gamma":     float(f[4]) if len(f) > 4 else 0.08,
            },
            "faa": 0.0,
            "plv": 0.50,
        }
