"""
vedantic_logic.py — Vedantic Logic Module
==========================================
Maps extracted EEG features to two classical Yogic/Vedantic frameworks:

  1. SWARA RHYTHMS  (Svara Shastra / Swara Yoga)
     ─────────────────────────────────────────────
     The Swara is the dominant breath/life-force channel:
       - Ida    (lunar / left nadi)  — parasympathetic, receptive, right-brain
       - Pingala (solar / right nadi) — sympathetic, active, left-brain
       - Sushumna (central / balanced) — both nadis equal; gateway to higher states

     In modern neuroscience terms, frontal Alpha asymmetry correlates
     with autonomic and hemispheric dominance:
       - Right frontal Alpha ↑  → left-hemisphere INHIBITED → right-brain active
         → parasympathetic bias → Ida tendency
       - Left frontal Alpha ↑   → right-hemisphere INHIBITED → left-brain active
         → sympathetic bias → Pingala tendency
       - Balanced               → Sushumna

     Detection thresholds (conservative, tunable via `asymmetry_threshold`):
       |asymmetry| > 10% of mean hemispheric Alpha → definite Swara state
       |asymmetry| ≤ 10%                            → Sushumna (balanced)

  2. TATTVA / CHAKRA CORRELATES  (Samkhya-Yoga elemental model)
     ─────────────────────────────────────────────────────────────
     The Tattvas are the 25 principles of manifestation in Samkhya.
     This module implements a hypothesis-generating layer — not a
     definitive mapping — to flag EEG signatures worth investigating:

       Gamma surge (>30 Hz, relative power > 12%)
         → "Potential Tattva Activation" — high-frequency neural binding
           events that may correspond to moments of perceptual clarity,
           insight (Prajna), or Spanda (the first tremor of conscious
           manifestation in Tantric frameworks).

       High Theta + low Beta (Niruddha-adjacent signature)
         → "Pratyahara Window" — sensory withdrawal phase; the nervous
           system quiets its environmental scanning (dorsal attention
           network suppression).

       Delta surge with retained Theta (non-sleep context)
         → "Turiya Approach" — the fourth state beyond Jagrat/Swapna/Sushupti;
           extremely rare in short sessions, flagged as an informational note.

     Important: These correlates are *hypotheses*, not confirmed facts.
     They give practitioners a starting vocabulary for first-person
     reporting and iterative model improvement.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ── Swara thresholds ──────────────────────────────────────────────────
# How much stronger one hemisphere's Alpha must be (as a fraction of
# the mean hemispheric Alpha) to declare Ida or Pingala dominance.
DEFAULT_ASYMMETRY_THRESHOLD = 0.10  # 10%

# ── Tattva / Chakra detection thresholds ─────────────────────────────
GAMMA_SPIKE_THRESHOLD       = 0.12  # Gamma > 12% of total power
THETA_HIGH_THRESHOLD        = 0.25  # Theta > 25% of total power
BETA_LOW_THRESHOLD          = 0.10  # Beta  < 10% of total power
DELTA_SURGE_THRESHOLD       = 0.30  # Delta > 30% of total power


@dataclass
class VedanticReading:
    """Structured output from the Vedantic logic layer."""

    # Swara (breath/nadi channel)
    swara:            str = "Sushumna (Balanced)"
    swara_confidence: str = "Low"    # 'Low', 'Moderate', 'High'
    swara_note:       str = ""

    # Tattva / Chakra correlates — zero or more flags
    tattva_flags:     List[str] = field(default_factory=list)

    # Yoga Sutra depth indicator (consistent with Chitta Bhumi for cross-check)
    contemplative_depth: str = "Surface"   # 'Surface', 'Emerging', 'Deep', 'Profound'

    def to_dict(self) -> dict:
        return {
            "swara": {
                "state":      self.swara,
                "confidence": self.swara_confidence,
                "note":       self.swara_note,
            },
            "tattva_flags":       self.tattva_flags,
            "contemplative_depth": self.contemplative_depth,
        }


def classify_swara(
    alpha_left:          float,
    alpha_right:         float,
    asymmetry_threshold: float = DEFAULT_ASYMMETRY_THRESHOLD,
) -> tuple:
    """
    Determine the active Swara (nadi) from hemispheric Alpha asymmetry.

    The asymmetry ratio is computed relative to the mean hemispheric Alpha
    (not the absolute difference) so that the threshold scales correctly
    across different signal amplitudes.

    Parameters
    ----------
    alpha_left  : float — Alpha absolute power, left-hemisphere electrodes
    alpha_right : float — Alpha absolute power, right-hemisphere electrodes
    asymmetry_threshold : float — fractional threshold for dominance detection

    Returns
    -------
    (swara_name, confidence, note)  — all strings
    """
    mean_alpha = (alpha_left + alpha_right) / 2.0

    if mean_alpha < 1e-12:
        # No meaningful Alpha signal — cannot determine Swara
        return (
            "Sushumna (Balanced)",
            "Low",
            "Insufficient Alpha power to determine Swara; measurement may be artefact."
        )

    # Asymmetry as a fraction of mean hemispheric Alpha
    asymmetry_ratio = (alpha_right - alpha_left) / mean_alpha

    if asymmetry_ratio > asymmetry_threshold:
        # Right hemisphere Alpha dominates → left brain inhibited →
        # parasympathetic / receptive → Ida (lunar nadi)
        magnitude = abs(asymmetry_ratio)
        confidence = (
            "High"     if magnitude > 0.25 else
            "Moderate" if magnitude > 0.10 else
            "Low"
        )
        return (
            "Ida (Parasympathetic / Lunar)",
            confidence,
            f"Right-hemisphere Alpha is {magnitude*100:.1f}% stronger than left. "
            "Ida nadi suggests parasympathetic dominance: restful, receptive, "
            "ideal for Yin practices, Yoga Nidra, and deep contemplation."
        )

    elif asymmetry_ratio < -asymmetry_threshold:
        # Left hemisphere Alpha dominates → right brain inhibited →
        # sympathetic / active → Pingala (solar nadi)
        magnitude = abs(asymmetry_ratio)
        confidence = (
            "High"     if magnitude > 0.25 else
            "Moderate" if magnitude > 0.10 else
            "Low"
        )
        return (
            "Pingala (Sympathetic / Solar)",
            confidence,
            f"Left-hemisphere Alpha is {magnitude*100:.1f}% stronger than right. "
            "Pingala nadi suggests sympathetic dominance: energised, analytical, "
            "ideal for Pranayama, dynamic Asana, and cognitive tasks."
        )

    else:
        # Balanced within the threshold → central channel open
        magnitude = abs(asymmetry_ratio)
        confidence = "High" if magnitude < 0.03 else "Moderate"
        return (
            "Sushumna (Balanced / Central)",
            confidence,
            f"Hemispheric Alpha balance within {magnitude*100:.1f}% of threshold. "
            "Sushumna suggests both nadis are in equilibrium: "
            "the optimal window for deep meditation and Kundalini practices."
        )


def detect_tattva_flags(band_relative: dict, gamma_spike: bool) -> List[str]:
    """
    Scan extracted band powers for Tattva/Chakra correlate flags.

    Parameters
    ----------
    band_relative : dict  — {band_name: relative_power, …} (values sum ≈ 1)
    gamma_spike   : bool  — pre-computed flag from FeatureExtractor

    Returns
    -------
    list of str — zero or more descriptive flag strings
    """
    flags = []

    delta = band_relative.get("delta", 0.0)
    theta = band_relative.get("theta", 0.0)
    beta  = band_relative.get("beta",  0.0)
    gamma = band_relative.get("gamma", 0.0)

    # ── Flag 1: Tattva Activation (Gamma surge) ───────────────────────
    # High-frequency gamma bursts (30–50 Hz) have been observed during
    # reports of insight, Samadhi-adjacent states, and coherent binding
    # across distributed cortical networks. In Tantric Yoga this is
    # sometimes called "Spanda" — the primordial throb of consciousness.
    if gamma_spike or gamma > GAMMA_SPIKE_THRESHOLD:
        flags.append(
            "Potential Tattva Activation — Gamma burst detected (>30 Hz). "
            "Associated with cortical binding / insight events (Spanda). "
            "Cross-reference with first-person report."
        )

    # ── Flag 2: Pratyahara Window (high Theta + low Beta) ────────────
    # Pratyahara (withdrawal of senses, Yoga Sutra 2.54) is characterised
    # by decreased responsiveness to external stimuli. Neurally, this
    # correlates with Theta dominance (thalamo-cortical gating) and
    # suppressed Beta (reduced sensorimotor/attentional processing).
    if theta > THETA_HIGH_THRESHOLD and beta < BETA_LOW_THRESHOLD:
        flags.append(
            "Pratyahara Window — Elevated Theta + suppressed Beta. "
            "Sensory withdrawal phase active; thalamic gating engaged. "
            "Optimal moment for inward Dharana (concentration practice)."
        )

    # ── Flag 3: Turiya Approach (Delta surge with retained Theta) ─────
    # Turiya, the 'fourth state', is described in the Mandukya Upanishad
    # as pure witness-consciousness underlying Jagrat/Swapna/Sushupti.
    # A non-sleep Delta surge combined with maintained Theta suggests
    # the meditator is on the threshold — rare and significant.
    if delta > DELTA_SURGE_THRESHOLD and theta > 0.20:
        flags.append(
            "Turiya Approach — Delta surge with retained Theta in waking context. "
            "Extremely rare; may indicate the onset of formless absorption. "
            "Verify practitioner is conscious and not drifting into sleep."
        )

    return flags


def assess_contemplative_depth(chitta_bhumi: Optional[str], band_relative: dict) -> str:
    """
    Translate the classifier's Chitta Bhumi label into a practitioner-facing
    depth descriptor, cross-validated by the raw band signature.

    Returns one of: 'Surface', 'Emerging', 'Deep', 'Profound'
    """
    theta = band_relative.get("theta", 0.0)
    delta = band_relative.get("delta", 0.0)
    alpha = band_relative.get("alpha", 0.0)
    beta  = band_relative.get("beta",  0.0)

    depth_map = {
        "Kshipta":   "Surface",
        "Vikshipta": "Emerging",
        "Ekagra":    "Deep",
        "Niruddha":  "Profound",
    }

    base = depth_map.get(chitta_bhumi or "", "Surface")

    # Upgrade/downgrade based on raw signature as a sanity check
    if base == "Profound" and beta > 0.20:
        # High Beta contradicts Niruddha — likely a transitional moment
        base = "Deep"
    if base == "Surface" and alpha > 0.35:
        # Strong Alpha with Kshipta label → classifier may be borderline
        base = "Emerging"

    return base


def analyze(
    info: dict,
    chitta_bhumi: Optional[str] = None,
    asymmetry_threshold: float = DEFAULT_ASYMMETRY_THRESHOLD,
) -> VedanticReading:
    """
    Main entry point: produce a VedanticReading from FeatureExtractor output.

    Parameters
    ----------
    info : dict
        The `info` dict returned by FeatureExtractor.extract():
        must contain 'band_relative', 'alpha_left', 'alpha_right', 'gamma_spike'.
    chitta_bhumi : str, optional
        Classifier prediction; used to compute contemplative depth.
    asymmetry_threshold : float
        Swara detection sensitivity (default 10%).

    Returns
    -------
    VedanticReading
    """
    band_rel     = info.get("band_relative", {})
    alpha_left   = info.get("alpha_left",    0.0)
    alpha_right  = info.get("alpha_right",   0.0)
    gamma_spike  = info.get("gamma_spike",   False)

    swara, confidence, note = classify_swara(
        alpha_left, alpha_right, asymmetry_threshold
    )
    tattva_flags = detect_tattva_flags(band_rel, gamma_spike)
    depth        = assess_contemplative_depth(chitta_bhumi, band_rel)

    return VedanticReading(
        swara            = swara,
        swara_confidence = confidence,
        swara_note       = note,
        tattva_flags     = tattva_flags,
        contemplative_depth = depth,
    )


if __name__ == "__main__":
    # Quick demo with hand-crafted Ekagra-like feature info
    demo_info = {
        "band_relative": {
            "delta": 0.15, "theta": 0.22, "alpha": 0.42,
            "beta": 0.16,  "gamma": 0.05,
        },
        "alpha_left":    0.18,
        "alpha_right":   0.28,   # right dominant → Ida
        "gamma_spike":   False,
    }
    reading = analyze(demo_info, chitta_bhumi="Ekagra")
    import json
    print(json.dumps(reading.to_dict(), indent=2))
