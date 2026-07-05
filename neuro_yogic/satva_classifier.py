"""
satva_classifier.py  (v2 — paper-aligned rewrite)
==================================================
Classifies each EEG epoch into the three Samkhya / Ayurvedic Gunas:

  Sattva  — clarity, purity, balance
  Rajas   — activity, passion, dynamism
  Tamas   — inertia, heaviness, dullness

Root-cause fix for "always 99% Rajas":
──────────────────────────────────────
The previous version used all-beta (13-30 Hz) as the Rajas driver.
But the Muse S frontal sensors (AF7, AF8) always show elevated frontal
beta — that's normal waking physiology, NOT a Rajasic state.

The paper is specific: Rajas is "desynchronized, low-amplitude HIGH Beta
(18–30 Hz) in the prefrontal areas." The distinction between:
  • Low beta / SMR (13-18 Hz) — calm, focused, Sattvic
  • High beta (18-30 Hz)      — stress/anxiety, Rajasic

...is the single most important fix for accurate classification.

Theory mapping (from paper):
─────────────────────────────
Sattva ↔ High-amplitude synchronized Alpha (8-12 Hz) + Frontal Midline
          Theta (4-7 Hz) + balanced FAA ≈ 0 + high PLV (coherence)
          Neurochemistry: Serotonin, GABA. ANS: Parasympathetic dominance.

Rajas  ↔ High Beta (18-30 Hz) desynchronization + suppressed Alpha
          + positive FAA (left PFC activation). Emotion: restless, driven.
          Neurochemistry: Dopamine, Cortisol. ANS: Sympathetic dominance.

Tamas  ↔ Dominant waking Delta (0.5-4 Hz) + absent Alpha/Gamma
          + low PLV. State: foggy, lethargic, depressed.
          Neurochemistry: Melatonin, Endorphins. ANS: Hypo-arousal.
"""

from typing import Optional


def classify_gunas(
    band_rel: dict,
    faa: float = 0.0,
    plv: float = 0.5,
    chitta_bhumi: Optional[str] = None,
    swara: Optional[str] = None,
) -> dict:
    """
    Classify EEG epoch into Sattva / Rajas / Tamas proportions.

    Parameters
    ----------
    band_rel     : relative band powers — must include "alpha", "theta",
                   "delta", "high_beta", "gamma".  "low_beta" optional.
                   Values should sum to ~1.0 (relative powers).
    faa          : Frontal Alpha Asymmetry = ln(α_right) − ln(α_left)
                   Positive → Pingala/Rajas, Negative → Ida, ≈0 → Sushumna.
    plv          : Phase Locking Value (0-1). High PLV = Sattva/Niruddha.
    chitta_bhumi : optional Chitta Bhumi label for secondary adjustment.
    swara        : optional Swara string for secondary adjustment.

    Returns
    -------
    dict — {"sattva": float, "rajas": float, "tamas": float} summing to 1.0
    """
    # ── Extract features ──────────────────────────────────────────────────────
    alpha    = float(band_rel.get("alpha",    0.0))
    theta    = float(band_rel.get("theta",    0.0))
    delta    = float(band_rel.get("delta",    0.0))
    gamma    = float(band_rel.get("gamma",    0.0))
    low_beta = float(band_rel.get("low_beta", 0.0))

    # high_beta is the PRIMARY Rajas marker — fall back to half of "beta"
    # if the new feature isn't present (backward compatibility)
    beta_total = float(band_rel.get("beta", 0.0))
    high_beta  = float(band_rel.get("high_beta", beta_total * 0.45))

    faa = float(faa)
    plv = float(plv)

    # ── SATTVA score ─────────────────────────────────────────────────────────
    # Primary: Alpha synchrony (paper: "high-amplitude, highly synchronized
    #          Alpha waves 8-12 Hz") — this is the single strongest predictor.
    # Secondary: Frontal Midline Theta (paper: "healthy, organized Fm-θ 4-7 Hz,
    #            reflecting focused inward attention without anxiety").
    # Tertiary: SMR / Low Beta = calm-focused state (slightly Sattvic).
    # Bonus: balanced FAA (|FAA| < 0.15) = Sushumna = Sattva.
    # Bonus: high PLV = interhemispheric coherence = Sattva/Niruddha.
    sat = (
        alpha    * 4.5 +                           # Alpha = primary Sattva driver
        theta    * 2.5 +                           # Frontal Midline Theta
        low_beta * 0.8 +                           # SMR/calm-focus (mild Sattva)
        max(0.0, plv - 0.50) * 2.5 +              # Coherence bonus (peaks at PLV=1)
        max(0.0, 0.20 - abs(faa)) * 1.5           # FAA balance bonus (max at FAA=0)
    )

    # ── RAJAS score ───────────────────────────────────────────────────────────
    # Primary: HIGH beta (18-30 Hz) — the paper's exact specification.
    # "Rajas manifests as desynchronized, low-amplitude, high-frequency EEG
    #  dominated by Beta waves (13–30 Hz), particularly High Beta (18–30 Hz)
    #  in the prefrontal areas."
    # Suppressed Alpha supports Rajas (reduced sensory gating).
    # Positive FAA = left PFC activation = Pingala = Rajasic tendency.
    raj = (
        high_beta * 5.5 +                          # HIGH beta = primary Rajas driver
        max(0.0, 0.20 - alpha) * 2.0 +            # Suppressed alpha supports Rajas
        max(0.0, faa) * 1.8 +                     # Positive FAA = Pingala/Rajas
        max(0.0, gamma - 0.10) * 0.8              # Gamma in agitated (not meditative) state
    )

    # ── TAMAS score ───────────────────────────────────────────────────────────
    # Primary: Waking Delta (paper: "pathological dominance of slow Delta waves
    #          0.5-4 Hz over frontal/central areas in waking state").
    # Tamas is NOT just delta — it requires ABSENCE of Alpha and Gamma too.
    # "Marked absence of synchronized Alpha or high-frequency Gamma oscillations."
    tam = (
        delta    * 4.5 +                           # Waking Delta = primary Tamas driver
        max(0.0, 0.15 - alpha)  * 2.5 +           # Absent alpha supports Tamas
        max(0.0, 0.06 - gamma)  * 2.0 +           # Absent gamma supports Tamas
        max(0.0, 0.45 - plv)    * 1.0             # Low coherence = disorganized
    )

    # ── Chitta Bhumi secondary adjustment (light touch) ───────────────────────
    if chitta_bhumi:
        _CHITTA_ADJ = {
            "Niruddha": ( 0.5, -0.3, -0.3),
            "Ekagra":   ( 0.4, -0.2, -0.2),
            "Vikshipta":( 0.0,  0.1,  0.0),
            "Kshipta":  (-0.2,  0.4, -0.1),
            "Mudha":    (-0.3, -0.2,  0.5),
        }
        if chitta_bhumi in _CHITTA_ADJ:
            ds, dr, dt = _CHITTA_ADJ[chitta_bhumi]
            sat = max(0.0, sat + ds)
            raj = max(0.0, raj + dr)
            tam = max(0.0, tam + dt)

    # ── Swara secondary adjustment (light touch) ──────────────────────────────
    if swara:
        sl = swara.lower()
        if "sushumna" in sl:
            sat = sat + 0.20;  raj = max(0.0, raj - 0.10)
        elif "pingala" in sl:
            raj = raj + 0.15;  sat = max(0.0, sat - 0.05)
        elif "ida" in sl:
            # Ida is parasympathetic — slightly Sattvic but can also be Tamasic
            # if delta is dominant (inward, passive → check overall state)
            if delta > 0.30:
                tam = tam + 0.10
            else:
                sat = sat + 0.08

    # ── Normalise to sum = 1.0 ────────────────────────────────────────────────
    # Use a small floor to prevent division by zero, but keep it tiny so
    # strong states still show high percentages (not diluted).
    sat = max(sat, 0.01)
    raj = max(raj, 0.01)
    tam = max(tam, 0.01)

    total = sat + raj + tam
    return {
        "sattva": round(sat / total, 4),
        "rajas":  round(raj / total, 4),
        "tamas":  round(tam / total, 4),
    }


def gunas_label(gunas: dict) -> str:
    """
    Return a human-readable dominant-Guna label.
    Falls back to "Balanced" when no Guna exceeds 45%.
    """
    dominant = max(gunas, key=lambda k: gunas.get(k, 0))
    value = gunas.get(dominant, 0)
    if value < 0.45:
        return "Balanced"
    return {"sattva": "Sattvic", "rajas": "Rajasic", "tamas": "Tamasic"}.get(dominant, "Balanced")


def gunas_note(gunas: dict) -> str:
    """Interpretive note for the dominant Guna state, aligned with paper."""
    dominant = max(gunas, key=lambda k: gunas.get(k, 0))
    value = gunas.get(dominant, 0)
    if value < 0.45:
        return (
            "The three Gunas are in relative equilibrium — a balanced, "
            "transitional mental state. Ideal for transitions into meditation."
        )
    notes = {
        "sattva": (
            "Sattva predominates — the mind is luminous, calm, and self-regulated. "
            "Alpha synchrony and parasympathetic tone are elevated. "
            "An optimal condition for contemplative practice and insight."
        ),
        "rajas": (
            "Rajas predominates — high-beta (18-30 Hz) prefrontal activity indicates "
            "an active, driven, and outward-directed mental state. "
            "Sympathetic nervous system is engaged. Channel this energy intentionally; "
            "pranayama (Nadi Shodhana) can help balance to Sattva."
        ),
        "tamas": (
            "Tamas predominates — elevated frontal delta in the waking state signals "
            "mental heaviness, cognitive fog, or low arousal. "
            "Stimulating pranayama (Kapalabhati, Bhastrika) and dynamic movement "
            "can help elevate the state toward Sattva."
        ),
    }
    return notes.get(dominant, "")
