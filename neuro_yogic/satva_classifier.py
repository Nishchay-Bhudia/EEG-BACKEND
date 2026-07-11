"""
satva_classifier.py  (v3 — decoupled from Swara/FAA)
==================================================
Classifies each EEG epoch into the three Samkhya / Ayurvedic Gunas:

  Sattva  — clarity, purity, balance (concentration)
  Rajas   — activity, passion, dynamism
  Tamas   — inertia, heaviness, dullness

IMPORTANT — what this actually is
──────────────────────────────────
The genuine, peer-reviewed research on Gunas is questionnaire-based (e.g.
the Vedic Personality Inventory / Gita-based Guna scales) — it measures
trait personality via self-report, not moment-to-moment brain state via
EEG. There is no established EEG signature of "Rajas" the way there is,
say, for sleep spindles. What follows is an inferential heuristic mapping
band-power/coherence markers of arousal (Rajas), disorganization (Tamas),
and calm-focus (Sattva) onto these three labels — useful for biofeedback,
not a validated clinical measure. Treat gunas_note()'s language as
descriptive framing, not a diagnostic claim.

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

Decoupled from Swara/FAA (v3 fix)
───────────────────────────────────
v2 tied Sattva/Rajas scoring to Frontal Alpha Asymmetry and added a
"Swara secondary adjustment" that pushed scores based on the Ida/Pingala/
Sushumna nadi label. That conflates two independent constructs: Gunas
here track activity level (Rajas), inertia (Tamas), and clarity/
concentration (Sattva) — none of which is established to correlate with
*which hemisphere* is more active (that's what FAA/Swara measures).
Someone can be calmly concentrated (Sattva) with either a left- or
right-lateralized FAA. This version removes both the FAA terms and the
Swara adjustment block; Gunas are now driven purely by band-power and PLV
(coherence, a genuinely independent measure of organized vs. disorganized
processing — not a lateralization measure, so it stays).

Theory mapping (from paper, band-power/coherence only):
─────────────────────────────────────────────────────────
Sattva ↔ High-amplitude synchronized Alpha (8-12 Hz) + Frontal Midline
          Theta (4-7 Hz) + high PLV (coherence, i.e. concentration).
          Neurochemistry: Serotonin, GABA. ANS: Parasympathetic dominance.

Rajas  ↔ High Beta (18-30 Hz) desynchronization + suppressed Alpha.
          Emotion: restless, driven, active.
          Neurochemistry: Dopamine, Cortisol. ANS: Sympathetic dominance.

Tamas  ↔ Dominant waking Delta (0.5-4 Hz) + absent Alpha/Gamma
          + low PLV (disorganized). State: foggy, lethargic, inert.
          Neurochemistry: Melatonin, Endorphins. ANS: Hypo-arousal.
"""

from typing import Optional


def classify_gunas(
    band_rel: dict,
    plv: float = 0.5,
    chitta_bhumi: Optional[str] = None,
) -> dict:
    """
    Classify EEG epoch into Sattva / Rajas / Tamas proportions.

    Parameters
    ----------
    band_rel     : relative band powers — must include "alpha", "theta",
                   "delta", "high_beta", "gamma".  "low_beta" optional.
                   Values should sum to ~1.0 (relative powers).
    plv          : Phase Locking Value (0-1). High PLV = Sattva (organized);
                   low PLV = Tamas (disorganized). This is an independent
                   coherence measure, not hemispheric lateralization, so it
                   is NOT the same thing as FAA/Swara — see module docstring.
    chitta_bhumi : optional Chitta Bhumi label for secondary adjustment.

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

    plv = float(plv)

    # ── SATTVA score ─────────────────────────────────────────────────────────
    # Primary: Alpha synchrony (paper: "high-amplitude, highly synchronized
    #          Alpha waves 8-12 Hz") — this is the single strongest predictor.
    # Secondary: Frontal Midline Theta (paper: "healthy, organized Fm-θ 4-7 Hz,
    #            reflecting focused inward attention without anxiety").
    # Tertiary: SMR / Low Beta = calm-focused state (slightly Sattvic).
    # Bonus: high PLV = interhemispheric coherence = organized/concentrated.
    sat = (
        alpha    * 4.5 +                           # Alpha = primary Sattva driver
        theta    * 2.5 +                           # Frontal Midline Theta
        low_beta * 0.8 +                           # SMR/calm-focus (mild Sattva)
        max(0.0, plv - 0.50) * 2.5                # Coherence bonus (peaks at PLV=1)
    )

    # ── RAJAS score ───────────────────────────────────────────────────────────
    # Primary: HIGH beta (18-30 Hz) — the paper's exact specification.
    # "Rajas manifests as desynchronized, low-amplitude, high-frequency EEG
    #  dominated by Beta waves (13–30 Hz), particularly High Beta (18–30 Hz)
    #  in the prefrontal areas."
    # Suppressed Alpha supports Rajas (reduced sensory gating).
    raj = (
        high_beta * 5.5 +                          # HIGH beta = primary Rajas driver
        max(0.0, 0.20 - alpha) * 2.0 +            # Suppressed alpha supports Rajas
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
    # Kept: Patanjali/Vyasa's own framework ties Guna dominance to Bhumi
    # progression (e.g. Niruddha = Gunatita/Sattva-transcendent), so this is
    # a within-tradition correlation, unlike the removed Swara coupling.
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
