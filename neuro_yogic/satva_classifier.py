"""
satva_classifier.py
===================
Classifies each EEG epoch into the three Ayurvedic / Samkhya Gunas:

  Sattva  — clarity, purity, balance (high Alpha, Ekagra/Niruddha, Sushumna)
  Rajas   — activity, passion, motion (high Beta/Gamma, Pingala, Kshipta/Vikshipta)
  Tamas   — inertia, dullness, heaviness (high Delta, low Alpha, suppressed arousal)

The classifier is heuristic-based and designed to complement the existing
Chitta Bhumi (yoga_classifier.py) and Swara (vedantic_logic.py) classifiers.

Output: {satva: float, rajas: float, tamas: float}  — values sum to 1.0

Theory mapping:
  Sattva  ↔ Alpha dominance + meditative depth (Ekagra / Niruddha) + Sushumna
  Rajas   ↔ Beta/Gamma dominance + active attention (Kshipta) + Pingala
  Tamas   ↔ Delta dominance + suppressed Theta/Alpha + heavy Ida (when non-meditative)
"""

from typing import Optional


# ── Guna weight tables ────────────────────────────────────────────────────────

# Each band contributes positive or negative weight per Guna.
# Values are tuned so that typical resting-state EEG (Alpha ~35%, Theta ~20%,
# Beta ~25%, Delta ~12%, Gamma ~8%) produces roughly equal Guna split,
# and strong meditative EEG (Alpha >40%, Theta >30%) produces clear Sattva.

_BAND_WEIGHTS = {
    #           sattva   rajas   tamas
    "delta":   (-0.5,   -0.5,    3.0),
    "theta":   ( 1.5,   -0.5,   -0.5),
    "alpha":   ( 3.0,   -1.5,   -1.5),
    "beta":    (-1.5,    3.0,   -0.5),
    "gamma":   (-0.5,    2.5,   -0.5),
}

_CHITTA_BONUS = {
    # (sattva_bonus, rajas_bonus, tamas_bonus)
    "Niruddha":   ( 2.0, -1.0, -1.0),
    "Ekagra":     ( 1.5, -0.5, -1.0),
    "Vikshipta":  ( 0.0,  1.0, -0.5),
    "Kshipta":    (-0.5,  2.0, -0.5),
}

_SWARA_BONUS = {
    # swara string -> (sattva, rajas, tamas) bonus
    "sushumna": ( 1.5, -0.5, -0.5),
    "ida":      ( 0.0, -0.5,  1.0),
    "pingala":  (-0.5,  1.5, -0.5),
}


def _softplus(x: float, floor: float = 0.02) -> float:
    """Ensure all Guna raw scores are positive before normalisation."""
    return max(x, floor)


def classify_gunas(
    band_rel: dict,
    chitta_bhumi: Optional[str] = None,
    swara: Optional[str] = None,
) -> dict:
    """
    Classify EEG epoch into Sattva / Rajas / Tamas proportions.

    Parameters
    ----------
    band_rel     : dict — relative band powers, e.g. {"delta": 0.12, "theta": 0.22, ...}
                   Values should sum to ~1.0 (relative powers from FeatureExtractor).
    chitta_bhumi : str or None — Chitta Bhumi label from YogaClassifier
                   ("Kshipta" | "Vikshipta" | "Ekagra" | "Niruddha")
    swara        : str or None — Swara state string from vedantic_logic
                   (should contain "Ida", "Pingala", or "Sushumna")

    Returns
    -------
    dict — {"sattva": float, "rajas": float, "tamas": float}
            values in [0, 1] that sum to 1.0, rounded to 4 dp
    """
    # ── Step 1: Band-power contribution ──────────────────────────────────────
    sat_raw = 0.0
    raj_raw = 0.0
    tam_raw = 0.0

    for band, (ws, wr, wt) in _BAND_WEIGHTS.items():
        power = float(band_rel.get(band, 0.0))
        sat_raw += ws * power
        raj_raw += wr * power
        tam_raw += wt * power

    # ── Step 2: Chitta Bhumi bonus ────────────────────────────────────────────
    if chitta_bhumi and chitta_bhumi in _CHITTA_BONUS:
        bs, br, bt = _CHITTA_BONUS[chitta_bhumi]
        # Scale bonus by a factor so it's significant but not overwhelming
        sat_raw += bs * 0.3
        raj_raw += br * 0.3
        tam_raw += bt * 0.3

    # ── Step 3: Swara bonus ───────────────────────────────────────────────────
    if swara:
        swara_lower = swara.lower()
        if "sushumna" in swara_lower:
            key = "sushumna"
        elif "pingala" in swara_lower:
            key = "pingala"
        elif "ida" in swara_lower:
            key = "ida"
        else:
            key = None

        if key and key in _SWARA_BONUS:
            bs, br, bt = _SWARA_BONUS[key]
            sat_raw += bs * 0.2
            raj_raw += br * 0.2
            tam_raw += bt * 0.2

    # ── Step 4: Special rule — deep Delta with retained Theta is Tamas not Sattva
    delta = float(band_rel.get("delta", 0.0))
    theta = float(band_rel.get("theta", 0.0))
    alpha = float(band_rel.get("alpha", 0.0))
    if delta > 0.30 and alpha < 0.20:
        # Strong Tamas override — sleep-like or very dull state
        tam_raw += 0.5

    # ── Step 5: Special rule — gamma surge with high beta is clearly Rajas
    gamma = float(band_rel.get("gamma", 0.0))
    beta  = float(band_rel.get("beta",  0.0))
    if gamma > 0.12 and beta > 0.25:
        raj_raw += 0.4

    # ── Step 6: Softplus to prevent zero or negative scores ──────────────────
    sat_raw = _softplus(sat_raw)
    raj_raw = _softplus(raj_raw)
    tam_raw = _softplus(tam_raw)

    # ── Step 7: Normalise to sum = 1.0 ───────────────────────────────────────
    total = sat_raw + raj_raw + tam_raw
    if total < 1e-10:
        return {"sattva": 0.334, "rajas": 0.333, "tamas": 0.333}

    return {
        "sattva": round(sat_raw / total, 4),
        "rajas":  round(raj_raw / total, 4),
        "tamas":  round(tam_raw / total, 4),
    }


def gunas_label(gunas: dict) -> str:
    """
    Return a human-readable dominant-Guna label, e.g. "Sattvic" / "Rajasic" / "Tamasic".
    Falls back to "Balanced" when no Guna exceeds 45%.
    """
    dominant = max(gunas, key=gunas.get)
    value = gunas[dominant]
    if value < 0.45:
        return "Balanced"
    return {"sattva": "Sattvic", "rajas": "Rajasic", "tamas": "Tamasic"}.get(dominant, "Balanced")


def gunas_note(gunas: dict) -> str:
    """One-sentence interpretive note for the dominant Guna state."""
    dominant = max(gunas, key=gunas.get)
    value = gunas[dominant]
    if value < 0.45:
        return ("The three Gunas are in relative equilibrium — a balanced, "
                "transitional mental state.")
    notes = {
        "sattva": (
            "Sattva predominates — the mind is luminous, clear, and calm. "
            "An ideal condition for contemplation, insight, and yogic practice."
        ),
        "rajas": (
            "Rajas predominates — the mind is active, driven, and outward-directed. "
            "Energy is high; channel it intentionally to avoid distraction."
        ),
        "tamas": (
            "Tamas predominates — the mind tends toward heaviness, dullness, or inertia. "
            "Stimulating Pranayama (Kapalabhati, Bhastrika) can help elevate the state."
        ),
    }
    return notes.get(dominant, "")
