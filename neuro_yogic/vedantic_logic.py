"""
vedantic_logic.py
=================
Maps EEG features to two classical Yogic frameworks:

SWARA RHYTHMS (Svara Shastra / Swara Yoga)
  The Swara is the dominant life-force channel (nadi):
  - Ida      -- lunar / left nadi -- parasympathetic, receptive, right-brain active
  - Pingala  -- solar / right nadi -- sympathetic, active, left-brain active
  - Sushumna -- central / balanced -- both nadis in equilibrium

  Threshold: |right - left| / mean > 10% -> Ida or Pingala.
             Within 10%                   -> Sushumna.

TATTVA / CHAKRA CORRELATES (Samkhya-Yoga elemental model)
  - Gamma surge (>12% relative) -> Potential Tattva Activation / Spanda
  - High Theta + low Beta       -> Pratyahara Window (sensory withdrawal)
  - Delta surge + retained Theta -> Turiya Approach (fourth state)
"""

from dataclasses import dataclass, field
from typing import List, Optional

ASYMMETRY_THRESHOLD = 0.10
GAMMA_THRESHOLD     = 0.12
THETA_HIGH          = 0.25
BETA_LOW            = 0.10
DELTA_SURGE         = 0.30


@dataclass
class VedanticReading:
    swara:               str       = "Sushumna (Balanced)"
    swara_confidence:    str       = "Low"
    swara_note:          str       = ""
    tattva_flags:        List[str] = field(default_factory=list)
    contemplative_depth: str       = "Surface"

    def to_dict(self) -> dict:
        return {
            "swara": {
                "state":      self.swara,
                "confidence": self.swara_confidence,
                "note":       self.swara_note,
            },
            "tattva_flags":        self.tattva_flags,
            "contemplative_depth": self.contemplative_depth,
        }


def _classify_swara(alpha_left: float, alpha_right: float) -> tuple:
    """Determine Swara from hemispheric Alpha asymmetry."""
    mean_alpha = (alpha_left + alpha_right) / 2.0
    if mean_alpha < 1e-12:
        return "Sushumna (Balanced)", "Low", "Insufficient Alpha power to determine Swara."

    ratio = (alpha_right - alpha_left) / mean_alpha

    if ratio > ASYMMETRY_THRESHOLD:
        mag  = abs(ratio)
        conf = "High" if mag > 0.25 else "Moderate" if mag > 0.10 else "Low"
        return (
            "Ida (Parasympathetic / Lunar)",
            conf,
            f"Right-hemisphere Alpha is {mag * 100:.1f}% stronger. "
            "Ida nadi: parasympathetic dominance -- ideal for Yoga Nidra, Yin, deep contemplation.",
        )
    elif ratio < -ASYMMETRY_THRESHOLD:
        mag  = abs(ratio)
        conf = "High" if mag > 0.25 else "Moderate" if mag > 0.10 else "Low"
        return (
            "Pingala (Sympathetic / Solar)",
            conf,
            f"Left-hemisphere Alpha is {mag * 100:.1f}% stronger. "
            "Pingala nadi: sympathetic dominance -- ideal for Pranayama, dynamic Asana, cognitive work.",
        )
    else:
        mag  = abs(ratio)
        conf = "High" if mag < 0.03 else "Moderate"
        return (
            "Sushumna (Balanced / Central)",
            conf,
            f"Hemispheric balance within {mag * 100:.1f}% of threshold. "
            "Sushumna open: optimal window for deep meditation and Kundalini practices.",
        )


def _detect_tattva_flags(band_rel: dict, gamma_spike: bool) -> List[str]:
    """Scan band powers for Tattva/Chakra correlate flags."""
    flags = []

    if gamma_spike or band_rel.get("gamma", 0) > GAMMA_THRESHOLD:
        flags.append(
            "Potential Tattva Activation -- Gamma burst >30 Hz detected. "
            "Associated with cortical binding / insight events (Spanda). "
            "Cross-reference with first-person report."
        )

    if band_rel.get("theta", 0) > THETA_HIGH and band_rel.get("beta", 0) < BETA_LOW:
        flags.append(
            "Pratyahara Window -- Elevated Theta + suppressed Beta. "
            "Sensory withdrawal phase active; thalamic gating engaged. "
            "Optimal moment for inward Dharana (concentration practice)."
        )

    if band_rel.get("delta", 0) > DELTA_SURGE and band_rel.get("theta", 0) > 0.20:
        flags.append(
            "Turiya Approach -- Delta surge with retained Theta in waking context. "
            "Rare; may indicate onset of formless absorption. "
            "Verify practitioner is conscious and not drifting into sleep."
        )

    return flags


def _assess_depth(chitta_bhumi: Optional[str], band_rel: dict) -> str:
    """Map Chitta Bhumi label to practitioner-facing depth descriptor."""
    depth_map = {
        "Kshipta":   "Surface",
        "Vikshipta": "Emerging",
        "Ekagra":    "Deep",
        "Niruddha":  "Profound",
    }
    base = depth_map.get(chitta_bhumi or "", "Surface")
    if base == "Profound" and band_rel.get("beta", 0) > 0.20:
        base = "Deep"     # High Beta contradicts Niruddha -- transitional moment
    if base == "Surface" and band_rel.get("alpha", 0) > 0.35:
        base = "Emerging" # Strong Alpha with Kshipta -> borderline
    return base


def vedantic_analyze(
    info: dict,
    chitta_bhumi: Optional[str] = None,
) -> VedanticReading:
    """
    Main entry: produce a VedanticReading from FeatureExtractor.extract() output.
    """
    band_rel    = info.get("band_relative", {})
    alpha_left  = info.get("alpha_left",   0.0)
    alpha_right = info.get("alpha_right",  0.0)
    gamma_spike = info.get("gamma_spike",  False)

    swara, confidence, note = _classify_swara(alpha_left, alpha_right)
    tattva_flags = _detect_tattva_flags(band_rel, gamma_spike)
    depth        = _assess_depth(chitta_bhumi, band_rel)

    return VedanticReading(
        swara               = swara,
        swara_confidence    = confidence,
        swara_note          = note,
        tattva_flags        = tattva_flags,
        contemplative_depth = depth,
    )
