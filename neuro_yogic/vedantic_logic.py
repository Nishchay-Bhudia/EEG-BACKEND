"""
vedantic_logic.py  (v2 — paper-aligned)
========================================
Maps EEG features to three classical Yogic frameworks:

SWARA (Nadi) CLASSIFICATION — paper thresholds
  Ida Nadi      (lunar / left):   FAA < −0.15  (paper: "negative FAA score
                                   indicating left-sided alpha accumulation
                                   and relative right prefrontal activation")
  Pingala Nadi  (solar / right):  FAA > +0.15  (paper: "positive FAA score
                                   indicating relative left-sided frontal
                                   cortical activation")
  Sushumna Nadi (central):        |FAA| ≤ 0.15 (paper: "Frontal Alpha
                                   Asymmetry balances toward zero ≈ 0")

TATTVA / CHAKRA CORRELATES
  - Gamma surge (>12% relative)       → Ajna/Sahasrara activation (Spanda)
  - High Theta + low High-Beta        → Pratyahara Window (Ekagra approach)
  - Delta surge (>35%) + low Alpha    → Tamasic heaviness (Mudha approach)
  - High PLV (>0.80) + low High-Beta  → Sushumna activation / Niruddha

TRIGUNAS
  Derived from band powers + FAA + PLV + Chitta Bhumi via satva_classifier.py

CHITTA BHUMI DEPTH MAPPING (v2 — 5 states)
  Mudha     → "Deep Inertia"  (lowest — Tamas dominant)
  Kshipta   → "Surface"       (scattered — Rajas dominant)
  Vikshipta → "Emerging"      (oscillating — Sattva rising)
  Ekagra    → "Deep"          (one-pointed — pure Sattva)
  Niruddha  → "Profound"      (mastered — beyond gunas)
"""

from dataclasses import dataclass, field
from typing import List, Optional

from neuro_yogic.satva_classifier import classify_gunas, gunas_label, gunas_note

# ── FAA Nadi thresholds (from paper) ─────────────────────────────────────────
# Paper: Ida FAA < −0.15, Pingala FAA > +0.15, Sushumna |FAA| ≤ 0.15
FAA_IDA_THRESHOLD      = -0.15   # Right prefrontal > Left prefrontal
FAA_PINGALA_THRESHOLD  =  0.15   # Left prefrontal  > Right prefrontal
# Old raw-difference threshold (kept for fallback)
_LEGACY_ASYM_THRESHOLD =  0.10

# Tattva thresholds (from paper)
GAMMA_THRESHOLD  = 0.12   # >12% relative = significant gamma surge
THETA_PRATYAHARA = 0.25   # >25% theta + low high-beta = Pratyahara window
HIGH_BETA_LOW    = 0.10   # <10% high-beta = suppressed stress bands
DELTA_SURGE      = 0.35   # >35% delta in waking = Tamasic heaviness
PLV_COHERENCE    = 0.80   # >80% PLV = Sushumna/Niruddha coherence


@dataclass
class VedanticReading:
    swara:               str       = "Sushumna (Balanced)"
    swara_confidence:    str       = "Low"
    swara_note:          str       = ""
    tattva_flags:        List[str] = field(default_factory=list)
    contemplative_depth: str       = "Surface"
    gunas:               dict      = field(default_factory=lambda: {"sattva": 0.334, "rajas": 0.333, "tamas": 0.333})
    guna_label:          str       = "Balanced"
    guna_note:           str       = ""

    def to_dict(self) -> dict:
        return {
            "swara": {
                "state":      self.swara,
                "confidence": self.swara_confidence,
                "note":       self.swara_note,
            },
            "tattva_flags":        self.tattva_flags,
            "contemplative_depth": self.contemplative_depth,
            "gunas": {
                "sattva":  self.gunas.get("sattva", 0.0),
                "rajas":   self.gunas.get("rajas",  0.0),
                "tamas":   self.gunas.get("tamas",  0.0),
                "label":   self.guna_label,
                "note":    self.guna_note,
            },
        }


# ── Swara / Nadi classification ───────────────────────────────────────────────

def _classify_swara(
    alpha_left: float,
    alpha_right: float,
    faa: Optional[float] = None,
) -> tuple:
    """
    Determine Swara (Nadi) from Frontal Alpha Asymmetry.

    Uses the standard ln-ratio FAA when available (from feature extractor).
    Falls back to raw (alpha_right - alpha_left) / mean for legacy callers.

    Paper thresholds:
      Ida      → FAA < −0.15  (right prefrontal activation)
      Pingala  → FAA > +0.15  (left prefrontal activation)
      Sushumna → |FAA| ≤ 0.15 (balanced / both nostrils)
    """
    # Prefer the pre-computed ln-ratio FAA (more accurate)
    if faa is not None:
        score = float(faa)
    else:
        # Legacy fallback: raw asymmetry ratio
        mean_alpha = (alpha_left + alpha_right) / 2.0
        if mean_alpha < 1e-12:
            return "Sushumna (Balanced)", "Low", "Insufficient Alpha power to determine Swara."
        score = (alpha_right - alpha_left) / mean_alpha

    mag = abs(score)

    if score < FAA_IDA_THRESHOLD:
        # Left-sided Alpha accumulation → Right hemisphere activation → Ida
        conf = "High" if mag > 0.40 else "Moderate" if mag > 0.20 else "Low"
        return (
            "Ida (Parasympathetic / Lunar)",
            conf,
            f"Frontal Alpha Asymmetry = {score:+.2f} (threshold < {FAA_IDA_THRESHOLD}). "
            "Right-hemisphere activation detected. Ida nadi: parasympathetic dominance — "
            "receptive, creative, and introspective. Ideal for Yoga Nidra, Yin practice, "
            "deep contemplation, and emotional processing.",
        )
    elif score > FAA_PINGALA_THRESHOLD:
        # Right-sided Alpha accumulation → Left hemisphere activation → Pingala
        conf = "High" if mag > 0.40 else "Moderate" if mag > 0.20 else "Low"
        return (
            "Pingala (Sympathetic / Solar)",
            conf,
            f"Frontal Alpha Asymmetry = {score:+.2f} (threshold > +{FAA_PINGALA_THRESHOLD}). "
            "Left-hemisphere activation detected. Pingala nadi: sympathetic dominance — "
            "analytical, goal-directed, and action-oriented. Ideal for Pranayama, "
            "dynamic Asana, cognitive work, and verbal/logical tasks.",
        )
    else:
        # Balanced → Sushumna
        conf = "High" if mag < 0.05 else "Moderate"
        return (
            "Sushumna (Balanced / Central)",
            conf,
            f"Frontal Alpha Asymmetry = {score:+.2f} — near equilibrium. "
            "Both hemispheres balanced. Sushumna nadi: autonomic coherence — "
            "ideal state for deep meditation, Samadhi approach, and unified awareness. "
            "The gateway to higher contemplative states.",
        )


# ── Tattva / Chakra flag detection ───────────────────────────────────────────

def _detect_tattva_flags(
    band_rel: dict,
    alpha_left: float,
    alpha_right: float,
    faa: float = 0.0,
    plv: float = 0.5,
) -> List[str]:
    """
    Detect significant yogic state indicators from EEG features.
    Returns a list of human-readable flag strings.
    """
    flags   = []
    gamma   = band_rel.get("gamma",     0.0)
    theta   = band_rel.get("theta",     0.0)
    delta   = band_rel.get("delta",     0.0)
    alpha   = band_rel.get("alpha",     0.0)
    high_beta = band_rel.get("high_beta", band_rel.get("beta", 0.0) * 0.45)

    # Gamma surge → Ajna/Sahasrara activation
    if gamma > GAMMA_THRESHOLD:
        flags.append(f"Gamma Surge ({gamma*100:.0f}%) — Ajna/Sahasrara activation: "
                     f"multi-sensory binding, peak insight, Spanda (divine pulse)")

    # Pratyahara window
    if theta > THETA_PRATYAHARA and high_beta < HIGH_BETA_LOW:
        flags.append(f"Pratyahara Window — Fm-θ ({theta*100:.0f}%) with suppressed "
                     f"high-beta ({high_beta*100:.0f}%): sensory withdrawal, "
                     f"approach to Ekagra")

    # Turiya approach — Delta surge with retained consciousness markers
    if delta > DELTA_SURGE and alpha > 0.15:
        flags.append(f"Turiya Approach — Delta ({delta*100:.0f}%) + waking Alpha "
                     f"({alpha*100:.0f}%): deep Yoga Nidra or restorative Delta-Alpha "
                     f"healing blend (Svapna-Jagrat boundary)")

    # Sushumna / Niruddha coherence signature
    if plv > PLV_COHERENCE and abs(faa) < 0.10:
        flags.append(f"Sushumna Activated — PLV {plv:.2f} + balanced FAA ({faa:+.2f}): "
                     f"interhemispheric coherence, unified awareness, Samadhi approach")

    # High-beta agitation warning
    if high_beta > 0.30:
        flags.append(f"High-Beta Agitation ({high_beta*100:.0f}%) — Kshipta tendency: "
                     f"prefrontal hyperarousal. Nadi Shodhana pranayama recommended.")

    # Tamasic heaviness
    if delta > 0.40 and alpha < 0.10:
        flags.append(f"Tamasic State — Delta surge ({delta*100:.0f}%) + absent Alpha "
                     f"({alpha*100:.0f}%): cognitive heaviness / Mudha bhumi. "
                     f"Stimulating pranayama (Kapalabhati) recommended.")

    return flags


# ── Contemplative depth ───────────────────────────────────────────────────────

_BHUMI_DEPTH = {
    "Mudha":     "Deep Inertia",   # NEW: lowest state
    "Kshipta":   "Surface",
    "Vikshipta": "Emerging",
    "Ekagra":    "Deep",
    "Niruddha":  "Profound",
}


# ── Main public function ──────────────────────────────────────────────────────

def vedantic_analyze(
    info: dict,
    chitta_bhumi: Optional[str] = None,
) -> VedanticReading:
    """
    Produce a complete VedanticReading from the feature-extractor info dict.

    Parameters
    ----------
    info         : dict from FeatureExtractor.extract() — must contain
                   band_relative, alpha_left, alpha_right, faa, plv.
    chitta_bhumi : optional Chitta Bhumi label (used for Guna secondary adj.)

    Returns
    -------
    VedanticReading dataclass with .to_dict() method.
    """
    band_rel    = info.get("band_relative", {})
    alpha_left  = float(info.get("alpha_left",  1e-12))
    alpha_right = float(info.get("alpha_right", 1e-12))
    faa         = float(info.get("faa",         0.0))
    plv         = float(info.get("plv",         0.50))

    # ── Swara classification (using ln-ratio FAA) ─────────────────────────
    swara_state, swara_conf, swara_note = _classify_swara(
        alpha_left, alpha_right, faa=faa
    )

    # ── Tattva / Chakra flags ─────────────────────────────────────────────
    tattva_flags = _detect_tattva_flags(band_rel, alpha_left, alpha_right, faa, plv)

    # ── Contemplative depth from Chitta Bhumi ─────────────────────────────
    contemplative_depth = _BHUMI_DEPTH.get(chitta_bhumi or "", "Surface")

    # ── Trigunas (paper-aligned scorer) ───────────────────────────────────
    # Extract swara key for secondary adjustment
    swara_key = (
        "sushumna" if "sushumna" in swara_state.lower()
        else "ida"    if "ida"     in swara_state.lower()
        else "pingala"
    )

    gunas = classify_gunas(
        band_rel,
        faa=faa,
        plv=plv,
        chitta_bhumi=chitta_bhumi,
        swara=swara_key,
    )
    glabel = gunas_label(gunas)
    gnote  = gunas_note(gunas)

    return VedanticReading(
        swara               = swara_state,
        swara_confidence    = swara_conf,
        swara_note          = swara_note,
        tattva_flags        = tattva_flags,
        contemplative_depth = contemplative_depth,
        gunas               = gunas,
        guna_label          = glabel,
        guna_note           = gnote,
    )
