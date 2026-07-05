"""
feature_extractor.py
====================
Enhanced signal processing pipeline based on:
"Electroencephalographic Mapping of Yogic Physiology: An Integrated
Neuroscientific Framework for Real-Time Biofeedback Calibration"

Band definitions aligned with paper specifications:
  delta     : 0.5–4 Hz   — Tamas / deep sleep / Mudha bhumi
  theta     : 4–8 Hz     — Sattva / creativity / Ekagra bhumi (Fm-θ)
  alpha     : 8–13 Hz    — Sattva / relaxed awareness / Vikshipta bhumi
  low_beta  : 13–18 Hz   — SMR/mid-beta — neutral to calm-focus
  high_beta : 18–30 Hz   ← PRIMARY Rajas / Kshipta marker (paper: "desynchronized
                           low-amplitude High Beta 18-30 Hz in prefrontal areas")
  gamma     : 30–50 Hz   — Niruddha / peak states / Gunatita

New derived metrics:
  faa : ln(alpha_right) − ln(alpha_left)
        Standard Frontal Alpha Asymmetry (neuroscience convention).
        Positive → left PFC activation → Pingala (FAA > +0.15 per paper)
        Negative → right PFC activation → Ida   (FAA < −0.15 per paper)
        Near zero → Sushumna equilibrium         (|FAA| ≤ 0.15 per paper)

  plv : Phase Locking Value between left & right alpha channels.
        Measures interhemispheric synchrony — the neural signature of Sushumna
        and Niruddha states. PLV > 0.80 = high coherence (paper: Niruddha).

Electrode layout (4-channel Muse S / Muse 2):
  Left  channels: indices 0, 1  (TP9, AF7)
  Right channels: indices 2, 3  (AF8, TP10)
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch, hilbert

# ── Band definitions (paper-aligned) ─────────────────────────────────────────
EEG_BANDS = {
    "delta":     (0.5,  4.0),
    "theta":     (4.0,  8.0),
    "alpha":     (8.0, 13.0),
    "low_beta":  (13.0, 18.0),   # SMR + mid-beta (calm-focus, Manipura/Vishuddha)
    "high_beta": (18.0, 30.0),   # Rajas marker (Kshipta / stress / fight-flight)
    "gamma":     (30.0, 50.0),   # Niruddha / peak states
}

# Feature vector column names (10-D)
FEATURE_COLUMNS = [
    "delta", "theta", "alpha", "low_beta", "high_beta", "gamma",
    "alpha_left", "alpha_right", "faa", "plv",
]

# NumPy 2.x renamed np.trapz -> np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class FeatureExtractor:
    """
    Transforms raw multi-channel EEG into the 10-D classifier feature vector.

    New vs. v1:
    • Splits beta into low_beta (13-18 Hz) and high_beta (18-30 Hz) — critical
      for accurate Rajas detection (the paper specifies HIGH beta as Rajas).
    • Computes FAA (ln-ratio Frontal Alpha Asymmetry) — the standard metric for
      Nadi/Swara lateralization used in clinical neuroscience.
    • Computes PLV (Phase Locking Value) between left/right alpha channels —
      the neural signature of interhemispheric coherence (Sushumna / Niruddha).
    """

    def __init__(
        self,
        sample_rate:   int                = 256,
        left_indices:  Optional[List[int]] = None,
        right_indices: Optional[List[int]] = None,
        notch_freq:    float              = 50.0,
    ) -> None:
        self._sr        = sample_rate
        self._left_idx  = left_indices  or [0, 1]
        self._right_idx = right_indices or [2, 3]
        self._nperseg   = max(sample_rate // 2, 32)   # 0.5-second Welch segments

        nyq = sample_rate / 2.0

        # Butterworth bandpass (0.5–50 Hz), order 4, SOS format
        self._bp_sos = butter(
            4, [0.5 / nyq, min(49.9, 50.0) / nyq],
            btype="bandpass", output="sos"
        )

        # IIR notch (50 or 60 Hz power-line interference)
        b, a = iirnotch(min(notch_freq / nyq, 0.999), Q=30.0)
        self._notch_sos = tf2sos(b, a)

        # Alpha bandpass for PLV computation (8–13 Hz)
        self._alpha_sos = butter(
            4, [8.0 / nyq, 13.0 / nyq],
            btype="bandpass", output="sos"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        raw_eeg: np.ndarray,
        meta: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Run the full pipeline on one EEG epoch.

        Parameters
        ----------
        raw_eeg : ndarray (n_channels, n_samples)  — raw µV data
        meta    : optional dict (e.g. {"is_padded": True})

        Returns
        -------
        features : ndarray (10,) — [delta, theta, alpha, low_beta, high_beta,
                                    gamma, alpha_left_rel, alpha_right_rel, faa, plv]
        info     : dict — all intermediate products for downstream classifiers
        """
        is_padded = (meta or {}).get("is_padded", False)
        raw_eeg   = np.atleast_2d(np.array(raw_eeg, dtype=np.float64))
        n_ch, n_samp = raw_eeg.shape

        # ── Step 1: Bandpass filter (0.5–50 Hz) ───────────────────────────────
        filtered = sosfiltfilt(self._bp_sos, raw_eeg, axis=1)

        # ── Step 2: Notch filter (power-line) ─────────────────────────────────
        filtered = sosfiltfilt(self._notch_sos, filtered, axis=1)

        # ── Step 3: Welch PSD ─────────────────────────────────────────────────
        # freqs: (n_freq,)   psd: (n_channels, n_freq)
        freqs, psd = welch(
            filtered,
            fs       = self._sr,
            nperseg  = min(self._nperseg, n_samp),
            noverlap = min(self._nperseg, n_samp) // 2,
            axis     = 1,
        )

        # ── Step 4: Band power integration (trapezoid rule) ───────────────────
        band_abs = {}
        for band_name, (lo, hi) in EEG_BANDS.items():
            mask         = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                band_abs[band_name] = 0.0
                continue
            power_per_ch = _trapz(psd[:, mask], freqs[mask], axis=1)
            band_abs[band_name] = float(np.mean(power_per_ch))

        # ── Step 5: Relative power normalisation ──────────────────────────────
        total    = sum(band_abs.values()) or 1e-10
        band_rel = {k: v / total for k, v in band_abs.items()}

        # Also expose a legacy "beta" key = low_beta + high_beta combined
        band_rel["beta"] = band_rel["low_beta"] + band_rel["high_beta"]

        # ── Step 6: Hemispheric Alpha Asymmetry ───────────────────────────────
        alpha_mask = (freqs >= 8.0) & (freqs < 13.0)

        def _mean_alpha(ch_idx: List[int]) -> float:
            valid = [i for i in ch_idx if i < n_ch]
            if not valid:
                return 1e-12
            vals = _trapz(psd[valid][:, alpha_mask], freqs[alpha_mask], axis=1)
            return float(max(np.mean(vals), 1e-12))

        alpha_left  = _mean_alpha(self._left_idx)
        alpha_right = _mean_alpha(self._right_idx)
        asymmetry   = alpha_right - alpha_left   # raw difference (legacy)

        # Standard FAA: ln(alpha_right) − ln(alpha_left)
        # This is the conventional neuroscience formula — it normalises for
        # overall alpha level and produces symmetric values around 0.
        try:
            faa = float(np.log(alpha_right) - np.log(alpha_left))
        except (ValueError, ZeroDivisionError):
            faa = 0.0
        faa = float(np.clip(faa, -2.0, 2.0))   # reasonable range for display

        # Relative alpha (for feature vector)
        alpha_left_rel  = alpha_left  / (total + 1e-12)
        alpha_right_rel = alpha_right / (total + 1e-12)

        # ── Step 7: Phase Locking Value (alpha interhemispheric coherence) ────
        # Measures how consistently left and right hemispheres oscillate in phase.
        # High PLV (>0.80) = Sushumna / Niruddha coherence signature.
        plv = self._compute_plv(filtered)

        # ── Step 8: Assemble 10-D feature vector ──────────────────────────────
        features = np.array([
            band_rel["delta"],
            band_rel["theta"],
            band_rel["alpha"],
            band_rel["low_beta"],
            band_rel["high_beta"],
            band_rel["gamma"],
            alpha_left_rel,
            alpha_right_rel,
            faa,
            plv,
        ], dtype=np.float64)

        info = {
            "band_absolute":   band_abs,
            "band_relative":   band_rel,
            "alpha_left":      alpha_left,
            "alpha_right":     alpha_right,
            "alpha_asymmetry": asymmetry,
            "faa":             faa,
            "plv":             plv,
            "gamma_spike":     band_rel["gamma"] > 0.12,
            "is_padded":       is_padded,
        }
        return features, info

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_plv(self, filtered: np.ndarray) -> float:
        """
        Compute alpha-band Phase Locking Value between left and right channels.

        PLV = |mean(exp(i * Δφ(t)))| where Δφ = φ_right − φ_left.
        Range: 0 (no coherence) → 1.0 (perfect phase lock).

        Returns 0.5 (neutral) if insufficient channels or samples.
        """
        n_ch, n_samp = filtered.shape
        left_idx  = [i for i in self._left_idx  if i < n_ch]
        right_idx = [i for i in self._right_idx if i < n_ch]
        if not left_idx or not right_idx or n_samp < 64:
            return 0.5

        try:
            # Alpha-bandpass the signals
            left_alpha  = sosfiltfilt(self._alpha_sos, filtered[left_idx[0]],  axis=0)
            right_alpha = sosfiltfilt(self._alpha_sos, filtered[right_idx[0]], axis=0)

            # Analytic signal → instantaneous phase
            left_phase  = np.angle(hilbert(left_alpha))
            right_phase = np.angle(hilbert(right_alpha))

            phase_diff = right_phase - left_phase
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            return float(np.clip(plv, 0.0, 1.0))
        except Exception:
            return 0.5
