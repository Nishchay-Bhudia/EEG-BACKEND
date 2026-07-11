"""
feature_extractor.py
====================
Enhanced signal processing pipeline based on:
"Electroencephalographic Mapping of Yogic Physiology: An Integrated
Neuroscientific Framework for Real-Time Biofeedback Calibration"

Band definitions aligned with paper specifications:
  delta     : 0.5–4 Hz   — Tamas / deep sleep / Mudha bhumi
  theta     : IAF-anchored (Klimesch 1999), ~4–8 Hz — Sattva / creativity / Ekagra bhumi (Fm-θ)
  alpha     : IAF-anchored (Klimesch 1999), ~8–13 Hz — Sattva / relaxed awareness / Vikshipta bhumi
  low_beta  : IAF-anchored upper edge — 18 Hz — SMR/mid-beta — neutral to calm-focus
  high_beta : 18–30 Hz   ← PRIMARY Rajas / Kshipta marker (paper: "desynchronized
                           low-amplitude High Beta 18-30 Hz in prefrontal areas")
  gamma     : 30–50 Hz   — Niruddha / peak states / Gunatita

Individual Alpha Frequency (IAF) anchoring
  A fixed 8–13 Hz alpha band assumes every subject's alpha rhythm peaks in
  the same place. It doesn't — resting peak alpha frequency varies person
  to person (commonly cited range ~7.5–12.5 Hz) and even session to session.
  Per Klimesch (1999), theta/alpha/low-beta band edges are anchored to each
  epoch's own detected alpha peak (IAF) instead of population-fixed bounds.
  This also mitigates a known confound in gamma/state classification: traits
  like resting gamma or alpha amplitude vary by individual (Cahn & Polich,
  2006 note this gamma difference is often a *trait* effect of long-term
  practice, not a within-session *state* change) — see personal baseline
  z-scoring below, which addresses the amplitude side of the same problem.

New derived metrics:
  faa : ln(alpha_right) − ln(alpha_left)
        Standard Frontal Alpha Asymmetry (neuroscience convention).
        Positive → left PFC activation → Pingala (FAA > +0.15 per paper)
        Negative → right PFC activation → Ida   (FAA < −0.15 per paper)
        Near zero → Sushumna equilibrium         (|FAA| ≤ 0.15 per paper)
        NOTE: FAA/Swara (Nadi) lateralization is intentionally NOT used as
        a Guna (Sattva/Rajas/Tamas) input — see satva_classifier.py.

  plv : Phase Locking Value between left & right alpha channels.
        Measures interhemispheric synchrony — the neural signature of Sushumna
        and Niruddha states. PLV > 0.80 = high coherence (paper: Niruddha).

  iaf : Individual Alpha Frequency detected from this epoch's own PSD
        (7–13 Hz search window). Falls back to the 10 Hz population average
        with iaf_detected=False when no clear peak is present.

Electrode layout (4-channel Muse S / Muse 2):
  Left  channels: indices 0, 1  (TP9, AF7)
  Right channels: indices 2, 3  (AF8, TP10)

Artifact detection (gating, not just informational)
  Each epoch is screened for gross contamination — amplitude blow-outs
  (blink/jaw/movement), flatline/saturation (poor electrode contact), and
  high kurtosis (spiky non-Gaussian noise, the standard EEGLAB-style
  automated-rejection heuristic). Callers (main.py) must reject epochs
  where `info["artifact"]["detected"]` is True *before* trusting FAA/PLV/
  band-power outputs — this module computes them regardless (so callers can
  still inspect what a garbage epoch looked like), but does not decide the
  HTTP-level accept/reject itself.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch, hilbert
from scipy.stats import kurtosis as _kurtosis

# ── Band definitions (paper-aligned; theta/alpha/low_beta are IAF-anchored
#    per-epoch at runtime — see _detect_iaf and _anchored_bands) ─────────────
EEG_BANDS = {
    "delta":     (0.5,  4.0),
    "theta":     (4.0,  8.0),
    "alpha":     (8.0, 13.0),
    "low_beta":  (13.0, 18.0),   # SMR + mid-beta (calm-focus, Manipura/Vishuddha)
    "high_beta": (18.0, 30.0),   # Rajas marker (Kshipta / stress / fight-flight)
    "gamma":     (30.0, 50.0),   # Niruddha / peak states
}

# ── IAF search parameters (Klimesch 1999) ────────────────────────────────────
IAF_SEARCH_LO = 7.0
IAF_SEARCH_HI = 13.0
IAF_DEFAULT   = 10.0    # population-average fallback when no clear peak found

# ── Artifact-detection thresholds ─────────────────────────────────────────────
# Consumer dry-electrode headband (Muse), forehead + mastoid placement.
#
# Recalibrated after real-hardware use: the original values (amplitude 300,
# kurtosis 5.0, channel-fraction 0.5) rejected most/all epochs in practice —
# a gate that's too strict is worse than one that's too lenient, since it
# silently starves every downstream reading (Chitta Bhumi, Gunas, Swara) of
# data rather than occasionally letting a slightly noisy epoch through.
# Two specific things were wrong:
#   - 300 µV doesn't tolerate a normal blink at frontal sites (AF7/AF8),
#     which alone can hit 100-200+ µV — that's expected signal, not noise.
#   - Kurtosis on a 512-sample (2s) window is a noisy statistic — clean
#     EEG can swing well past 5.0 just from estimation variance at this
#     window length, not because anything is actually wrong with the signal.
ARTIFACT_AMPLITUDE_UV   = 600.0   # |sample| beyond this = blink/jaw/movement blow-out
ARTIFACT_FLATLINE_STD   = 0.05    # channel std below this = poor contact / disconnect
ARTIFACT_KURTOSIS       = 12.0    # excess kurtosis above this = spiky non-Gaussian noise
ARTIFACT_CHANNEL_FRAC   = 0.75    # reject epoch only if >=75% of channels are flagged

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
        # Zero-pad to >=1 Hz frequency bins for peak-finding (IAF detection).
        # This doesn't add true spectral resolution beyond what the 0.5s
        # window supports, but it interpolates the PSD finely enough that
        # peak-picking isn't limited to 3 coarse candidate bins.
        self._nfft      = max(sample_rate, self._nperseg)

        nyq = sample_rate / 2.0

        # Butterworth bandpass (0.5–50 Hz), order 4, SOS format
        self._bp_sos = butter(
            4, [0.5 / nyq, min(49.9, 50.0) / nyq],
            btype="bandpass", output="sos"
        )

        # IIR notch (50 or 60 Hz power-line interference)
        b, a = iirnotch(min(notch_freq / nyq, 0.999), Q=30.0)
        self._notch_sos = tf2sos(b, a)

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
        info     : dict — all intermediate products for downstream classifiers.
                   Includes "artifact" (dict with "detected" bool) — callers
                   MUST check this and reject the epoch before trusting the
                   rest of `info`; this method does not raise on its own.
        """
        is_padded = (meta or {}).get("is_padded", False)
        raw_eeg   = np.atleast_2d(np.array(raw_eeg, dtype=np.float64))
        n_ch, n_samp = raw_eeg.shape

        # ── Step 0: Artifact screening (on raw signal, before filtering) ──────
        artifact = self._detect_artifacts(raw_eeg)

        # ── Step 1: Bandpass filter (0.5–50 Hz) ───────────────────────────────
        filtered = sosfiltfilt(self._bp_sos, raw_eeg, axis=1)

        # ── Step 2: Notch filter (power-line) ─────────────────────────────────
        filtered = sosfiltfilt(self._notch_sos, filtered, axis=1)

        # ── Step 3: Welch PSD ─────────────────────────────────────────────────
        # freqs: (n_freq,)   psd: (n_channels, n_freq)
        seg_len = min(self._nperseg, n_samp)
        freqs, psd = welch(
            filtered,
            fs       = self._sr,
            nperseg  = seg_len,
            noverlap = seg_len // 2,
            nfft     = max(self._nfft, seg_len),
            axis     = 1,
        )

        # ── Step 3b: Individual Alpha Frequency + IAF-anchored band edges ─────
        iaf, iaf_detected = self._detect_iaf(freqs, psd)
        bands = self._anchored_bands(iaf)

        # ── Step 4: Band power integration (trapezoid rule) ───────────────────
        band_abs = {}
        for band_name, (lo, hi) in bands.items():
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

        # ── Step 6: Hemispheric Alpha Asymmetry (uses the same IAF-anchored
        #    alpha band as Step 4, so FAA reflects this person's own rhythm) ──
        alpha_lo, alpha_hi = bands["alpha"]
        alpha_mask = (freqs >= alpha_lo) & (freqs < alpha_hi)

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
        plv = self._compute_plv(filtered, alpha_lo, alpha_hi)

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
            "band_edges":      bands,
            "alpha_left":      alpha_left,
            "alpha_right":     alpha_right,
            "alpha_asymmetry": asymmetry,
            "faa":             faa,
            "plv":             plv,
            "iaf":             iaf,
            "iaf_detected":    iaf_detected,
            "gamma_spike":     band_rel["gamma"] > 0.12,
            "is_padded":       is_padded,
            "artifact":        artifact,
        }
        return features, info

    # ── Private helpers ────────────────────────────────────────────────────────

    def _detect_iaf(self, freqs: np.ndarray, psd: np.ndarray) -> Tuple[float, bool]:
        """
        Individual/Peak Alpha Frequency detection (Klimesch, 1999).

        Finds the local power maximum within a 7–13 Hz search window on the
        channel-averaged PSD, instead of assuming every subject's alpha
        rhythm peaks at the same population-average frequency. A genuine
        peak must sit strictly inside the window (not at its edge, which
        usually means we're seeing a slope rather than a real peak) and
        clear the window's mean power by a margin. Otherwise falls back to
        the population-average default and flags iaf_detected=False so
        callers know the bands below are population- not individually-anchored.
        """
        mask = (freqs >= IAF_SEARCH_LO) & (freqs <= IAF_SEARCH_HI)
        if mask.sum() < 3:
            return IAF_DEFAULT, False

        band_freqs = freqs[mask]
        band_power = np.mean(psd[:, mask], axis=0)

        peak_i     = int(np.argmax(band_power))
        peak_freq  = float(band_freqs[peak_i])
        peak_power = float(band_power[peak_i])
        mean_power = float(np.mean(band_power)) or 1e-12

        at_edge   = peak_i == 0 or peak_i == len(band_freqs) - 1
        prominent = peak_power > mean_power * 1.3
        if at_edge or not prominent:
            return IAF_DEFAULT, False

        return peak_freq, True

    def _anchored_bands(self, iaf: float) -> dict:
        """
        Klimesch-style band edges anchored to this epoch's IAF.

        Only theta/alpha/low_beta shift with IAF — delta, high_beta, and
        gamma have no established individual-anchoring convention in the
        literature and stay at their fixed population ranges.
        """
        theta_lo = max(iaf - 6.0, 4.0)
        theta_hi = iaf - 2.0
        alpha_lo = iaf - 2.0
        alpha_hi = iaf + 2.0
        return {
            "delta":     (0.5, 4.0),
            "theta":     (theta_lo, theta_hi),
            "alpha":     (alpha_lo, alpha_hi),
            "low_beta":  (alpha_hi, 18.0),
            "high_beta": (18.0, 30.0),
            "gamma":     (30.0, 50.0),
        }

    def _detect_artifacts(self, raw_eeg: np.ndarray) -> dict:
        """
        Screen raw EEG for gross contamination (EEGLAB-style heuristics).

        Per channel: amplitude blow-out (blink/jaw/movement), flatline/
        saturation (poor electrode contact), and high excess kurtosis
        (spiky non-Gaussian noise). The epoch is flagged `detected=True`
        once at least ARTIFACT_CHANNEL_FRAC of channels trip any check.
        This method only reports — main.py is responsible for actually
        rejecting flagged epochs before returning a classification.
        """
        n_ch = raw_eeg.shape[0]
        flagged: List[int] = []
        reasons: dict = {}

        for ch in range(n_ch):
            sig = raw_eeg[ch]
            ch_reasons = []

            if np.any(np.abs(sig) > ARTIFACT_AMPLITUDE_UV):
                ch_reasons.append("amplitude")

            std = float(np.std(sig))
            if std < ARTIFACT_FLATLINE_STD:
                ch_reasons.append("flatline")
            elif sig.size > 3:
                k = float(_kurtosis(sig, fisher=True, bias=False))
                if k > ARTIFACT_KURTOSIS:
                    ch_reasons.append("kurtosis")

            if ch_reasons:
                flagged.append(ch)
                reasons[ch] = ch_reasons

        frac_flagged = (len(flagged) / n_ch) if n_ch else 0.0
        return {
            "detected":         frac_flagged >= ARTIFACT_CHANNEL_FRAC,
            "channels_flagged": flagged,
            "channel_reasons":  reasons,
            "fraction_flagged": round(frac_flagged, 3),
        }

    def _compute_plv(
        self,
        filtered: np.ndarray,
        alpha_lo: float = 8.0,
        alpha_hi: float = 13.0,
    ) -> float:
        """
        Compute alpha-band Phase Locking Value between left and right channels.

        PLV = |mean(exp(i * Δφ(t)))| where Δφ = φ_right − φ_left.
        Range: 0 (no coherence) → 1.0 (perfect phase lock).
        Uses the same IAF-anchored alpha band as the rest of the epoch, so
        the coherence measure tracks this person's own alpha rhythm.

        Returns 0.5 (neutral) if insufficient channels or samples.
        """
        n_ch, n_samp = filtered.shape
        left_idx  = [i for i in self._left_idx  if i < n_ch]
        right_idx = [i for i in self._right_idx if i < n_ch]
        if not left_idx or not right_idx or n_samp < 64:
            return 0.5

        try:
            nyq = self._sr / 2.0
            alpha_sos = butter(
                4, [alpha_lo / nyq, alpha_hi / nyq],
                btype="bandpass", output="sos"
            )
            # Alpha-bandpass the signals
            left_alpha  = sosfiltfilt(alpha_sos, filtered[left_idx[0]],  axis=0)
            right_alpha = sosfiltfilt(alpha_sos, filtered[right_idx[0]], axis=0)

            # Analytic signal → instantaneous phase
            left_phase  = np.angle(hilbert(left_alpha))
            right_phase = np.angle(hilbert(right_alpha))

            phase_diff = right_phase - left_phase
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            return float(np.clip(plv, 0.0, 1.0))
        except Exception:
            return 0.5
