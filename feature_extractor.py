"""
feature_extractor.py — Signal Processing & Feature Extraction
==============================================================
Transforms raw, noisy EEG voltage traces into the 8-dimensional feature
vector consumed by the YogaClassifier.

Pipeline (per 2-second epoch):
  1. Bandpass filter  1–50 Hz   — removes DC drift and high-freq EMG noise
  2. Notch filter at 50/60 Hz   — removes AC power-line interference
  3. Welch's PSD                — estimates spectral power across frequencies
  4. Band power integration     — sums PSD within canonical EEG bands
  5. Relative power normalisation — converts absolute µV²/Hz → 0–1 scale
  6. Hemispheric asymmetry      — Right α − Left α (scalar index)

Why Welch's Method instead of raw FFT?
---------------------------------------
A direct FFT of a 2-second window produces a noisy, single-trial spectrum.
Welch's method divides the window into overlapping sub-segments, computes a
periodogram for each, and averages them. This dramatically reduces variance
in the PSD estimate at the cost of some frequency resolution — which is
the right trade-off for EEG bands that span 2–13 Hz each.

Electrode layout assumption
----------------------------
For a 4-channel headset (e.g., Muse 2: TP9, AF7, AF8, TP10):
  Left hemisphere  channels: indices 0, 1  (TP9, AF7)
  Right hemisphere channels: indices 2, 3  (AF8, TP10)

For an 8-channel headset (OpenBCI Cyton with standard 10-20):
  Left  → F3, C3, P3, O1  (channels 1, 3, 5, 7)
  Right → F4, C4, P4, O2  (channels 2, 4, 6, 8)

The `left_indices` / `right_indices` constructor parameters let you
override the defaults for any arbitrary headset layout.
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
from scipy.signal import welch


# Canonical EEG frequency bands (Hz) — universally accepted ranges.
# The boundaries between bands are historical conventions, not sharp
# physiological boundaries, but they are consistent across the literature.
BANDS = {
    #  name       : (low_Hz, high_Hz)
    "delta" : (1.0,   4.0),   # deep sleep, unconscious processing
    "theta" : (4.0,   8.0),   # drowsiness, meditation, memory encoding
    "alpha" : (8.0,  13.0),   # relaxed wakefulness, cortical idling
    "beta"  : (13.0, 30.0),   # active thinking, focus, anxiety
    "gamma" : (30.0, 50.0),   # high-level binding, insight, Tattva activation
}


class FeatureExtractor:
    """
    Converts a raw multi-channel EEG array into a normalised 8-D feature vector.

    Parameters
    ----------
    sample_rate : int
        Sampling frequency in Hz (e.g., 256 for Muse 2, 250 for Cyton).
    left_indices : list of int, optional
        Column indices (into the EEG channel array) that correspond to
        left-hemisphere electrodes. Default: [0, 1] (Muse 2 TP9, AF7).
    right_indices : list of int, optional
        Right-hemisphere electrode indices. Default: [2, 3] (Muse 2 AF8, TP10).
    notch_freq : float
        Power-line interference frequency (50 Hz in EU/Asia, 60 Hz in US/Canada).
    nperseg : int or None
        Welch segment length in samples. None → scipy default (≈ 1/8 of window).
    """

    def __init__(
        self,
        sample_rate: int = 256,
        left_indices:  Optional[List[int]] = None,
        right_indices: Optional[List[int]] = None,
        notch_freq:    float = 50.0,
        nperseg:       Optional[int] = None,
    ) -> None:
        self._sr           = sample_rate
        self._left_idx     = left_indices  if left_indices  is not None else [0, 1]
        self._right_idx    = right_indices if right_indices is not None else [2, 3]
        self._notch_freq   = notch_freq
        self._nperseg      = nperseg or (sample_rate // 2)  # 0.5 s segments

        # Pre-compute filter coefficients once at init time (not per epoch).
        # Using second-order sections (sos) for numerical stability —
        # direct-form coefficients (b, a) can overflow for high-order filters.
        self._bp_sos    = self._build_bandpass(low=1.0, high=50.0, order=4)
        self._notch_sos = self._build_notch(freq=notch_freq, Q=30.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, raw_eeg: np.ndarray, meta: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Run the full signal processing pipeline on one EEG epoch.

        Parameters
        ----------
        raw_eeg : np.ndarray, shape (n_channels, n_samples)
            Raw EEG data in µV from EEGStreamer.get_latest_data().
        meta : dict, optional
            Metadata dict from EEGStreamer.get_latest_data() — used only
            to propagate `is_padded` flag into the returned info dict.

        Returns
        -------
        features : np.ndarray, shape (8,)
            [delta_rel, theta_rel, alpha_rel, beta_rel, gamma_rel,
             alpha_left, alpha_right, alpha_asymmetry]

        info : dict
            Intermediate values for debugging and the Vedantic logic layer:
            {
              'band_absolute'  : dict  — µV²/Hz per band (averaged across channels)
              'band_relative'  : dict  — normalised band powers (sum ≈ 1)
              'alpha_left'     : float — mean Alpha power, left hemisphere
              'alpha_right'    : float — mean Alpha power, right hemisphere
              'alpha_asymmetry': float — right − left (Swara index)
              'gamma_spike'    : bool  — True if Gamma relative power > threshold
              'is_padded'      : bool  — True if epoch was zero-padded
            }
        """
        is_padded = (meta or {}).get("is_padded", False)

        # ── Step 1: Bandpass filter (1–50 Hz) ─────────────────────────
        # sosfiltfilt is zero-phase (applies the filter forward and backward),
        # which preserves the exact timing of EEG peaks — critical for
        # event-related potential research but also good practice here.
        filtered = sosfiltfilt(self._bp_sos, raw_eeg, axis=1)

        # ── Step 2: Notch filter (50 or 60 Hz) ────────────────────────
        # Power-line interference creates a sharp sinusoidal artefact.
        # A notch filter (narrow band-stop) removes it without distorting
        # the surrounding spectrum.
        filtered = sosfiltfilt(self._notch_sos, filtered, axis=1)

        # ── Step 3: Welch PSD — all channels ──────────────────────────
        # freqs: 1-D array of frequency bins (Hz)
        # psd  : 2-D array (n_channels × n_freq_bins) in µV²/Hz
        freqs, psd = welch(
            filtered,
            fs         = self._sr,
            nperseg    = self._nperseg,
            noverlap   = self._nperseg // 2,  # 50% overlap between sub-segments
            axis       = 1,
        )

        # ── Step 4: Integrate PSD within each canonical band ──────────
        # For each band, find the frequency indices that fall within [low, high]
        # and sum (trapezoid-integrate) the PSD across those bins.
        # We average across ALL EEG channels to get a scalar per band.
        band_abs = {}
        # np.trapezoid is the NumPy 2.x name; np.trapz was removed in 2.0.
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        for band_name, (lo, hi) in BANDS.items():
            mask          = (freqs >= lo) & (freqs <= hi)
            power_per_ch  = _trapz(psd[:, mask], freqs[mask], axis=1)  # (n_ch,)
            band_abs[band_name] = float(np.mean(power_per_ch))

        # ── Step 5: Relative power normalisation ──────────────────────
        # Absolute power varies hugely between individuals and sessions
        # (impedance, headset fit, skull thickness). Relative power removes
        # this inter-session variability and is the standard input for
        # EEG-based classifiers.
        total     = sum(band_abs.values()) or 1e-10  # guard against all-zero
        band_rel  = {k: v / total for k, v in band_abs.items()}

        # ── Step 6: Hemispheric Alpha Asymmetry ───────────────────────
        # Compute Alpha power separately for left and right electrodes.
        #
        # Neuroscientific basis (Frontal Alpha Asymmetry):
        #   - Left frontal Alpha ↑  → right-hemisphere activation (approach,
        #     positive affect, Pingala / sympathetic tendency)
        #   - Right frontal Alpha ↑ → left-hemisphere activation (withdrawal,
        #     parasympathetic rest, Ida tendency)
        #
        # Swara correlation: higher RIGHT Alpha → Ida (lunar/calm);
        #                    higher LEFT  Alpha → Pingala (solar/active).
        alpha_mask = (freqs >= BANDS["alpha"][0]) & (freqs <= BANDS["alpha"][1])

        def _mean_alpha(ch_indices: List[int]) -> float:
            """Mean Alpha absolute power across specified electrode subset."""
            valid = [i for i in ch_indices if i < psd.shape[0]]
            if not valid:
                return 0.0
            power_per_ch = _trapz(psd[valid][:, alpha_mask], freqs[alpha_mask], axis=1)
            return float(np.mean(power_per_ch))

        alpha_left   = _mean_alpha(self._left_idx)
        alpha_right  = _mean_alpha(self._right_idx)

        # Asymmetry index: positive → right dominant (Ida);
        #                  negative → left dominant (Pingala);
        #                  near zero → Sushumna (balanced).
        alpha_asymmetry = alpha_right - alpha_left

        # ── Step 7: Gamma spike detection ─────────────────────────────
        # A sudden surge in relative Gamma power (>30 Hz) may correlate
        # with high-frequency binding events, associated in traditional
        # frameworks with Tattva activation moments.
        GAMMA_SPIKE_THRESHOLD = 0.12  # >12% of total power in Gamma
        gamma_spike = band_rel["gamma"] > GAMMA_SPIKE_THRESHOLD

        # ── Assemble feature vector ────────────────────────────────────
        features = np.array([
            band_rel["delta"],
            band_rel["theta"],
            band_rel["alpha"],
            band_rel["beta"],
            band_rel["gamma"],
            alpha_left  / (total or 1e-10),   # normalise hemispheric powers too
            alpha_right / (total or 1e-10),
            alpha_asymmetry / (total or 1e-10),
        ], dtype=np.float64)

        info = {
            "band_absolute":   band_abs,
            "band_relative":   band_rel,
            "alpha_left":      alpha_left,
            "alpha_right":     alpha_right,
            "alpha_asymmetry": alpha_asymmetry,
            "gamma_spike":     gamma_spike,
            "is_padded":       is_padded,
        }

        return features, info

    # ------------------------------------------------------------------
    # Filter builders
    # ------------------------------------------------------------------

    def _build_bandpass(
        self, low: float, high: float, order: int = 4
    ) -> np.ndarray:
        """
        Design a Butterworth bandpass filter and return second-order sections.

        A Butterworth filter has maximally flat magnitude response in the
        passband (no ripple), which is important so that we don't
        artificially boost any EEG frequency over another.
        """
        nyq  = self._sr / 2.0
        sos  = butter(
            order,
            [low / nyq, high / nyq],
            btype   = "bandpass",
            output  = "sos",
        )
        return sos

    def _build_notch(self, freq: float, Q: float = 30.0) -> np.ndarray:
        """
        Design an IIR notch (band-stop) filter centred at `freq` Hz.

        Q controls the bandwidth: higher Q → narrower notch.
        Q=30 removes the power-line fundamental while leaving adjacent
        neural signal intact.
        """
        nyq = self._sr / 2.0
        b, a = iirnotch(freq / nyq, Q)
        # Convert to SOS for numerical stability
        from scipy.signal import tf2sos
        sos = tf2sos(b, a)
        return sos


if __name__ == "__main__":
    print("=== FeatureExtractor — Synthetic signal demo ===")

    SR = 256
    T  = 2.0  # seconds
    t  = np.linspace(0, T, int(SR * T), endpoint=False)

    # Synthesise a signal that mimics Ekagra (Alpha-dominant, low Beta)
    # 4 channels: TP9, AF7, AF8, TP10  (Muse 2 layout)
    def make_channel(alpha_amp, beta_amp, noise=0.5):
        return (
            alpha_amp * np.sin(2 * np.pi * 10.0 * t)   # 10 Hz Alpha
          + beta_amp  * np.sin(2 * np.pi * 20.0 * t)   # 20 Hz Beta
          + noise     * np.random.randn(len(t))         # white noise
        )

    raw = np.vstack([
        make_channel(alpha_amp=5.0, beta_amp=1.0),  # TP9  (left)
        make_channel(alpha_amp=4.5, beta_amp=1.0),  # AF7  (left)
        make_channel(alpha_amp=6.0, beta_amp=0.8),  # AF8  (right) — higher alpha
        make_channel(alpha_amp=5.5, beta_amp=0.8),  # TP10 (right)
    ])

    extractor = FeatureExtractor(sample_rate=SR)
    features, info = extractor.extract(raw)

    print(f"Feature vector : {np.round(features, 4)}")
    print(f"Band relative  : {info['band_relative']}")
    print(f"Alpha asymmetry: {info['alpha_asymmetry']:.4f} (positive = Ida)")
    print(f"Gamma spike    : {info['gamma_spike']}")
