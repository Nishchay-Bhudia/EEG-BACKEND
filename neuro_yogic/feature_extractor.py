"""
feature_extractor.py
====================
Signal processing pipeline (per 2-second EEG epoch):

  Step 1 -- Butterworth bandpass filter (1-50 Hz)
    Removes DC drift (electrode polarisation) and high-frequency EMG
    (muscle) artefacts. Zero-phase (forward + backward pass) preserves
    the exact timing of neural peaks.

  Step 2 -- IIR notch filter (50 or 60 Hz)
    Removes AC power-line interference. Q=30 narrows the notch so only
    the interference frequency is attenuated, leaving surrounding EEG
    bands intact.

  Step 3 -- Welch's Power Spectral Density
    Divides the window into 50%-overlapping sub-segments, computes a
    periodogram for each, and averages them. This dramatically reduces
    PSD variance vs. a single FFT.

  Step 4 -- Band power integration (trapezoid rule)
    Integrates PSD within each canonical EEG band and averages across
    all EEG channels to produce one scalar per band.

  Step 5 -- Relative power normalisation
    Divides each absolute band power by the total power across all bands.
    Relative powers are session- and person-invariant.

  Step 6 -- Hemispheric Alpha Asymmetry
    Asymmetry = right_alpha - left_alpha.
    Positive -> right hemisphere Alpha dominant -> Ida (parasympathetic).
    Negative -> left hemisphere Alpha dominant  -> Pingala (sympathetic).
    Near zero -> Sushumna (balanced).

Electrode layout (default: 4-channel Muse 2):
  Left  channels: indices 0, 1  (TP9, AF7)
  Right channels: indices 2, 3  (AF8, TP10)
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

EEG_BANDS = {
    "delta": (1.0,   4.0),
    "theta": (4.0,   8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

# NumPy 2.x renamed np.trapz -> np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class FeatureExtractor:
    """Transforms raw multi-channel EEG into the 8-D classifier feature vector."""

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
        self._nperseg   = sample_rate // 2  # 0.5-second Welch segments

        # Pre-compute filter coefficients (done once; not per epoch)
        nyq = sample_rate / 2.0

        # Butterworth bandpass (1-50 Hz), order 4, in second-order sections
        self._bp_sos = butter(4, [1.0 / nyq, 50.0 / nyq], btype="bandpass", output="sos")

        # IIR notch (power-line interference)
        b, a = iirnotch(notch_freq / nyq, Q=30.0)
        self._notch_sos = tf2sos(b, a)

    def extract(
        self,
        raw_eeg: np.ndarray,
        meta: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Run the full pipeline on one EEG epoch.

        Parameters
        ----------
        raw_eeg : ndarray (n_channels, n_samples)  -- raw uV data
        meta    : dict from EEGStreamer.get_latest_data()

        Returns
        -------
        features : ndarray (8,)
        info     : dict with band powers, asymmetry, gamma_spike, is_padded
        """
        is_padded = (meta or {}).get("is_padded", False)

        # Step 1: Bandpass filter
        filtered = sosfiltfilt(self._bp_sos, raw_eeg, axis=1)

        # Step 2: Notch filter
        filtered = sosfiltfilt(self._notch_sos, filtered, axis=1)

        # Step 3: Welch PSD
        # freqs: (n_freq_bins,)   psd: (n_channels, n_freq_bins)
        freqs, psd = welch(
            filtered,
            fs       = self._sr,
            nperseg  = self._nperseg,
            noverlap = self._nperseg // 2,
            axis     = 1,
        )

        # Step 4: Band power integration
        band_abs = {}
        for band_name, (lo, hi) in EEG_BANDS.items():
            mask         = (freqs >= lo) & (freqs <= hi)
            power_per_ch = _trapz(psd[:, mask], freqs[mask], axis=1)
            band_abs[band_name] = float(np.mean(power_per_ch))

        # Step 5: Relative power normalisation
        total    = sum(band_abs.values()) or 1e-10
        band_rel = {k: v / total for k, v in band_abs.items()}

        # Step 6: Hemispheric Alpha Asymmetry
        alpha_mask = (freqs >= EEG_BANDS["alpha"][0]) & (freqs <= EEG_BANDS["alpha"][1])

        def _mean_alpha(ch_idx: List[int]) -> float:
            valid = [i for i in ch_idx if i < psd.shape[0]]
            if not valid:
                return 0.0
            return float(np.mean(_trapz(psd[valid][:, alpha_mask], freqs[alpha_mask], axis=1)))

        alpha_left  = _mean_alpha(self._left_idx)
        alpha_right = _mean_alpha(self._right_idx)
        asymmetry   = alpha_right - alpha_left

        gamma_spike = band_rel["gamma"] > 0.12  # >12% of total power

        features = np.array([
            band_rel["delta"],
            band_rel["theta"],
            band_rel["alpha"],
            band_rel["beta"],
            band_rel["gamma"],
            alpha_left  / total,
            alpha_right / total,
            asymmetry   / total,
        ], dtype=np.float64)

        info = {
            "band_absolute":   band_abs,
            "band_relative":   band_rel,
            "alpha_left":      alpha_left,
            "alpha_right":     alpha_right,
            "alpha_asymmetry": asymmetry,
            "gamma_spike":     gamma_spike,
            "is_padded":       is_padded,
        }
        return features, info
