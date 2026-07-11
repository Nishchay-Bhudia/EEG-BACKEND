"""
main.py -- Flask REST API Server  (v2 — paper-aligned classifier)
=================================================================
Exposes the EEG analysis pipeline as HTTP endpoints.

Endpoints
---------
GET  /status          -- health check + model-ready flag
POST /analyze         -- analyze one EEG epoch (raw data); returns full analysis
POST /analyze/bands   -- analyze pre-computed band powers (lightweight path)

Key changes in v2:
  • YogaClassifier is now rule-based (no training needed) — /status is
    immediately ready on startup.
  • /analyze uses classify_from_info(info) so the rule-based scorer sees
    the full feature set (faa, plv, high_beta, low_beta) from FeatureExtractor.
  • /analyze/bands accepts optional high_beta / low_beta / faa / plv fields.
    If high_beta is absent, total beta is split 55/45 (low/high) as an estimate.
  • _build_response includes faa / plv / alpha_asymmetry in the response.

Key changes in v3:
  • Both /analyze and /analyze/bands accept optional `baseline` (personal
    calibration) and `recent_probs` (temporal smoothing history) — see
    yoga_classifier.py.
  • /analyze now actually rejects (422) artifact-contaminated epochs
    instead of only reporting a padding flag — see
    FeatureExtractor._detect_artifacts.
  • Chitta Bhumi band edges (theta/alpha/low_beta) are IAF-anchored per
    epoch rather than fixed population ranges — see
    FeatureExtractor._detect_iaf.
  • Gunas no longer take FAA/Swara as input — see satva_classifier.py.

CORS
----
All origins allowed by default. Set CORS_ORIGINS env var to restrict.
"""

import logging
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from neuro_yogic.data_generator import DEFAULT_MODEL_PATH, save_dataset
from neuro_yogic.feature_extractor import FeatureExtractor
from neuro_yogic.satva_classifier import classify_gunas
from neuro_yogic.vedantic_logic import vedantic_analyze
from neuro_yogic.yoga_classifier import YogaClassifier

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}}, supports_credentials=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Global classifier ──────────────────────────────────────────────────────────
# YogaClassifier v2 is rule-based — no training required.
# It is always "ready" immediately on import.
_classifier: YogaClassifier = YogaClassifier()
_model_ready: bool = False
_model_lock = threading.Lock()


def _startup_training() -> None:
    """
    'Training' is now a no-op for the rule-based classifier.
    This function marks the model ready immediately so /status returns
    model_ready=true within seconds of startup (no 30-second wait).
    """
    global _model_ready
    log.info("[Startup] YogaClassifier v2 — rule-based, no training required.")
    with _model_lock:
        try:
            # Rule-based classifier: train_model is a no-op that returns mock metrics
            metrics = _classifier.train_model("")
            log.info(f"[Startup] Classifier ready: {metrics.get('method', 'rule-based')}")
            _model_ready = True
        except Exception as exc:
            log.error(f"[Startup] Classifier init failed: {exc}")
            _model_ready = True  # mark ready anyway — rule-based always works


def _safe_float(value):
    """Convert a value to float, returning None if it is None or invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_response(chitta: str, probs: dict, info: dict,
                    blood_oxygen=None, heart_rate=None) -> dict:
    """
    Assemble the standard API response JSON.

    FIX (v1 → v2):
    • VedanticReading is a @dataclass — use reading.to_dict() not .get().
    • Gunas are computed inside vedantic_analyze; no separate classify_gunas call.
    • Now includes faa, plv, and new band names (high_beta, low_beta) in response.
    """
    reading = vedantic_analyze(info, chitta_bhumi=chitta)
    band_rel = info.get("band_relative", {})
    vedantic = reading.to_dict()

    resp = {
        "chitta_bhumi": {
            "state":         chitta,
            "depth":         reading.contemplative_depth,
            "confidence":    probs.get(chitta, "—"),
            "probabilities": probs,
        },
        "swara":         vedantic["swara"],
        "depth":         reading.contemplative_depth,
        "tattva_flags":  reading.tattva_flags,
        # eeg_spectrum and band_relative both include all bands: delta, theta,
        # alpha, low_beta, high_beta, gamma (and legacy "beta" = low+high)
        "eeg_spectrum":  band_rel,
        "band_relative": band_rel,
        "hemispheric_asymmetry": {
            "asymmetry":   info.get("alpha_asymmetry", 0),
            "faa":         info.get("faa", 0),          # ln-ratio FAA (v2 new)
            "plv":         info.get("plv", 0.5),         # PLV coherence (v2 new)
            "alpha_left":  info.get("alpha_left",  0),
            "alpha_right": info.get("alpha_right", 0),
        },
        "gunas":     vedantic["gunas"],
        "is_padded": info.get("is_padded", False),
        "iaf":          info.get("iaf"),
        "iaf_detected": info.get("iaf_detected", False),
        "signal_quality": {
            "fraction_flagged": info.get("artifact", {}).get("fraction_flagged", 0.0),
            "channels_flagged": info.get("artifact", {}).get("channels_flagged", []),
        },
    }
    if blood_oxygen is not None:
        resp["blood_oxygen"] = _safe_float(blood_oxygen)
    if heart_rate is not None:
        resp["heart_rate"] = _safe_float(heart_rate)

    return resp


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """Health check — returns whether the classifier is ready."""
    return jsonify({
        "status":      "ok",
        "model_ready": _model_ready,
        "board":       "web-bluetooth",
        "version":     "2.0",
        "classifier":  "rule-based (paper-derived thresholds)",
        "message":     "Ready." if _model_ready else "Initialising — try again in a moment.",
    })


@app.post("/analyze")
def analyze():
    """
    Analyze one epoch of raw EEG data from the headband.

    Request body (JSON)
    -------------------
    eeg_data     : list[list[float]] -- (n_channels × n_samples), raw µV values
    sample_rate  : int               -- samples/sec (256 for Muse 2 / Muse S)
    blood_oxygen : float | null      -- optional SpO2 %
    heart_rate   : float | null      -- optional HR BPM (from Muse S PPG)
    baseline     : dict | null       -- optional personal calibration, per band:
                                        {"delta": {"mean":.., "std":..}, ...} —
                                        see yoga_classifier._apply_baseline.
    recent_probs : list | null       -- optional last few epochs' probability
                                        dicts (oldest→newest) for temporal
                                        smoothing — see yoga_classifier._smooth_probs.

    Returns
    -------
    JSON with chitta_bhumi, swara, tattva_flags, depth, eeg_spectrum (all 6
    bands including high_beta), band_relative, hemispheric_asymmetry (+ faa/plv),
    gunas (sattva/rajas/tamas percentages), and optionally blood_oxygen/heart_rate.

    422 is returned (not a classification) when the epoch is too contaminated
    by artifacts to trust — see FeatureExtractor._detect_artifacts. Artifact
    detection is a real gate here, not just an informational flag.
    """
    if not _model_ready:
        return jsonify({"error": "Classifier initialising. Try again in a few seconds."}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    eeg_data     = body.get("eeg_data")
    sample_rate  = int(body.get("sample_rate", 256))
    blood_oxygen = body.get("blood_oxygen")
    heart_rate   = body.get("heart_rate")
    baseline     = body.get("baseline") if isinstance(body.get("baseline"), dict) else None
    recent_probs = body.get("recent_probs") if isinstance(body.get("recent_probs"), list) else None

    if eeg_data is None:
        return jsonify({"error": "Missing 'eeg_data' field."}), 400

    try:
        raw_eeg = np.array(eeg_data, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid eeg_data: {exc}"}), 400

    if raw_eeg.ndim != 2 or raw_eeg.shape[0] < 1 or raw_eeg.shape[1] < 2:
        return jsonify({"error": "eeg_data must be 2-D (n_channels × n_samples) with ≥2 samples."}), 400

    try:
        extractor = FeatureExtractor(sample_rate=sample_rate)
        meta = {"sample_rate": sample_rate, "is_padded": False}
        features, info = extractor.extract(raw_eeg, meta)
    except Exception as exc:
        log.exception("Feature extraction failed")
        return jsonify({"error": f"Feature extraction failed: {exc}"}), 500

    if np.all(features == 0):
        return jsonify({"error": "All-zero signal — check electrode contact and headband connection."}), 422

    artifact = info.get("artifact", {})
    if artifact.get("detected"):
        return jsonify({
            "error": "Signal quality too poor to classify — artifacts detected on "
                     f"{len(artifact.get('channels_flagged', []))} channel(s).",
            "artifact": artifact,
        }), 422

    try:
        with _model_lock:
            # v2: classify_from_info uses the full info dict (faa, plv, high_beta, etc.)
            chitta, probs = _classifier.classify_from_info(
                info, baseline=baseline, recent_probs=recent_probs
            )
    except Exception as exc:
        log.exception("Classification failed")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    return jsonify(_build_response(chitta, probs, info, blood_oxygen, heart_rate))


@app.post("/analyze/bands")
def analyze_bands():
    """
    Analyze pre-computed band powers (e.g. from muse-js or BrainFlow on the client).

    Request body (JSON)
    -------------------
    delta, theta, alpha : float -- relative band powers (0-1)
    beta                : float -- total beta (13-30 Hz); split 55/45 if high_beta absent
    gamma               : float -- relative gamma power
    high_beta           : float -- (optional) HIGH beta (18-30 Hz) — the Rajas marker.
                                   If omitted, estimated as beta * 0.45.
    low_beta            : float -- (optional) LOW beta (13-18 Hz).
                                   If omitted, estimated as beta * 0.55.
    alpha_left          : float -- (optional) left-hemisphere alpha
    alpha_right         : float -- (optional) right-hemisphere alpha
    faa                 : float -- (optional) pre-computed FAA = ln(α_right)−ln(α_left)
    plv                 : float -- (optional) pre-computed PLV (0-1)
    blood_oxygen        : float | null -- optional SpO2 %
    heart_rate          : float | null -- optional HR BPM
    baseline            : dict | null  -- optional personal calibration (see /analyze)
    recent_probs        : list | null  -- optional recent-epoch history (see /analyze)
    """
    if not _model_ready:
        return jsonify({"error": "Classifier initialising. Try again in a few seconds."}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    required = ["delta", "theta", "alpha", "beta", "gamma"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        delta  = float(body["delta"])
        theta  = float(body["theta"])
        alpha  = float(body["alpha"])
        beta   = float(body["beta"])
        gamma  = float(body["gamma"])

        # High/low beta split — use provided values or estimate from total beta
        high_beta = float(body.get("high_beta", beta * 0.45))
        low_beta  = float(body.get("low_beta",  beta * 0.55))

        alpha_left  = float(body.get("alpha_left",  alpha / 2))
        alpha_right = float(body.get("alpha_right", alpha / 2))

        # FAA — use provided value or compute from alpha_left/right
        if "faa" in body:
            faa = float(body["faa"])
        else:
            try:
                faa = float(np.log(max(alpha_right, 1e-12)) - np.log(max(alpha_left, 1e-12)))
            except Exception:
                faa = 0.0

        plv = float(body.get("plv", 0.50))

    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid band power value: {exc}"}), 400

    blood_oxygen = body.get("blood_oxygen")
    heart_rate   = body.get("heart_rate")
    baseline     = body.get("baseline") if isinstance(body.get("baseline"), dict) else None
    recent_probs = body.get("recent_probs") if isinstance(body.get("recent_probs"), list) else None

    # Normalise so all 6 bands sum to 1
    total = delta + theta + alpha + high_beta + low_beta + gamma or 1e-10
    band_rel = {
        "delta":     delta     / total,
        "theta":     theta     / total,
        "alpha":     alpha     / total,
        "low_beta":  low_beta  / total,
        "high_beta": high_beta / total,
        "gamma":     gamma     / total,
        "beta":      (low_beta + high_beta) / total,   # legacy combined field
    }

    info = {
        "band_relative":   band_rel,
        "alpha_left":      alpha_left,
        "alpha_right":     alpha_right,
        "alpha_asymmetry": alpha_right - alpha_left,
        "faa":             float(np.clip(faa, -2.0, 2.0)),
        "plv":             float(np.clip(plv, 0.0, 1.0)),
        "gamma_spike":     band_rel["gamma"] > 0.12,
        "is_padded":       False,
    }

    try:
        with _model_lock:
            chitta, probs = _classifier.classify_from_info(
                info, baseline=baseline, recent_probs=recent_probs
            )
    except Exception as exc:
        log.exception("Classification failed")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    return jsonify(_build_response(chitta, probs, info, blood_oxygen, heart_rate))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Start the Flask dev server (local dev only — production uses gunicorn)."""
    port = int(os.environ.get("PORT", 5000))
    log.info(f"[Server] Starting on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)


# Mark ready at module import (rule-based, no blocking training thread needed).
_training_thread = threading.Thread(target=_startup_training, daemon=True, name="model-init")
_training_thread.start()

if __name__ == "__main__":
    main()
