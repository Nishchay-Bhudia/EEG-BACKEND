"""
main.py -- Flask REST API Server
=================================
Exposes the EEG analysis pipeline as HTTP endpoints so a Vercel frontend
(or any client) can POST raw EEG data and receive a live Chitta Bhumi /
Swara / Tattva / Triguna analysis in return.

Endpoints
---------
GET  /status          -- health check + model-ready flag
POST /analyze         -- analyze one EEG epoch; returns full classification
POST /analyze/bands   -- analyze pre-computed band powers (lightweight path)

Usage
-----
    python run.py                        # starts on PORT env var (default 5000)
    python -m neuro_yogic.main           # same

Vercel Frontend -> POST /analyze
---------------------------------
Send a JSON body with 2 seconds of raw EEG from the headband:

    {
        "eeg_data":    [[ch0_s0, ch0_s1, ...], [ch1_s0, ...]],  // (n_channels x n_samples)
        "sample_rate": 256
    }

Or, if your frontend already computes band powers (e.g. via muse-js):

    POST /analyze/bands
    {
        "delta": 0.12, "theta": 0.18, "alpha": 0.40,
        "beta": 0.20, "gamma": 0.10,
        "alpha_left": 0.20, "alpha_right": 0.25
    }

Both endpoints return the same JSON response shape, now including a
"gunas" block with Sattva / Rajas / Tamas percentages.

CORS
----
All origins are allowed so the Vercel frontend can call this freely.
Restrict CORS_ORIGINS in production by setting the environment variable.
"""

import asyncio
import logging
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from neuro_yogic.data_generator import DEFAULT_MODEL_PATH, save_dataset
from neuro_yogic.feature_extractor import FeatureExtractor
from neuro_yogic.satva_classifier import classify_gunas          # ← NEW
from neuro_yogic.vedantic_logic import vedantic_analyze
from neuro_yogic.yoga_classifier import YogaClassifier

# ── App setup ─────────────────────────────────────────────────────────
app = Flask(__name__)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Global model (loaded once at startup) ─────────────────────────────
_classifier: YogaClassifier = YogaClassifier(n_estimators=200)
_model_ready: bool = False
_model_lock = threading.Lock()


def _startup_training() -> None:
    """Train (or load) the classifier in a background thread at startup."""
    global _model_ready
    log.info("[Startup] Checking for pre-trained model ...")
    with _model_lock:
        try:
            if os.path.exists(DEFAULT_MODEL_PATH):
                _classifier.load()
                log.info("[Startup] Pre-trained model loaded.")
            else:
                log.info("[Startup] No saved model — generating dataset and training ...")
                csv_path = save_dataset()
                _classifier.train_model(csv_path)
                _classifier.save()
                log.info("[Startup] Training complete.")
            _model_ready = True
        except Exception as exc:
            log.error(f"[Startup] Training failed: {exc}")


def _build_response(chitta: str, probs: dict, info: dict) -> dict:
    """Assemble the standard API response JSON."""
    reading  = vedantic_analyze(info, chitta_bhumi=chitta)
    band_rel = info.get("band_relative", {})

    # ── Trigunas ────────────────────────────────────────────────────
    # classify_gunas() accepts the relative band-power dict and the
    # current Chitta Bhumi state.  It returns a dict with keys:
    #   sattva, rajas, tamas  (floats that sum to 1.0)
    #   label                 ("Sattvic" / "Rajasic" / "Tamasic" / "Balanced")
    #   note                  (short human-readable interpretation)
    gunas = classify_gunas(band_rel, chitta_bhumi=chitta)          # ← NEW

    return {
        "chitta_bhumi": {
            "state":         chitta,
            "confidence":    f"{max(probs.values()) * 100:.1f}%",
            "probabilities": {k: round(v * 100, 1) for k, v in probs.items()},
        },
        "swara":       reading.to_dict()["swara"],
        "tattva":      reading.tattva_flags or ["No active Tattva flags this epoch"],
        "depth":       reading.contemplative_depth,
        "eeg_spectrum": {k: round(v * 100, 2) for k, v in band_rel.items()},
        "hemispheric_asymmetry": {
            "index":     round(info.get("alpha_asymmetry", 0), 6),
            "direction": (
                "Right > Left (Ida)"     if info.get("alpha_asymmetry", 0) > 0 else
                "Left > Right (Pingala)" if info.get("alpha_asymmetry", 0) < 0 else
                "Balanced (Sushumna)"
            ),
        },
        "is_padded": info.get("is_padded", False),
        "gunas":     gunas,                                         # ← NEW
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """Health check — returns whether the ML model is ready."""
    return jsonify({
        "status":      "ok",
        "model_ready": _model_ready,
        "message":     "Model is ready." if _model_ready else "Model is still loading — try again in a few seconds.",
    })


@app.post("/analyze")
def analyze():
    """
    Analyze one epoch of raw EEG data from the headband.

    Request body (JSON)
    -------------------
    eeg_data    : list[list[float]]  -- shape (n_channels, n_samples), raw µV values
    sample_rate : int                -- samples per second (e.g. 256 for Muse 2)

    Returns
    -------
    JSON with chitta_bhumi, swara, tattva, depth, eeg_spectrum,
    hemispheric_asymmetry, and gunas (Sattva/Rajas/Tamas).
    """
    if not _model_ready:
        return jsonify({"error": "Model is still loading. Try again in a few seconds."}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    eeg_data = body.get("eeg_data")
    sample_rate = int(body.get("sample_rate", 256))

    if eeg_data is None:
        return jsonify({"error": "Missing 'eeg_data' field."}), 400

    try:
        raw_eeg = np.array(eeg_data, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid eeg_data: {exc}"}), 400

    if raw_eeg.ndim != 2 or raw_eeg.shape[0] < 1 or raw_eeg.shape[1] < 2:
        return jsonify({"error": "eeg_data must be a 2-D array (n_channels x n_samples) with at least 2 samples."}), 400

    try:
        extractor = FeatureExtractor(sample_rate=sample_rate)
        meta = {"sample_rate": sample_rate, "is_padded": False}
        features, info = extractor.extract(raw_eeg, meta)
    except Exception as exc:
        log.exception("Feature extraction failed")
        return jsonify({"error": f"Feature extraction failed: {exc}"}), 500

    if np.all(features == 0):
        return jsonify({"error": "All-zero signal — check electrode contact and headband connection."}), 422

    try:
        with _model_lock:
            chitta = _classifier.predict(features)
            probs  = _classifier.predict_proba(features)
    except Exception as exc:
        log.exception("Classification failed")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    return jsonify(_build_response(chitta, probs, info))


@app.post("/analyze/bands")
def analyze_bands():
    """
    Analyze pre-computed band powers (e.g. from muse-js or BrainFlow on the client).

    Request body (JSON)
    -------------------
    delta, theta, alpha, beta, gamma : float  -- relative band powers (0-1)
    alpha_left, alpha_right           : float  -- hemispheric alpha powers (optional)

    All values should be relative powers that sum to ~1.
    """
    if not _model_ready:
        return jsonify({"error": "Model is still loading. Try again in a few seconds."}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    required = ["delta", "theta", "alpha", "beta", "gamma"]
    missing  = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        delta = float(body["delta"])
        theta = float(body["theta"])
        alpha = float(body["alpha"])
        beta  = float(body["beta"])
        gamma = float(body["gamma"])
        alpha_left  = float(body.get("alpha_left",  alpha / 2))
        alpha_right = float(body.get("alpha_right", alpha / 2))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid band power value: {exc}"}), 400

    asymmetry = alpha_right - alpha_left
    total     = delta + theta + alpha + beta + gamma or 1e-10

    features = np.array([
        delta / total, theta / total, alpha / total,
        beta  / total, gamma / total,
        alpha_left  / total,
        alpha_right / total,
        asymmetry   / total,
    ], dtype=np.float64)

    info = {
        "band_relative":   {"delta": delta/total, "theta": theta/total, "alpha": alpha/total,
                            "beta": beta/total, "gamma": gamma/total},
        "alpha_left":      alpha_left,
        "alpha_right":     alpha_right,
        "alpha_asymmetry": asymmetry,
        "gamma_spike":     (gamma / total) > 0.12,
        "is_padded":       False,
    }

    try:
        with _model_lock:
            chitta = _classifier.predict(features)
            probs  = _classifier.predict_proba(features)
    except Exception as exc:
        log.exception("Classification failed")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    return jsonify(_build_response(chitta, probs, info))


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Start the Flask server. Model training runs in a background thread."""
    port = int(os.environ.get("PORT", 5000))

    training_thread = threading.Thread(target=_startup_training, daemon=True)
    training_thread.start()

    log.info(f"[Server] Starting on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
