"""
main.py -- Flask REST API Server
=================================
Exposes the EEG analysis pipeline as HTTP endpoints so a Vercel frontend
(or any client) can POST raw EEG data and receive a live Chitta Bhumi /
Swara / Tattva / Triguna analysis in return.

Endpoints
---------
GET  /status          -- health check + model-ready flag
POST /analyze         -- analyze one EEG epoch (raw data); returns full classification
POST /analyze/bands   -- analyze pre-computed band powers (lightweight path)

Usage (local dev)
-----------------
  python run.py                  # starts on PORT env var (default 5000)
  python -m neuro_yogic.main     # same

Production (Render)
-------------------
  gunicorn -w 1 -b 0.0.0.0:$PORT "neuro_yogic.main:app"
  (see render.yaml — single worker so all requests share the in-memory ML model)

Vercel Frontend -> POST /analyze
---------------------------------
Send a JSON body with 2 seconds of raw EEG from the headband:

  {
    "eeg_data": [[ch0_s0, ch0_s1, ...], [ch1_s0, ...]],  // (n_channels x n_samples)
    "sample_rate": 256,
    "blood_oxygen": 98.5,   // optional — pass if headband provides it
    "heart_rate": 72.0      // optional — pass if headband provides it
  }

Or, if your frontend already computes band powers (e.g. via muse-js):

  POST /analyze/bands
  {
    "delta": 0.12, "theta": 0.18, "alpha": 0.40,
    "beta": 0.20, "gamma": 0.10,
    "alpha_left": 0.20, "alpha_right": 0.25,
    "blood_oxygen": 98.5,   // optional
    "heart_rate": 72.0      // optional
  }

Both endpoints return the same JSON response shape, including a "gunas" block
with Sattva / Rajas / Tamas percentages, and optionally blood_oxygen /
heart_rate if those were provided in the request.

CORS
----
All origins are allowed by default so the Vercel frontend can call freely.
Restrict in production by setting the CORS_ORIGINS environment variable to a
comma-separated list of allowed origins, e.g.:
  CORS_ORIGINS=https://your-app.vercel.app
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
# FIX: supports_credentials must be False when origins="*" (CORS spec requirement).
# The frontend does not send credentials to this backend, so False is correct.
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}}, supports_credentials=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Global model (loaded once at startup) ──────────────────────────────────────
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
    """Assemble the standard API response JSON."""
    reading = vedantic_analyze(info, chitta_bhumi=chitta)
    band_rel = info.get("band_relative", {})

    gunas = classify_gunas(info)

    resp = {
        "chitta_bhumi": {
            "state": chitta,
            "depth": reading.get("contemplative_depth", "Surface"),
            "confidence": probs.get(chitta, "—"),
            "probabilities": probs,
        },
        "swara": {
            "state": reading.get("swara", {}).get("state", "—"),
            "confidence": reading.get("swara", {}).get("confidence", "—"),
            "note": reading.get("swara", {}).get("note", ""),
        },
        "depth": reading.get("contemplative_depth", "Surface"),
        "tattva_flags": reading.get("tattva_flags", []),
        "eeg_spectrum": band_rel,
        "band_relative": band_rel,
        "hemispheric_asymmetry": {
            "asymmetry": info.get("alpha_asymmetry", 0),
            "alpha_left": info.get("alpha_left", 0),
            "alpha_right": info.get("alpha_right", 0),
        },
        "gunas": gunas,
        "is_padded": info.get("is_padded", False),
    }

    # Only include vitals if the device actually reported them
    if blood_oxygen is not None:
        resp["blood_oxygen"] = _safe_float(blood_oxygen)
    if heart_rate is not None:
        resp["heart_rate"] = _safe_float(heart_rate)

    return resp


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """Health check — returns whether the ML model is ready.
    
    FIX: Added 'board' field that the frontend test button expects.
    """
    return jsonify({
        "status": "ok",
        "model_ready": _model_ready,
        "board": "web-bluetooth",      # FIX: frontend shows data.board in test button
        "version": "2.0",
        "message": "Model is ready." if _model_ready else "Model is still loading — try again in a few seconds.",
    })


@app.post("/analyze")
def analyze():
    """
    Analyze one epoch of raw EEG data from the headband.

    Request body (JSON)
    -------------------
    eeg_data    : list[list[float]] -- (n_channels x n_samples), raw values
    sample_rate : int               -- samples per second (e.g. 256 for Muse 2/S)
    blood_oxygen: float | null      -- optional SpO2 % from headband
    heart_rate  : float | null      -- optional HR BPM from headband

    Returns
    -------
    JSON with chitta_bhumi, swara, tattva_flags, depth, eeg_spectrum,
    band_relative, hemispheric_asymmetry, gunas, and optionally
    blood_oxygen / heart_rate.
    """
    if not _model_ready:
        return jsonify({"error": "Model is still loading. Try again in a few seconds."}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    eeg_data = body.get("eeg_data")
    sample_rate = int(body.get("sample_rate", 256))
    blood_oxygen = body.get("blood_oxygen")
    heart_rate = body.get("heart_rate")

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
            probs = _classifier.predict_proba(features)
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
    delta, theta, alpha, beta, gamma : float -- relative band powers (0-1, sum ~1)
    alpha_left, alpha_right          : float -- hemispheric alpha (optional)
    blood_oxygen                     : float | null -- optional SpO2 %
    heart_rate                       : float | null -- optional HR BPM
    """
    if not _model_ready:
        return jsonify({"error": "Model is still loading. Try again in a few seconds."}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    required = ["delta", "theta", "alpha", "beta", "gamma"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        delta = float(body["delta"])
        theta = float(body["theta"])
        alpha = float(body["alpha"])
        beta = float(body["beta"])
        gamma = float(body["gamma"])
        alpha_left = float(body.get("alpha_left", alpha / 2))
        alpha_right = float(body.get("alpha_right", alpha / 2))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid band power value: {exc}"}), 400

    blood_oxygen = body.get("blood_oxygen")
    heart_rate = body.get("heart_rate")

    asymmetry = alpha_right - alpha_left
    total = delta + theta + alpha + beta + gamma or 1e-10

    features = np.array([
        delta / total, theta / total, alpha / total,
        beta / total, gamma / total,
        alpha_left / total,
        alpha_right / total,
        asymmetry / total,
    ], dtype=np.float64)

    info = {
        "band_relative": {
            "delta": delta / total, "theta": theta / total, "alpha": alpha / total,
            "beta": beta / total, "gamma": gamma / total,
        },
        "alpha_left": alpha_left,
        "alpha_right": alpha_right,
        "alpha_asymmetry": asymmetry,
        "gamma_spike": (gamma / total) > 0.12,
        "is_padded": False,
    }

    try:
        with _model_lock:
            chitta = _classifier.predict(features)
            probs = _classifier.predict_proba(features)
    except Exception as exc:
        log.exception("Classification failed")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    return jsonify(_build_response(chitta, probs, info, blood_oxygen, heart_rate))


# ── Entry point (local dev only — production uses gunicorn) ───────────────────

def main() -> None:
    """Start the Flask dev server (local dev only). Training is already running from module init."""
    port = int(os.environ.get("PORT", 5000))
    log.info(f"[Server] Starting on port {port} ...")
    # debug=False is important — debug mode causes a reload that starts a second training thread
    app.run(host="0.0.0.0", port=port, debug=False)


# Module-level: start training as soon as gunicorn (or anything) imports this module.
# Gunicorn workers import the module — this thread starts once per worker process.
# Direct-run (python run.py) also imports the module, so training starts before app.run().
_training_thread = threading.Thread(target=_startup_training, daemon=True, name="model-training")
_training_thread.start()

if __name__ == "__main__":
    main()
