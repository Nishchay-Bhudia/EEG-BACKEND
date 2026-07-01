"""
server.py — FastAPI Web Server for EEG Neuro-Yogic Backend
===========================================================
Wraps the existing inference pipeline (EEGStreamer, FeatureExtractor,
YogaClassifier, VedanticLogic) into an HTTP API consumed by the Vercel UI.

Endpoints
---------
GET  /status        — model readiness check
POST /analyze       — raw EEG (+ optional PPG) → full classification + vitals
POST /analyze/bands — pre-computed band powers → classification

PPG / Biometrics
----------------
The /analyze endpoint optionally accepts PPG data from the headset.
If ppg_ir / ppg_red are omitted or empty, heart_rate and spo2 are
returned as null — the UI will simply show "—" without any error.

Usage (local dev)
-----------------
    uvicorn server:app --host 0.0.0.0 --port 10000 --reload

Usage (Render)
--------------
Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

import os
import time
import asyncio
import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── BrainFlow / inference imports ─────────────────────────────────────────────
try:
    from neuro_yogic.feature_extractor import FeatureExtractor
    from neuro_yogic.yoga_classifier import YogaClassifier, DEFAULT_MODEL_PATH, DEFAULT_LABELS_PATH
    from neuro_yogic import vedantic_logic as vedantic
    from neuro_yogic.data_generator import save_dataset
    NEURO_AVAILABLE = True
except ImportError:
    NEURO_AVAILABLE = False

logger = logging.getLogger("uvicorn.error")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EEG Neuro-Yogic Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────
_clf: Optional[object] = None
_extractor: Optional[object] = None
_model_ready = False
_model_loading = False


async def _load_model():
    """Load or train the YogaClassifier on startup (non-blocking)."""
    global _clf, _extractor, _model_ready, _model_loading
    if not NEURO_AVAILABLE:
        logger.warning("neuro_yogic modules not available — running in stub mode")
        _model_ready = True
        return

    _model_loading = True
    try:
        loop = asyncio.get_event_loop()
        clf = YogaClassifier(n_estimators=200)

        if not os.path.exists(DEFAULT_MODEL_PATH):
            logger.info("[Startup] Generating training data and training model…")
            csv_path = await loop.run_in_executor(
                None, lambda: save_dataset(n_samples_per_class=500)
            )
            await loop.run_in_executor(None, lambda: clf.train_model(csv_path))
            clf.save()
            logger.info("[Startup] Model trained and saved.")
        else:
            logger.info("[Startup] Loading pre-trained model…")
            await loop.run_in_executor(None, clf.load)
            logger.info("[Startup] Model loaded.")

        _clf = clf
        _extractor = FeatureExtractor(sample_rate=256)
        _model_ready = True
    except Exception as exc:
        logger.error(f"[Startup] Model load failed: {exc}")
    finally:
        _model_loading = False


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_load_model())


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    eeg_data: List[List[float]]          # shape: [n_channels][n_samples]
    sample_rate: int = 256
    # Optional PPG channels — null / empty list → vitals not computed
    ppg_ir: Optional[List[float]] = None   # infrared PPG channel
    ppg_red: Optional[List[float]] = None  # red PPG channel
    ppg_sample_rate: int = 64


class BandsRequest(BaseModel):
    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    # Optional vitals already computed by the browser
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None


# ── PPG / vitals helpers ──────────────────────────────────────────────────────

def _compute_heart_rate(ppg_ir: List[float], sample_rate: int = 64) -> Optional[float]:
    """
    Estimate heart rate (BPM) from an IR PPG signal using peak detection.

    Returns None if the signal is too short, too noisy, or the computed
    BPM falls outside the physiological range (40–200 BPM).
    """
    if not ppg_ir or len(ppg_ir) < sample_rate * 4:
        return None  # need at least 4 seconds

    sig = np.array(ppg_ir, dtype=np.float64)

    # ── 1. Smooth with a moving-average filter ─────────────────────────────
    kernel = max(4, sample_rate // 16)   # ~62 ms window
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    pad = kernel // 2
    padded = np.pad(sig, pad, mode="edge")
    smoothed = np.convolve(padded, np.ones(kernel) / kernel, mode="valid")

    # ── 2. Normalise so peak-finding thresholds are scale-invariant ────────
    sig_min, sig_max = smoothed.min(), smoothed.max()
    if sig_max - sig_min < 1e-6:
        return None  # flat signal
    norm = (smoothed - sig_min) / (sig_max - sig_min)

    # ── 3. Peak detection ──────────────────────────────────────────────────
    # min_distance: physiological minimum ~0.3 s between heartbeats
    # threshold: peaks must exceed 40% of the normalised range
    min_dist = max(1, int(sample_rate * 0.3))
    threshold = 0.4

    peaks = []
    i = 1
    while i < len(norm) - 1:
        if norm[i] > threshold and norm[i] >= norm[i - 1] and norm[i] >= norm[i + 1]:
            # Ensure minimum distance from previous peak
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
        i += 1

    if len(peaks) < 2:
        return None

    # ── 4. BPM from mean inter-peak interval ──────────────────────────────
    intervals = [peaks[k + 1] - peaks[k] for k in range(len(peaks) - 1)]
    mean_interval = np.mean(intervals)
    bpm = (sample_rate * 60.0) / mean_interval

    if bpm < 40 or bpm > 200:
        return None

    return round(float(bpm), 1)


def _compute_spo2(
    ppg_ir: List[float],
    ppg_red: List[float],
) -> Optional[float]:
    """
    Estimate blood oxygen saturation (SpO₂, %) from IR and red PPG channels.

    Uses the standard AC/DC ratio (R-value) method with an empirical
    linear approximation calibration curve:
        SpO₂ ≈ 110 − 25 × R
    where R = (AC_red / DC_red) / (AC_ir / DC_ir).

    Returns None if the signal quality is insufficient or the computed
    value falls outside the physiological range (85–100 %).
    """
    if not ppg_ir or not ppg_red:
        return None
    n = min(len(ppg_ir), len(ppg_red))
    if n < 64:
        return None

    ir = np.array(ppg_ir[-n:], dtype=np.float64)
    red = np.array(ppg_red[-n:], dtype=np.float64)

    dc_ir = np.mean(ir)
    dc_red = np.mean(red)
    if dc_ir < 1e-6 or dc_red < 1e-6:
        return None

    # AC amplitude: peak-to-peak of the AC-coupled signal
    ac_ir = (np.percentile(ir, 95) - np.percentile(ir, 5)) / 2.0
    ac_red = (np.percentile(red, 95) - np.percentile(red, 5)) / 2.0
    if ac_ir < 1e-6 or ac_red < 1e-6:
        return None

    R = (ac_red / dc_red) / (ac_ir / dc_ir)
    spo2 = 110.0 - 25.0 * R

    # Clamp to physiological range and round
    spo2 = round(float(np.clip(spo2, 85.0, 100.0)), 1)

    # Reject implausible outliers (R too far from expected ~0.4–0.6 for healthy adults)
    if R < 0.2 or R > 1.5:
        return None

    return spo2


def _bands_from_raw(
    eeg_data: List[List[float]],
    sample_rate: int,
) -> Optional[dict]:
    """
    Run FeatureExtractor on raw EEG and return band powers + asymmetry.
    Returns None if extraction fails or the model is not ready.
    """
    if not NEURO_AVAILABLE or _extractor is None:
        return None
    try:
        raw = np.array(eeg_data, dtype=np.float64)  # (n_ch, n_samp)
        _extractor._sr = sample_rate
        features, info = _extractor.extract(raw, meta=None)
        return {"features": features, "info": info}
    except Exception as exc:
        logger.warning(f"Feature extraction failed: {exc}")
        return None


def _classify(features: np.ndarray) -> Optional[dict]:
    """Run YogaClassifier on a feature vector. Returns None on failure."""
    if not NEURO_AVAILABLE or _clf is None or not _model_ready:
        return None
    try:
        chitta = _clf.predict(features)
        probs = _clf.predict_proba(features)
        return {"chitta": chitta, "probs": probs}
    except Exception as exc:
        logger.warning(f"Classification failed: {exc}")
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "ok": True,
        "model_ready": _model_ready,
        "board": "render-backend",
        "neuro_available": NEURO_AVAILABLE,
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Full inference from raw EEG data.

    Optionally processes PPG data (ppg_ir, ppg_red) to return heart_rate
    and spo2.  If PPG fields are absent or empty, those keys are null in
    the response — the UI shows "—" without raising any error.
    """
    t0 = time.perf_counter()

    # ── Vitals (PPG — completely optional) ───────────────────────────────────
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None

    if req.ppg_ir and len(req.ppg_ir) >= req.ppg_sample_rate * 4:
        loop = asyncio.get_event_loop()
        heart_rate = await loop.run_in_executor(
            None,
            lambda: _compute_heart_rate(req.ppg_ir, req.ppg_sample_rate),
        )
        if req.ppg_red and len(req.ppg_red) >= 64:
            spo2 = await loop.run_in_executor(
                None,
                lambda: _compute_spo2(req.ppg_ir, req.ppg_red),
            )

    # ── EEG inference ─────────────────────────────────────────────────────────
    if not _model_ready:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "latency_ms": round(elapsed, 1),
            "model_ready": False,
            "heart_rate": heart_rate,
            "spo2": spo2,
        }

    loop = asyncio.get_event_loop()
    extracted = await loop.run_in_executor(
        None,
        lambda: _bands_from_raw(req.eeg_data, req.sample_rate),
    )

    if extracted is None:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "error": "Feature extraction failed",
            "latency_ms": round(elapsed, 1),
            "heart_rate": heart_rate,
            "spo2": spo2,
        }

    features = extracted["features"]
    info = extracted["info"]

    result = await loop.run_in_executor(
        None, lambda: _classify(features)
    )
    if result is None:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "error": "Classification failed",
            "latency_ms": round(elapsed, 1),
            "heart_rate": heart_rate,
            "spo2": spo2,
        }

    chitta = result["chitta"]
    probs = result["probs"]

    # ── Vedantic interpretation ───────────────────────────────────────────────
    reading: Optional[object] = None
    if NEURO_AVAILABLE:
        try:
            band_rel = info.get("band_relative", {})
            asym = info.get("alpha_asymmetry", 0.0)
            reading = vedantic.interpret(chitta, band_rel, asym)
        except Exception as exc:
            logger.warning(f"Vedantic interpretation failed: {exc}")

    elapsed = (time.perf_counter() - t0) * 1000

    band_rel = info.get("band_relative", {})
    asym = info.get("alpha_asymmetry", 0.0)
    top_prob = max(probs.values()) if probs else 0.0

    # Map chitta to depth
    depth_map = {
        "Kshipta": "Surface",
        "Vikshipta": "Emerging",
        "Ekagra": "Deep",
        "Niruddha": "Profound",
    }
    depth = depth_map.get(chitta, "Surface")

    # Probabilities as percentage strings
    prob_strs = {k: f"{v * 100:.1f}%" for k, v in sorted(probs.items(), key=lambda x: -x[1])}

    # Swara from asymmetry
    if asym < -0.04:
        swara_state = "Ida Nadi — right hemisphere dominant"
        swara_conf = "High" if abs(asym) > 0.12 else "Moderate"
    elif asym > 0.04:
        swara_state = "Pingala Nadi — left hemisphere dominant"
        swara_conf = "High" if abs(asym) > 0.12 else "Moderate"
    else:
        swara_state = "Sushumna — both nadis balanced"
        swara_conf = "Moderate"

    # Tattva flags
    tattva_flags = []
    alpha = band_rel.get("alpha", 0)
    theta = band_rel.get("theta", 0)
    delta = band_rel.get("delta", 0)
    gamma = band_rel.get("gamma", 0)
    if alpha > 0.35 and theta < 0.25:
        tattva_flags.append("Pratyahara Window detected")
    if theta > 0.28 and alpha > 0.28:
        tattva_flags.append("Potential Tattva Activation")
    if theta > 0.32 and delta > 0.12:
        tattva_flags.append("Turiya Approach")
    if gamma > 0.12:
        tattva_flags.append("Gamma Spike")

    # Trigunas
    sat = alpha * 3.0 + theta * 1.5
    raj = band_rel.get("beta", 0) * 3.0 + gamma * 2.5
    tam = delta * 3.0
    total = max(sat + raj + tam, 1e-9)
    sat /= total; raj /= total; tam /= total
    dominant_guna = "Sattvic" if sat >= raj and sat >= tam else ("Rajasic" if raj >= tam else "Tamasic")

    # Vedantic reading fields (if available)
    swara_note = ""
    contemplative_depth = depth
    if reading is not None:
        try:
            swara_state = getattr(reading, "swara", swara_state)
            swara_conf = getattr(reading, "swara_confidence", swara_conf)
            swara_note = getattr(reading, "swara_note", "")
            contemplative_depth = getattr(reading, "contemplative_depth", depth)
            vedantic_flags = getattr(reading, "tattva_flags", None)
            if vedantic_flags:
                tattva_flags = vedantic_flags
        except Exception:
            pass

    return {
        "latency_ms": round(elapsed, 1),
        "data_quality": "✓ clean" if not info.get("is_padded") else "⚠ padded",
        "chitta_bhumi": {
            "state": chitta,
            "depth": contemplative_depth,
            "confidence": f"{top_prob * 100:.1f}%",
            "probabilities": prob_strs,
        },
        "depth": contemplative_depth,
        "swara": {
            "state": swara_state,
            "confidence": swara_conf,
            "note": swara_note,
        },
        "tattva": tattva_flags,
        "tattva_flags": tattva_flags,
        "eeg_spectrum": band_rel,
        "alpha_asymmetry": round(asym, 6),
        "gunas": {
            "sattva": round(sat, 4),
            "rajas": round(raj, 4),
            "tamas": round(tam, 4),
            "label": dominant_guna,
        },
        # ── Vitals — null when PPG not available on this headset ──────────
        "heart_rate": heart_rate,
        "spo2": spo2,
    }


@app.post("/analyze/bands")
async def analyze_bands(req: BandsRequest):
    """
    Classify from pre-computed band powers.
    Also accepts optional heart_rate / spo2 already computed by the browser
    and echoes them back so the UI can store them with the epoch.
    """
    t0 = time.perf_counter()

    delta = req.delta
    theta = req.theta
    alpha = req.alpha
    beta = req.beta
    gamma = req.gamma
    total = delta + theta + alpha + beta + gamma or 1.0

    alpha_r = alpha / total
    theta_r = theta / total
    beta_r = beta / total
    delta_r = delta / total
    gamma_r = gamma / total

    # Simple heuristic classification without ML model
    logits = [
        beta_r * 3.0 + gamma_r * 1.5 - alpha_r * 1.5,   # Kshipta
        alpha_r * 1.5 + beta_r * 1.5 - theta_r * 0.5,    # Vikshipta
        alpha_r * 3.5 + theta_r * 1.0 - beta_r * 2.0,    # Ekagra
        theta_r * 3.0 + delta_r * 2.0 - beta_r * 2.5,    # Niruddha
    ]
    states = ["Kshipta", "Vikshipta", "Ekagra", "Niruddha"]
    max_logit = max(logits)
    exps = [2.718281828 ** (l - max_logit) for l in logits]
    s = sum(exps)
    probs = [e / s for e in exps]
    best = probs.index(max(probs))
    chitta = states[best]
    depth_map = {"Kshipta": "Surface", "Vikshipta": "Emerging", "Ekagra": "Deep", "Niruddha": "Profound"}
    depth = depth_map[chitta]

    elapsed = (time.perf_counter() - t0) * 1000
    band_rel = {"delta": delta_r, "theta": theta_r, "alpha": alpha_r, "beta": beta_r, "gamma": gamma_r}

    # Trigunas
    sat = alpha_r * 3.0 + theta_r * 1.5
    raj = beta_r * 3.0 + gamma_r * 2.5
    tam = delta_r * 3.0
    g_total = max(sat + raj + tam, 1e-9)
    sat /= g_total; raj /= g_total; tam /= g_total
    dominant_guna = "Sattvic" if sat >= raj and sat >= tam else ("Rajasic" if raj >= tam else "Tamasic")

    return {
        "latency_ms": round(elapsed, 1),
        "chitta_bhumi": {
            "state": chitta,
            "depth": depth,
            "confidence": f"{max(probs) * 100:.1f}%",
            "probabilities": {s: f"{p * 100:.1f}%" for s, p in zip(states, probs)},
        },
        "depth": depth,
        "swara": {"state": "Sushumna — both nadis balanced", "confidence": "Moderate", "note": ""},
        "tattva": [],
        "tattva_flags": [],
        "eeg_spectrum": band_rel,
        "gunas": {
            "sattva": round(sat, 4),
            "rajas": round(raj, 4),
            "tamas": round(tam, 4),
            "label": dominant_guna,
        },
        # Echo back browser-computed vitals
        "heart_rate": req.heart_rate,
        "spo2": req.spo2,
    }
