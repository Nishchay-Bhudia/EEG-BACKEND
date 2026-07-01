"""
server.py — FastAPI Web Server for EEG Neuro-Yogic Backend
===========================================================
Wraps the neuro_yogic package (FeatureExtractor → YogaClassifier →
vedantic_analyze) into an HTTP API consumed by the Vercel UI.

IMPORTANT: This file must sit at the repo root (same level as main.py and
the neuro_yogic/ package directory).  neuro_yogic/__init__.py must exist
(even if empty) for Python to treat it as a package.

Endpoints
---------
GET  /status        — model readiness probe (polled by the UI)
POST /analyze       — raw EEG + optional PPG → full classification + vitals
POST /analyze/bands — pre-computed band powers → fast classification

Render start command
--------------------
    uvicorn server:app --host 0.0.0.0 --port $PORT

Local dev
---------
    uvicorn server:app --host 0.0.0.0 --port 10000 --reload
"""

import asyncio
import os
import time
import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── neuro_yogic imports (graceful stub when package isn't importable) ─────────
try:
    from neuro_yogic.feature_extractor import FeatureExtractor
    from neuro_yogic.yoga_classifier import (
        YogaClassifier,
        DEFAULT_MODEL_PATH,
    )
    from neuro_yogic.data_generator import save_dataset
    from neuro_yogic.vedantic_logic import vedantic_analyze   # NOT vedantic.analyze
    NEURO_OK = True
except ImportError as _e:
    NEURO_OK = False
    logging.warning(f"[server] neuro_yogic import failed: {_e} — running in stub mode")

logger = logging.getLogger("uvicorn.error")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EEG Neuro-Yogic API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Vercel preview + production URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────
_clf: Optional[object]        = None
_extractor: Optional[object]  = None
_model_ready: bool            = False


async def _bootstrap_model() -> None:
    """Train or load the RandomForest classifier at startup."""
    global _clf, _extractor, _model_ready
    if not NEURO_OK:
        logger.warning("[Startup] neuro_yogic unavailable — stub mode, /analyze will use heuristics")
        _model_ready = True   # still "ready" — will use fallback path
        return

    loop = asyncio.get_event_loop()
    clf = YogaClassifier(n_estimators=200)

    try:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            logger.info("[Startup] Training YogaClassifier (first run, ~5 s) …")
            csv = await loop.run_in_executor(
                None, lambda: save_dataset(n_samples_per_class=500)
            )
            await loop.run_in_executor(None, lambda: clf.train_model(csv))
            clf.save()
            logger.info("[Startup] Model trained and saved.")
        else:
            logger.info("[Startup] Loading pre-trained model …")
            await loop.run_in_executor(None, clf.load)
            logger.info("[Startup] Model loaded.")

        _clf       = clf
        _extractor = FeatureExtractor(sample_rate=256)
        _model_ready = True
    except Exception as exc:
        logger.error(f"[Startup] Model bootstrap failed: {exc}")
        _model_ready = True   # allow status OK so UI doesn't spin forever


@app.on_event("startup")
async def on_startup() -> None:
    asyncio.create_task(_bootstrap_model())


# ── Pydantic request schemas ──────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    eeg_data: List[List[float]]           # [n_channels][n_samples]
    sample_rate: int = 256
    ppg_ir:  Optional[List[float]] = None  # IR PPG channel (Muse 2/S)
    ppg_red: Optional[List[float]] = None  # Red PPG channel
    ppg_sample_rate: int = 64


class BandsRequest(BaseModel):
    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta:  float = 0.0
    gamma: float = 0.0
    heart_rate: Optional[float] = None   # browser-computed, echoed back
    spo2:       Optional[float] = None


# ── PPG / biometrics helpers ──────────────────────────────────────────────────

def _heart_rate_from_ppg(ppg_ir: List[float], fs: int = 64) -> Optional[float]:
    """
    Estimate BPM from IR PPG via simple peak detection.
    Returns None if signal is too short, flat, or BPM is out of range (40–200).
    """
    if not ppg_ir or len(ppg_ir) < fs * 4:
        return None

    sig = np.asarray(ppg_ir, dtype=np.float64)

    # Moving-average smooth (~125 ms)
    k   = max(4, round(fs * 0.125))
    sig = np.convolve(sig, np.ones(k) / k, mode="same")

    lo, hi = sig.min(), sig.max()
    if hi - lo < 10:
        return None   # flat / no pulsatile signal

    norm = (sig - lo) / (hi - lo)

    # Peak detection — must exceed 40 % amplitude, ≥ 0.3 s apart
    min_gap  = max(1, round(fs * 0.30))
    threshold = 0.40
    peaks: List[int] = []
    for i in range(1, len(norm) - 1):
        if norm[i] > threshold and norm[i] >= norm[i - 1] and norm[i] >= norm[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap:
                peaks.append(i)

    if len(peaks) < 2:
        return None

    intervals = [peaks[j + 1] - peaks[j] for j in range(len(peaks) - 1)]
    mean_gap  = float(np.mean(intervals))
    bpm       = (fs * 60.0) / mean_gap

    return round(bpm, 1) if 40 <= bpm <= 200 else None


def _spo2_from_ppg(ppg_ir: List[float], ppg_red: List[float]) -> Optional[float]:
    """
    Estimate SpO₂ (%) using AC/DC ratio method:
        R = (AC_red / DC_red) / (AC_ir / DC_ir)
        SpO₂ ≈ 110 − 25 × R  (empirical linear calibration)
    Returns None if signal quality is insufficient or R is implausible.
    """
    n = min(len(ppg_ir), len(ppg_red))
    if n < 64:
        return None

    ir  = np.asarray(ppg_ir[-n:],  dtype=np.float64)
    red = np.asarray(ppg_red[-n:], dtype=np.float64)

    dc_ir  = float(ir.mean())
    dc_red = float(red.mean())
    if dc_ir < 1 or dc_red < 1:
        return None

    ac_ir  = float((np.percentile(ir,  95) - np.percentile(ir,  5)) / 2)
    ac_red = float((np.percentile(red, 95) - np.percentile(red, 5)) / 2)
    if ac_ir < 1 or ac_red < 1:
        return None

    R = (ac_red / dc_red) / (ac_ir / dc_ir)
    if not (0.2 <= R <= 1.5):
        return None

    spo2 = float(np.clip(110.0 - 25.0 * R, 85.0, 100.0))
    return round(spo2, 1)


# ── Heuristic fallback (used when model not loaded or neuro_yogic missing) ───

_STATES = ["Kshipta", "Vikshipta", "Ekagra", "Niruddha"]
_DEPTH  = {"Kshipta": "Surface", "Vikshipta": "Emerging", "Ekagra": "Deep", "Niruddha": "Profound"}

def _heuristic_classify(band_rel: dict) -> dict:
    d = band_rel.get("delta", 0.1)
    t = band_rel.get("theta", 0.2)
    a = band_rel.get("alpha", 0.3)
    b = band_rel.get("beta",  0.3)
    g = band_rel.get("gamma", 0.1)

    logits = [
        b * 3.0 + g * 1.5 - a * 1.5,    # Kshipta
        a * 1.5 + b * 1.5 - t * 0.5,    # Vikshipta
        a * 3.5 + t * 1.0 - b * 2.0,    # Ekagra
        t * 3.0 + d * 2.0 - b * 2.5,    # Niruddha
    ]
    m    = max(logits)
    exps = [2.71828 ** (x - m) for x in logits]
    s    = sum(exps)
    probs = [e / s for e in exps]
    best  = probs.index(max(probs))

    return {
        "chitta": _STATES[best],
        "probs":  {st: probs[i] for i, st in enumerate(_STATES)},
    }


def _bands_from_eeg(eeg_data: List[List[float]], sample_rate: int):
    """Run FeatureExtractor on raw EEG. Returns (features, info) or None."""
    if not NEURO_OK or _extractor is None:
        return None
    try:
        raw = np.asarray(eeg_data, dtype=np.float64)
        _extractor._sr = sample_rate
        return _extractor.extract(raw, meta=None)
    except Exception as exc:
        logger.warning(f"FeatureExtractor failed: {exc}")
        return None


def _classify_eeg(features):
    """Run YogaClassifier. Returns (chitta, probs) or None."""
    if not NEURO_OK or _clf is None:
        return None
    try:
        chitta = _clf.predict(features)
        probs  = _clf.predict_proba(features)
        return chitta, probs
    except Exception as exc:
        logger.warning(f"YogaClassifier failed: {exc}")
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "ok":            True,
        "model_ready":   _model_ready,
        "neuro_ok":      NEURO_OK,
        "board":         "render-fastapi",
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Full inference from raw EEG.

    Optional PPG (ppg_ir, ppg_red) enables heart_rate and spo2 in the
    response.  When PPG is absent or the headset doesn't support it,
    both fields are null — the UI shows '—' silently.
    """
    t0 = time.perf_counter()

    # ── Biometrics (PPG — completely optional) ────────────────────────────────
    heart_rate: Optional[float] = None
    spo2:       Optional[float] = None

    if req.ppg_ir and len(req.ppg_ir) >= req.ppg_sample_rate * 4:
        loop = asyncio.get_event_loop()
        heart_rate = await loop.run_in_executor(
            None, lambda: _heart_rate_from_ppg(req.ppg_ir, req.ppg_sample_rate)
        )
        if req.ppg_red and len(req.ppg_red) >= 64:
            spo2 = await loop.run_in_executor(
                None, lambda: _spo2_from_ppg(req.ppg_ir, req.ppg_red)
            )

    # ── EEG feature extraction + classification ───────────────────────────────
    band_rel: dict = {}
    chitta:   str  = "Kshipta"
    probs:    dict = {}
    info:     dict = {}

    loop = asyncio.get_event_loop()
    extracted = await loop.run_in_executor(
        None, lambda: _bands_from_eeg(req.eeg_data, req.sample_rate)
    )

    if extracted is not None:
        features, info = extracted
        band_rel = info.get("band_relative", {})

        clf_result = await loop.run_in_executor(None, lambda: _classify_eeg(features))
        if clf_result is not None:
            chitta, probs = clf_result
        else:
            # RF model not ready yet — heuristic fallback
            res    = _heuristic_classify(band_rel)
            chitta = res["chitta"]
            probs  = res["probs"]
    else:
        # FeatureExtractor not available — compute band powers from FFT inline
        eeg = np.asarray(req.eeg_data[0] if req.eeg_data else [], dtype=np.float64)
        if len(eeg) >= 64:
            sz    = 1 << int(np.log2(len(eeg)))
            freqs = np.fft.rfftfreq(sz, d=1.0 / req.sample_rate)
            psd   = np.abs(np.fft.rfft(eeg[:sz])) ** 2
            b = lambda lo, hi: float(psd[(freqs >= lo) & (freqs < hi)].sum())
            d, t, a, be, g = b(0.5,4), b(4,8), b(8,13), b(13,30), b(30,50)
            tot = d + t + a + be + g or 1
            band_rel = {
                "delta": d/tot, "theta": t/tot, "alpha": a/tot,
                "beta": be/tot, "gamma": g/tot,
            }
        res    = _heuristic_classify(band_rel)
        chitta = res["chitta"]
        probs  = res["probs"]

    # ── Vedantic interpretation ───────────────────────────────────────────────
    depth  = _DEPTH.get(chitta, "Surface")
    swara_state = "Sushumna (Balanced / Central)"
    swara_conf  = "Moderate"
    swara_note  = "Hemispheric balance within threshold. Sushumna open."
    tattva_flags: List[str] = []
    gunas_dict = {"sattva": 0.334, "rajas": 0.333, "tamas": 0.333,
                  "label": "Balanced", "note": ""}
    asym = float(info.get("alpha_asymmetry", 0.0))

    if NEURO_OK:
        try:
            reading = vedantic_analyze(info, chitta_bhumi=chitta)
            v = reading.to_dict()
            sw           = v.get("swara", {})
            swara_state  = sw.get("state",      swara_state)
            swara_conf   = sw.get("confidence", swara_conf)
            swara_note   = sw.get("note",       swara_note)
            tattva_flags = v.get("tattva_flags", [])
            depth        = v.get("contemplative_depth", depth)
            g            = v.get("gunas", {})
            gunas_dict   = {
                "sattva": g.get("sattva", 0.334),
                "rajas":  g.get("rajas",  0.333),
                "tamas":  g.get("tamas",  0.333),
                "label":  g.get("label",  "Balanced"),
                "note":   g.get("note",   ""),
            }
        except Exception as exc:
            logger.warning(f"vedantic_analyze failed: {exc}")
    else:
        # Heuristic gunas when neuro_yogic unavailable
        a_r = band_rel.get("alpha", 0.3)
        b_r = band_rel.get("beta",  0.2)
        d_r = band_rel.get("delta", 0.1)
        g_r = band_rel.get("gamma", 0.1)
        sat = max(a_r * 3.0, 0.05)
        raj = max(b_r * 3.0 + g_r * 2.5, 0.05)
        tam = max(d_r * 3.0, 0.05)
        tot = sat + raj + tam
        sat /= tot; raj /= tot; tam /= tot
        dom = "Sattvic" if sat >= raj and sat >= tam else ("Rajasic" if raj >= tam else "Tamasic")
        gunas_dict = {"sattva": round(sat, 4), "rajas": round(raj, 4),
                      "tamas": round(tam, 4), "label": dom, "note": ""}

    top_prob   = max(probs.values()) if probs else 0.0
    prob_strs  = {k: f"{v * 100:.1f}%" for k, v in probs.items()}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "latency_ms":   round(elapsed_ms, 1),
        "data_quality": "⚠ padded" if info.get("is_padded") else "✓ clean",
        "chitta_bhumi": {
            "state":         chitta,
            "depth":         depth,
            "confidence":    f"{top_prob * 100:.1f}%",
            "probabilities": prob_strs,
        },
        "depth":    depth,
        "swara": {
            "state":      swara_state,
            "confidence": swara_conf,
            "note":       swara_note,
        },
        "tattva":       tattva_flags,
        "tattva_flags": tattva_flags,
        "eeg_spectrum": band_rel,
        "alpha_asymmetry": round(asym, 6),
        "gunas": gunas_dict,
        # Biometrics — null when headset has no PPG sensor
        "heart_rate": heart_rate,
        "spo2":       spo2,
    }


@app.post("/analyze/bands")
async def analyze_bands(req: BandsRequest):
    """Classify from pre-computed relative band powers."""
    t0 = time.perf_counter()

    tot = (req.delta + req.theta + req.alpha + req.beta + req.gamma) or 1.0
    band_rel = {
        "delta": req.delta / tot,
        "theta": req.theta / tot,
        "alpha": req.alpha / tot,
        "beta":  req.beta  / tot,
        "gamma": req.gamma / tot,
    }

    res    = _heuristic_classify(band_rel)
    chitta = res["chitta"]
    probs  = res["probs"]
    depth  = _DEPTH[chitta]

    a_r = band_rel["alpha"]; b_r = band_rel["beta"]
    d_r = band_rel["delta"]; g_r = band_rel["gamma"]
    sat = max(a_r * 3.0, 0.05)
    raj = max(b_r * 3.0 + g_r * 2.5, 0.05)
    tam = max(d_r * 3.0, 0.05)
    gt  = sat + raj + tam
    sat /= gt; raj /= gt; tam /= gt
    dom = "Sattvic" if sat >= raj and sat >= tam else ("Rajasic" if raj >= tam else "Tamasic")

    return {
        "latency_ms":   round((time.perf_counter() - t0) * 1000, 1),
        "data_quality": "✓ pre-computed",
        "chitta_bhumi": {
            "state":         chitta,
            "depth":         depth,
            "confidence":    f"{max(probs.values()) * 100:.1f}%",
            "probabilities": {s: f"{p * 100:.1f}%" for s, p in probs.items()},
        },
        "depth":    depth,
        "swara":    {"state": "Sushumna (Balanced / Central)", "confidence": "Moderate", "note": ""},
        "tattva":   [],
        "tattva_flags": [],
        "eeg_spectrum": band_rel,
        "alpha_asymmetry": 0.0,
        "gunas": {
            "sattva": round(sat, 4), "rajas": round(raj, 4), "tamas": round(tam, 4),
            "label": dom, "note": "",
        },
        # Echo browser-computed vitals
        "heart_rate": req.heart_rate,
        "spo2":       req.spo2,
    }
