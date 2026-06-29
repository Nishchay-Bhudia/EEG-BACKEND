"""
main.py — Async Real-Time Inference Loop
=========================================
Orchestrates all four modules into a single live classification pipeline:

  EEGStreamer  →  FeatureExtractor  →  YogaClassifier  →  VedanticLogic
                                                         →  Console Output

Loop cadence: every 1.0 second, a fresh 2-second sliding window is pulled,
processed, classified, and pretty-printed as a JSON block.

The 2-second window slides rather than tumbles — each epoch overlaps with
the previous one by ~1 second, giving temporal continuity and allowing the
classifier to detect state *transitions* rather than just point-in-time states.

Usage
-----
# Run with Synthetic Board (no hardware required):
    python -m neuro_yogic.main

# Run with Muse 2 over Bluetooth (headset must be paired first):
    python -m neuro_yogic.main --board muse2

# Use a pre-trained model without retraining:
    python -m neuro_yogic.main --no-retrain

# Adjust loop interval and window length:
    python -m neuro_yogic.main --interval 0.5 --window 4.0
"""

import asyncio
import json
import signal
import sys
import time
import argparse
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from neuro_yogic.data_generator   import save_dataset
from neuro_yogic.yoga_classifier  import YogaClassifier, DEFAULT_MODEL_PATH, DEFAULT_LABELS_PATH
from neuro_yogic.eeg_streamer      import EEGStreamer
from neuro_yogic.feature_extractor import FeatureExtractor
from neuro_yogic import vedantic_logic as vedantic

# ── ANSI colour helpers (degrades gracefully on terminals without colour) ──
try:
    import sys as _sys
    _HAS_COLOR = _sys.stdout.isatty()
except Exception:
    _HAS_COLOR = False

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _HAS_COLOR else text

CYAN   = lambda t: _c(t, "96")
GREEN  = lambda t: _c(t, "92")
YELLOW = lambda t: _c(t, "93")
PURPLE = lambda t: _c(t, "95")
RED    = lambda t: _c(t, "91")
BOLD   = lambda t: _c(t, "1")
DIM    = lambda t: _c(t, "2")


# ── Chitta Bhumi depth bar ─────────────────────────────────────────────
DEPTH_BARS = {
    "Kshipta":   "░░░░░░░░",
    "Vikshipta": "███░░░░░",
    "Ekagra":    "██████░░",
    "Niruddha":  "████████",
}

SWARA_SYMBOLS = {
    "Ida":       "🌙 Ida",
    "Pingala":   "☀️  Pingala",
    "Sushumna":  "⚖️  Sushumna",
}


def _fmt_epoch(
    epoch_num:    int,
    chitta:       str,
    probs:        dict,
    reading:      vedantic.VedanticReading,
    info:         dict,
    elapsed_ms:   float,
    is_padded:    bool,
) -> str:
    """
    Format a single inference epoch as a human-readable JSON-style block.
    """
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3] + " UTC"

    # Determine which Swara keyword is active for the symbol lookup
    swara_key = "Sushumna"
    if "Ida" in reading.swara:        swara_key = "Ida"
    elif "Pingala" in reading.swara:  swara_key = "Pingala"

    depth_bar  = DEPTH_BARS.get(chitta, "░░░░░░░░")
    band_rel   = info["band_relative"]
    asym       = info["alpha_asymmetry"]

    # Confidence: top probability
    top_prob = max(probs.values())
    conf_str = f"{top_prob*100:.1f}%"

    out = {
        "timestamp":    ts,
        "epoch":        epoch_num,
        "latency_ms":   round(elapsed_ms, 1),
        "data_quality": "⚠ padded (buffer filling)" if is_padded else "✓ clean",

        "chitta_bhumi": {
            "state":      chitta,
            "depth_bar":  depth_bar,
            "depth":      reading.contemplative_depth,
            "confidence": conf_str,
            "probabilities": {k: f"{v*100:.1f}%" for k, v in sorted(
                probs.items(), key=lambda x: -x[1]
            )},
        },

        "swara": {
            "state":      reading.swara,
            "confidence": reading.swara_confidence,
            "note":       reading.swara_note,
        },

        "tattva_correlates": reading.tattva_flags if reading.tattva_flags else [
            "No active Tattva flags this epoch"
        ],

        "eeg_spectrum": {
            band: f"{power*100:.1f}%"
            for band, power in band_rel.items()
        },

        "hemispheric_asymmetry": {
            "index":       round(asym, 6),
            "direction":   "Right > Left (Ida)" if asym > 0 else (
                           "Left > Right (Pingala)" if asym < 0 else "Balanced"),
        },
    }

    # Pretty-print as JSON with colour accents on the state line
    raw_json = json.dumps(out, indent=2, ensure_ascii=False)
    return raw_json


def print_header(board_type: str) -> None:
    print()
    print(BOLD("╔══════════════════════════════════════════════════════════════╗"))
    print(BOLD("║   Neuro-Yogic Cognitive Mapping Platform  ·  v1.0           ║"))
    print(BOLD("║   Real-Time EEG → Chitta Bhumi + Swara + Tattva             ║"))
    print(BOLD("╚══════════════════════════════════════════════════════════════╝"))
    print(f"  Board  : {CYAN(board_type.upper())}")
    print(f"  Press  : {DIM('Ctrl-C to stop gracefully')}")
    print()


def print_epoch(epoch_num: int, formatted: str, chitta: str) -> None:
    """Print a formatted epoch block with colour decoration."""
    colour = {
        "Kshipta":   RED,
        "Vikshipta": YELLOW,
        "Ekagra":    GREEN,
        "Niruddha":  PURPLE,
    }.get(chitta, lambda t: t)

    sep = colour("─" * 66)
    print(sep)
    print(colour(f"  EPOCH {epoch_num:>4}  ·  {chitta.upper()}"))
    print(sep)
    print(formatted)
    print()


async def training_phase(retrain: bool) -> YogaClassifier:
    """
    Either train a fresh classifier or load a pre-trained one.

    Training is done in the asyncio executor so it doesn't block the event
    loop (RandomForest training is CPU-bound, ~1–3 seconds for 2400 samples).
    """
    clf = YogaClassifier(n_estimators=200)

    if retrain or not __import__("os").path.exists(DEFAULT_MODEL_PATH):
        print(GREEN("[Phase 1] Generating synthetic training dataset …"))
        loop = asyncio.get_event_loop()

        # Run blocking I/O and CPU work off the event loop
        csv_path = await loop.run_in_executor(
            None, lambda: save_dataset(n_samples_per_class=500)
        )
        print(GREEN("[Phase 1] Training YogaClassifier (Random Forest) …"))
        await loop.run_in_executor(None, lambda: clf.train_model(csv_path))
        clf.save()
        print(GREEN("[Phase 1] Training complete and model saved.\n"))
    else:
        print(GREEN(f"[Phase 1] Loading pre-trained model from '{DEFAULT_MODEL_PATH}' …"))
        clf.load()
        print(GREEN("[Phase 1] Model loaded.\n"))

    return clf


async def inference_loop(
    clf:            YogaClassifier,
    streamer:       EEGStreamer,
    extractor:      FeatureExtractor,
    interval:       float = 1.0,
    window_seconds: float = 2.0,
) -> None:
    """
    Continuously pull data from the EEG stream, extract features,
    classify the Chitta Bhumi, and print the Vedantic reading.

    Graceful Bluetooth packet-drop handling:
      - EEGStreamer zero-pads incomplete windows; `is_padded` is flagged.
      - Any BrainFlow exception during data fetch is caught and retried
        after a short backoff, rather than crashing the loop.
    """
    epoch = 0
    loop  = asyncio.get_event_loop()

    print(GREEN("[Phase 3] Live inference loop started. Streaming …\n"))

    while True:
        epoch_start = time.perf_counter()
        epoch += 1

        # ── 1. Fetch latest EEG window ─────────────────────────────────
        try:
            raw_eeg, meta = await loop.run_in_executor(
                None,
                lambda: streamer.get_latest_data(window_seconds=window_seconds),
            )
        except RuntimeError as exc:
            print(RED(f"[EEGStreamer] Data fetch error (epoch {epoch}): {exc}"))
            print(DIM("  → Retrying in 2 s …"))
            await asyncio.sleep(2.0)
            continue

        # ── 2. Signal processing & feature extraction ──────────────────
        try:
            features, info = extractor.extract(raw_eeg, meta)
        except Exception as exc:
            print(RED(f"[FeatureExtractor] Processing error: {exc}"))
            await asyncio.sleep(interval)
            continue

        # Check for degenerate (all-zero) feature vector — can happen
        # if the headset is not properly seated on the scalp.
        if np.all(features == 0):
            print(YELLOW(f"[Epoch {epoch}] Zero-amplitude signal — check electrode contact."))
            await asyncio.sleep(interval)
            continue

        # ── 3. Chitta Bhumi classification ─────────────────────────────
        try:
            chitta = clf.predict(features)
            probs  = clf.predict_proba(features)
        except Exception as exc:
            print(RED(f"[YogaClassifier] Inference error: {exc}"))
            await asyncio.sleep(interval)
            continue

        # ── 4. Vedantic logic mapping ──────────────────────────────────
        reading = vedantic.analyze(info, chitta_bhumi=chitta)

        # ── 5. Console output ──────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - epoch_start) * 1000
        formatted  = _fmt_epoch(
            epoch_num   = epoch,
            chitta      = chitta,
            probs       = probs,
            reading     = reading,
            info        = info,
            elapsed_ms  = elapsed_ms,
            is_padded   = meta.get("is_padded", False),
        )
        print_epoch(epoch, formatted, chitta)

        # ── 6. Sleep until next epoch ──────────────────────────────────
        processing_time = (time.perf_counter() - epoch_start)
        sleep_time = max(0.0, interval - processing_time)
        await asyncio.sleep(sleep_time)


async def run(
    board_type:     str   = "synthetic",
    window_seconds: float = 2.0,
    interval:       float = 1.0,
    retrain:        bool  = True,
) -> None:
    """
    Full pipeline: train → connect → stream → infer.

    Parameters
    ----------
    board_type     : 'synthetic' | 'muse2' | 'brainbit' | 'cyton'
    window_seconds : EEG epoch length in seconds
    interval       : Inference cadence in seconds
    retrain        : Force retrain even if a saved model exists
    """
    print_header(board_type)

    # ── Phase 1: ML Training ───────────────────────────────────────────
    clf = await training_phase(retrain)

    # ── Phase 2: Hardware Connection ──────────────────────────────────
    print(GREEN(f"[Phase 2] Connecting to EEG board: {board_type.upper()} …"))
    streamer  = EEGStreamer(board_type=board_type)
    extractor = FeatureExtractor(sample_rate=256)  # overridden after start()

    try:
        streamer.start()
        # Update extractor sample rate from the actual board
        extractor = FeatureExtractor(sample_rate=streamer.sample_rate or 256)
        print(GREEN(f"[Phase 2] Board connected. Sample rate: {streamer.sample_rate} Hz\n"))

        # ── Phase 3: Live Inference Loop ───────────────────────────────
        await inference_loop(
            clf            = clf,
            streamer       = streamer,
            extractor      = extractor,
            interval       = interval,
            window_seconds = window_seconds,
        )

    except KeyboardInterrupt:
        print(f"\n{YELLOW('[Main] KeyboardInterrupt received — shutting down gracefully …')}")
    except ConnectionError as exc:
        print(RED(f"[Main] Board connection failed: {exc}"))
    finally:
        streamer.stop()
        print(GREEN("[Main] Session complete. Namaste. 🙏"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuro-Yogic Cognitive Mapping Platform — Real-Time EEG Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neuro_yogic.main                    # Synthetic board, retrain each run
  python -m neuro_yogic.main --board muse2      # Connect to Muse 2 via Bluetooth
  python -m neuro_yogic.main --no-retrain       # Reuse saved model (faster start)
  python -m neuro_yogic.main --interval 0.5     # 500 ms inference cadence
  python -m neuro_yogic.main --window 4.0       # 4-second EEG windows
        """
    )
    parser.add_argument(
        "--board", default="synthetic",
        choices=["synthetic", "muse2", "brainbit", "cyton"],
        help="EEG board type (default: synthetic)",
    )
    parser.add_argument(
        "--window", type=float, default=2.0,
        help="EEG window length in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Inference cadence in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-retrain", action="store_true",
        help="Skip training and load model from disk",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            board_type     = args.board,
            window_seconds = args.window,
            interval       = args.interval,
            retrain        = not args.no_retrain,
        )
    )


if __name__ == "__main__":
    main()
