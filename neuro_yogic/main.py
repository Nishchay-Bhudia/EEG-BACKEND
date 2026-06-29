"""
main.py -- Async Real-Time Inference Loop
=========================================
Orchestrates all four modules into a single live classification pipeline:

  EEGStreamer -> FeatureExtractor -> YogaClassifier -> VedanticLogic
                                  -> Console Output

Loop cadence: every 1.0 second, a fresh 2-second sliding window is pulled,
processed, classified, and pretty-printed as a JSON block.

The 2-second window slides rather than tumbles -- each epoch overlaps with
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

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from neuro_yogic.data_generator import DEFAULT_MODEL_PATH, save_dataset
from neuro_yogic.eeg_streamer import EEGStreamer
from neuro_yogic.feature_extractor import FeatureExtractor
from neuro_yogic.vedantic_logic import vedantic_analyze
from neuro_yogic.yoga_classifier import YogaClassifier

DEPTH_BARS = {
    "Kshipta":   "░░░░░░░░",
    "Vikshipta": "███░░░░░",
    "Ekagra":    "██████░░",
    "Niruddha":  "████████",
}


def _fmt_reading(
    epoch_num:  int,
    chitta:     str,
    probs:      dict,
    reading,
    info:       dict,
    elapsed_ms: float,
    is_padded:  bool,
) -> str:
    """Format one inference epoch as a JSON string."""
    ts       = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3] + " UTC"
    band_rel = info["band_relative"]

    out = {
        "timestamp":    ts,
        "epoch":        epoch_num,
        "latency_ms":   round(elapsed_ms, 1),
        "data_quality": "PADDED (buffer filling)" if is_padded else "clean",

        "chitta_bhumi": {
            "state":      chitta,
            "depth_bar":  DEPTH_BARS.get(chitta, "░░░░░░░░"),
            "depth":      reading.contemplative_depth,
            "confidence": f"{max(probs.values()) * 100:.1f}%",
            "probabilities": {k: f"{v * 100:.1f}%" for k, v in sorted(
                probs.items(), key=lambda x: -x[1]
            )},
        },

        "swara": {
            "state":      reading.swara,
            "confidence": reading.swara_confidence,
            "note":       reading.swara_note,
        },

        "tattva_correlates": reading.tattva_flags or ["No active Tattva flags this epoch"],

        "eeg_spectrum": {
            band: f"{power * 100:.1f}%"
            for band, power in band_rel.items()
        },

        "hemispheric_asymmetry": {
            "index":     round(info["alpha_asymmetry"], 6),
            "direction": (
                "Right > Left (Ida)"     if info["alpha_asymmetry"] > 0 else
                "Left > Right (Pingala)" if info["alpha_asymmetry"] < 0 else
                "Balanced (Sushumna)"
            ),
        },
    }
    return json.dumps(out, indent=2, ensure_ascii=False)


def _print_header(board_type: str) -> None:
    print()
    print("=" * 68)
    print("    Neuro-Yogic Cognitive Mapping Platform  v1.0")
    print("    EEG -> Chitta Bhumi . Swara . Tattva / Chakra")
    print("=" * 68)
    print(f"  Board  : {board_type.upper()}")
    print(f"  Press  : Ctrl-C to stop gracefully")
    print()


def _print_epoch(epoch_num: int, formatted: str, chitta: str) -> None:
    sep = "-" * 68
    print(sep)
    print(f"  EPOCH {epoch_num:>4}  .  {chitta.upper()}")
    print(sep)
    print(formatted)
    print()


async def _training_phase(retrain: bool) -> YogaClassifier:
    """Train a fresh classifier or load from disk (runs in executor to stay async)."""
    clf  = YogaClassifier(n_estimators=200)
    loop = asyncio.get_event_loop()

    if retrain or not os.path.exists(DEFAULT_MODEL_PATH):
        print("[Phase 1] Generating synthetic training dataset ...")
        csv_path = await loop.run_in_executor(None, save_dataset)
        print("[Phase 1] Training YogaClassifier (Random Forest, 200 trees) ...")
        await loop.run_in_executor(None, lambda: clf.train_model(csv_path))
        clf.save()
        print("[Phase 1] Training complete.\n")
    else:
        print(f"[Phase 1] Loading pre-trained model from '{DEFAULT_MODEL_PATH}' ...")
        clf.load()
        print("[Phase 1] Model loaded.\n")

    return clf


async def _inference_loop(
    clf:            YogaClassifier,
    streamer:       EEGStreamer,
    extractor:      FeatureExtractor,
    interval:       float,
    window_seconds: float,
) -> None:
    """Continuously fetch EEG data, extract features, classify, and print."""
    epoch = 0
    loop  = asyncio.get_event_loop()
    print("[Phase 3] Live inference loop started. Streaming ...\n")

    while True:
        t0 = time.perf_counter()
        epoch += 1

        # Fetch latest EEG window
        try:
            raw_eeg, meta = await loop.run_in_executor(
                None, lambda: streamer.get_latest_data(window_seconds)
            )
        except RuntimeError as exc:
            print(f"[EEGStreamer] Fetch error (epoch {epoch}): {exc}")
            await asyncio.sleep(2.0)
            continue

        # Signal processing & feature extraction
        try:
            features, info = extractor.extract(raw_eeg, meta)
        except Exception as exc:
            print(f"[FeatureExtractor] Error: {exc}")
            await asyncio.sleep(interval)
            continue

        if np.all(features == 0):
            print(f"[Epoch {epoch}] Zero-amplitude -- check electrode contact.")
            await asyncio.sleep(interval)
            continue

        # Chitta Bhumi classification
        try:
            chitta = clf.predict(features)
            probs  = clf.predict_proba(features)
        except Exception as exc:
            print(f"[YogaClassifier] Error: {exc}")
            await asyncio.sleep(interval)
            continue

        # Vedantic mapping
        reading = vedantic_analyze(info, chitta_bhumi=chitta)

        # Console output
        elapsed_ms = (time.perf_counter() - t0) * 1000
        formatted  = _fmt_reading(
            epoch_num  = epoch,
            chitta     = chitta,
            probs      = probs,
            reading    = reading,
            info       = info,
            elapsed_ms = elapsed_ms,
            is_padded  = meta.get("is_padded", False),
        )
        _print_epoch(epoch, formatted, chitta)

        sleep_time = max(0.0, interval - (time.perf_counter() - t0))
        await asyncio.sleep(sleep_time)


async def run(
    board_type:     str   = "synthetic",
    window_seconds: float = 2.0,
    interval:       float = 1.0,
    retrain:        bool  = True,
) -> None:
    """
    Full pipeline: train -> connect -> stream -> infer -> print.

    Parameters
    ----------
    board_type     : 'synthetic' | 'muse2' | 'brainbit' | 'cyton'
    window_seconds : EEG epoch length in seconds
    interval       : Inference cadence in seconds
    retrain        : Force retrain even if a saved model exists
    """
    _print_header(board_type)

    # Phase 1: ML Training
    clf = await _training_phase(retrain)

    # Phase 2: Hardware Connection
    print(f"[Phase 2] Connecting to board: {board_type.upper()} ...")
    streamer = EEGStreamer(board_type=board_type)

    try:
        streamer.start()
        extractor = FeatureExtractor(sample_rate=streamer.sample_rate or 256)
        print(f"[Phase 2] Board ready -- {streamer.sample_rate} Hz\n")

        # Phase 3: Live Inference Loop
        await _inference_loop(
            clf            = clf,
            streamer       = streamer,
            extractor      = extractor,
            interval       = interval,
            window_seconds = window_seconds,
        )

    except KeyboardInterrupt:
        print("\n[Main] Shutting down gracefully ...")
    except ConnectionError as exc:
        print(f"[Main] Board connection failed: {exc}")
    finally:
        streamer.stop()
        print("[Main] Session complete. Namaste.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuro-Yogic Cognitive Mapping Platform -- Real-Time EEG Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neuro_yogic.main                        # Synthetic board, retrain
  python -m neuro_yogic.main --board muse2          # Real Muse 2 headset
  python -m neuro_yogic.main --no-retrain           # Reuse saved model
  python -m neuro_yogic.main --interval 0.5         # 500 ms cadence
  python -m neuro_yogic.main --window 4.0           # 4-second EEG windows
        """
    )
    parser.add_argument("--board", default="synthetic",
                        choices=["synthetic", "muse2", "brainbit", "cyton"],
                        help="EEG board type (default: synthetic)")
    parser.add_argument("--window", type=float, default=2.0,
                        help="EEG epoch length in seconds (default: 2.0)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Inference cadence in seconds (default: 1.0)")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Skip retraining and load saved model from disk")
    args = parser.parse_args()

    asyncio.run(run(
        board_type     = args.board,
        window_seconds = args.window,
        interval       = args.interval,
        retrain        = not args.no_retrain,
    ))


if __name__ == "__main__":
    main()
