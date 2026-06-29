"""
eeg_streamer.py
===============
BrainFlow hardware abstraction layer.

Supported boards (change board_type -- no other code changes needed):
  'synthetic' -> BoardIds.SYNTHETIC_BOARD   (built-in simulator, no hardware)
  'muse2'     -> BoardIds.MUSE_2_BOARD      (InteraXon Muse 2, Bluetooth LE)
  'brainbit'  -> BoardIds.BRAINBIT_BOARD    (BrainBit, Bluetooth LE)
  'cyton'     -> BoardIds.CYTON_BOARD       (OpenBCI Cyton, serial USB)

Bluetooth packet-drop handling:
  BrainFlow fills dropped packets with linear interpolation internally.
  If the ring-buffer has fewer samples than requested (first ~2 s of
  the session), EEGStreamer zero-pads the left edge so the FFT pipeline
  always receives a fixed-length array. is_padded=True flags this case.
"""

import time
from typing import Optional, Tuple

import numpy as np

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    BoardIds = None

SUPPORTED_BOARDS = {
    "synthetic": "SYNTHETIC_BOARD",
    "muse2":     "MUSE_2_BOARD",
    "brainbit":  "BRAINBIT_BOARD",
    "cyton":     "CYTON_BOARD",
}


class EEGStreamer:
    """BrainFlow-backed EEG data source with a clean three-method interface."""

    def __init__(
        self,
        board_type:  str            = "synthetic",
        serial_port: Optional[str]  = None,
        mac_address: Optional[str]  = None,
        log_level:   int            = 1,
    ) -> None:
        if not BRAINFLOW_AVAILABLE:
            raise ImportError("brainflow not installed. Run: pip install brainflow")

        board_type = board_type.lower()
        if board_type not in SUPPORTED_BOARDS:
            raise ValueError(f"Unknown board '{board_type}'. Choose: {list(SUPPORTED_BOARDS)}")

        self._board_type = board_type
        self._board_id   = getattr(BoardIds, SUPPORTED_BOARDS[board_type])

        params = BrainFlowInputParams()
        if serial_port:
            params.serial_port = serial_port
        if mac_address:
            params.mac_address = mac_address

        BoardShim.set_log_level(log_level)
        self._board        = BoardShim(self._board_id, params)
        self._streaming    = False
        self._sample_rate: Optional[int]  = None
        self._eeg_channels: Optional[list] = None
        self._ts_channel:  Optional[int]  = None

    def start(self, buffer_size: int = 45000) -> None:
        """Open the board connection and begin streaming into the ring-buffer."""
        if self._streaming:
            return
        board_name = SUPPORTED_BOARDS.get(self._board_type, "UNKNOWN")
        print(f"[EEGStreamer] Connecting to {board_name} ...")
        try:
            self._board.prepare_session()
        except Exception as exc:
            raise ConnectionError(f"Board connection failed: {exc}") from exc
        self._board.start_stream(buffer_size)
        self._streaming    = True
        self._sample_rate  = BoardShim.get_sampling_rate(self._board_id)
        self._eeg_channels = BoardShim.get_eeg_channels(self._board_id)
        self._ts_channel   = BoardShim.get_timestamp_channel(self._board_id)
        print(
            f"[EEGStreamer] Streaming at {self._sample_rate} Hz | "
            f"{len(self._eeg_channels)} EEG channels"
        )
        time.sleep(1.0)  # warm-up: let ring-buffer fill

    def stop(self) -> None:
        """Stop streaming and release the board connection."""
        if not self._streaming:
            return
        try:
            self._board.stop_stream()
            self._board.release_session()
        except Exception:
            pass
        self._streaming = False
        print("[EEGStreamer] Session released.")

    def get_latest_data(self, window_seconds: float = 2.0) -> Tuple[np.ndarray, dict]:
        """
        Retrieve the most recent `window_seconds` of EEG from the ring-buffer.

        Returns
        -------
        eeg_data : ndarray, shape (n_channels, n_samples)
        meta     : dict with sample_rate, is_padded, timestamp, ...
        """
        if not self._streaming:
            raise RuntimeError("Not streaming -- call start() first.")

        n_req = int(window_seconds * self._sample_rate)

        try:
            raw = self._board.get_current_board_data(n_req)
        except Exception as exc:
            raise RuntimeError(f"Ring-buffer read failed: {exc}") from exc

        eeg   = raw[self._eeg_channels, :]
        n_got = eeg.shape[1]

        if n_got < n_req:
            pad = n_req - n_got
            eeg = np.pad(eeg, ((0, 0), (pad, 0)), mode="constant")

        ts = float(raw[self._ts_channel, -1]) if raw.shape[1] > 0 else time.time()

        meta = {
            "sample_rate":         self._sample_rate,
            "eeg_channels":        self._eeg_channels,
            "n_samples_requested": n_req,
            "n_samples_received":  n_got,
            "is_padded":           n_got < n_req,
            "timestamp":           ts,
        }
        return eeg, meta

    @property
    def sample_rate(self) -> Optional[int]:
        return self._sample_rate

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    def __enter__(self) -> "EEGStreamer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
