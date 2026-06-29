"""
eeg_streamer.py — Hardware Abstraction Module (EEGStreamer)
===========================================================
Wraps BrainFlow's BoardShim API to provide a clean, hardware-agnostic
interface for starting an EEG stream and fetching rolling data windows.

BrainFlow is the industry standard for consumer EEG hardware because a
single API supports 50+ boards — changing one parameter switches between:
  - BoardIds.MUSE_2_BOARD       → InteraXon Muse 2 (Bluetooth LE)
  - BoardIds.BRAINBIT_BOARD     → BrainBit (Bluetooth LE)
  - BoardIds.SYNTHETIC_BOARD    → Software simulator (no hardware needed)
  - BoardIds.CYTON_BOARD        → OpenBCI Cyton (8-channel research grade)

Bluetooth Packet Drop Handling
--------------------------------
Consumer EEG headsets lose packets (~1–5% under typical conditions).
BrainFlow's ring-buffer automatically fills gaps with linear interpolation
when `get_board_data()` is called, so the downstream signal pipeline
always receives a contiguous array of the requested length.
If the ring-buffer contains fewer samples than requested (e.g., very early
in the session), EEGStreamer pads the tail with zeros so the FFT pipeline
never receives a ragged array. The feature extractor flags low-data windows
via `is_padded=True` in its returned metadata dict.
"""

import time
import numpy as np
from typing import Optional, Tuple

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    # Allow the module to be imported even when brainflow is not installed
    # (e.g., during unit tests or documentation builds).
    BRAINFLOW_AVAILABLE = False
    BoardIds = None


# Map of human-readable board names to BrainFlow BoardIds
SUPPORTED_BOARDS = {
    "synthetic": "SYNTHETIC_BOARD",
    "muse2":     "MUSE_2_BOARD",
    "brainbit":  "BRAINBIT_BOARD",
    "cyton":     "CYTON_BOARD",
}


class EEGStreamer:
    """
    Hardware-agnostic EEG data source using BrainFlow.

    All board-specific details (channel layout, sample rate, Bluetooth
    protocol) are handled internally by BrainFlow; this class exposes
    only three methods the rest of the pipeline needs:

        start()                  — open the board connection and begin streaming
        get_latest_data(secs)    — retrieve the last `secs` seconds of EEG data
        stop()                   — cleanly release the board connection

    Example
    -------
    # Test without hardware (Synthetic Board):
    streamer = EEGStreamer(board_type="synthetic")
    streamer.start()
    data, meta = streamer.get_latest_data(window_seconds=2.0)
    streamer.stop()

    # Switch to Muse 2 (change one line):
    streamer = EEGStreamer(board_type="muse2")
    """

    def __init__(
        self,
        board_type: str = "synthetic",
        serial_port: Optional[str] = None,
        mac_address: Optional[str] = None,
        ip_address: Optional[str] = None,
        ip_port: int = 0,
        log_level: int = 1,  # 1 = WARN; reduces BrainFlow verbosity
    ) -> None:
        """
        Parameters
        ----------
        board_type : str
            One of 'synthetic', 'muse2', 'brainbit', 'cyton' (see SUPPORTED_BOARDS).
        serial_port : str, optional
            Required for serial-connected boards (e.g., OpenBCI Cyton: '/dev/ttyUSB0').
        mac_address : str, optional
            Required for some Bluetooth boards (e.g., BrainBit).
        ip_address : str, optional
            Required for WiFi-connected boards.
        ip_port : int
            Port for WiFi boards.
        log_level : int
            BrainFlow log level (0=DEBUG, 1=WARN, 2=ERROR, 3=CRITICAL, 4=FATAL).
        """
        if not BRAINFLOW_AVAILABLE:
            raise ImportError(
                "brainflow is not installed. Run: pip install brainflow"
            )

        board_type = board_type.lower()
        if board_type not in SUPPORTED_BOARDS:
            raise ValueError(
                f"Unknown board_type '{board_type}'. "
                f"Choose one of: {list(SUPPORTED_BOARDS.keys())}"
            )

        # Resolve BrainFlow BoardId enum member
        board_id_name = SUPPORTED_BOARDS[board_type]
        self._board_id = getattr(BoardIds, board_id_name)
        self._board_type = board_type

        # BrainFlowInputParams carries all connection details.
        # Fields not relevant to a given board are simply ignored by BrainFlow.
        params = BrainFlowInputParams()
        if serial_port:  params.serial_port  = serial_port
        if mac_address:  params.mac_address  = mac_address
        if ip_address:   params.ip_address   = ip_address
        if ip_port:      params.ip_port      = ip_port

        # Suppress verbose BrainFlow logging (set before BoardShim init)
        BoardShim.set_log_level(log_level)

        self._board     = BoardShim(self._board_id, params)
        self._streaming = False

        # Cache board metadata after connection (populated in start())
        self._sample_rate:    Optional[int]        = None
        self._eeg_channels:   Optional[list]       = None
        self._timestamp_chan: Optional[int]        = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, buffer_size: int = 45000) -> None:
        """
        Open the board connection and begin streaming EEG data into
        BrainFlow's internal ring-buffer.

        Parameters
        ----------
        buffer_size : int
            Ring-buffer capacity in samples. At 256 Hz, 45000 ≈ 175 seconds.
            Must be larger than the largest `get_latest_data` window you need.
        """
        if self._streaming:
            print("[EEGStreamer] Already streaming — ignoring start() call.")
            return

        print(f"[EEGStreamer] Connecting to board: {self._board_type.upper()} "
              f"(BoardId={self._board_id}) …")

        try:
            self._board.prepare_session()
        except BrainFlowError as exc:
            raise ConnectionError(
                f"[EEGStreamer] Failed to prepare board session: {exc}\n"
                "  • For Bluetooth boards, ensure the headset is powered on and paired.\n"
                "  • For the Synthetic Board this should never fail — check your install."
            ) from exc

        self._board.start_stream(buffer_size)
        self._streaming = True

        # Cache metadata (needs an active session to resolve correctly)
        self._sample_rate    = BoardShim.get_sampling_rate(self._board_id)
        self._eeg_channels   = BoardShim.get_eeg_channels(self._board_id)
        self._timestamp_chan = BoardShim.get_timestamp_channel(self._board_id)

        print(f"[EEGStreamer] Stream started. "
              f"Sample rate: {self._sample_rate} Hz, "
              f"EEG channels: {self._eeg_channels}")

        # Brief warm-up pause: BrainFlow needs ~1 s to populate the ring-buffer
        # before the first meaningful data window is available.
        time.sleep(1.0)

    def stop(self) -> None:
        """Stop streaming and release the board connection."""
        if not self._streaming:
            return
        try:
            self._board.stop_stream()
            self._board.release_session()
        except BrainFlowError as exc:
            print(f"[EEGStreamer] Warning during stop: {exc}")
        finally:
            self._streaming = False
        print("[EEGStreamer] Stream stopped and session released.")

    # ------------------------------------------------------------------
    # Data Retrieval
    # ------------------------------------------------------------------

    def get_latest_data(
        self, window_seconds: float = 2.0
    ) -> Tuple[np.ndarray, dict]:
        """
        Retrieve the last `window_seconds` of EEG data from BrainFlow's
        ring-buffer.

        BrainFlow fills Bluetooth-lost packets with linear interpolation
        internally, so the returned array is always contiguous.

        If the ring-buffer holds fewer samples than requested (e.g., within
        the first seconds of the stream), the shortfall is zero-padded on
        the left so downstream FFT calculations always receive a fixed-length
        array. The `meta['is_padded']` flag indicates this condition.

        Parameters
        ----------
        window_seconds : float
            Duration of the data window in seconds.

        Returns
        -------
        eeg_data : np.ndarray, shape (n_channels, n_samples)
            Raw EEG data in microvolts for each EEG channel.
            n_samples = window_seconds × sample_rate.
        meta : dict
            {
              'sample_rate': int,
              'eeg_channels': list[int],
              'n_samples_requested': int,
              'n_samples_received': int,
              'is_padded': bool,
              'timestamp': float,  # epoch seconds of last sample
            }
        """
        if not self._streaming:
            raise RuntimeError(
                "[EEGStreamer] Not streaming. Call start() first."
            )

        n_requested = int(window_seconds * self._sample_rate)

        try:
            # get_current_board_data(n) returns the *most recent* n samples
            # without removing them from the ring-buffer (non-destructive read).
            raw = self._board.get_current_board_data(n_requested)
        except BrainFlowError as exc:
            raise RuntimeError(
                f"[EEGStreamer] Failed to read from ring-buffer: {exc}"
            ) from exc

        # Extract only the EEG channels (discard accelerometer, aux, etc.)
        eeg_data = raw[self._eeg_channels, :]  # shape: (n_eeg_ch, n_samples_received)

        n_received  = eeg_data.shape[1]
        is_padded   = n_received < n_requested

        # Zero-pad on the left if the buffer hasn't filled yet.
        # Left-padding preserves the most recent (rightmost) samples intact.
        if is_padded:
            pad_width = n_requested - n_received
            eeg_data = np.pad(eeg_data, ((0, 0), (pad_width, 0)), mode="constant")

        # Timestamp of the last sample (UTC epoch seconds)
        timestamp = (
            float(raw[self._timestamp_chan, -1])
            if raw.shape[1] > 0
            else time.time()
        )

        meta = {
            "sample_rate":          self._sample_rate,
            "eeg_channels":         self._eeg_channels,
            "n_samples_requested":  n_requested,
            "n_samples_received":   n_received,
            "is_padded":            is_padded,
            "timestamp":            timestamp,
        }

        return eeg_data, meta

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> Optional[int]:
        return self._sample_rate

    @property
    def eeg_channels(self) -> Optional[list]:
        return self._eeg_channels

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


if __name__ == "__main__":
    print("=== EEGStreamer — Synthetic Board Demo ===")
    with EEGStreamer(board_type="synthetic") as streamer:
        for i in range(3):
            data, meta = streamer.get_latest_data(window_seconds=2.0)
            print(f"[{i+1}] shape={data.shape}, "
                  f"padded={meta['is_padded']}, "
                  f"sr={meta['sample_rate']} Hz")
            time.sleep(1.0)
    print("Done.")
