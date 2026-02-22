"""LSL receiver for EEG data.

Non-blocking pull_chunk() for use with QTimer-based processing loops.
"""

import numpy as np
from loguru import logger
from pylsl import StreamInlet, resolve_byprop


class LSLSource:
    """Receives EEG samples from an LSL stream."""

    def __init__(
        self,
        stream_name: str = "lock_in_eeg_processed",
        channel_count: int = 8,
    ) -> None:
        self._stream_name = stream_name
        self._channel_count = channel_count
        self._inlet: StreamInlet | None = None
        logger.debug(
            "LSLSource created | stream={!r} channels={}",
            stream_name, channel_count,
        )

    def start(self) -> None:
        logger.info("Resolving LSL stream {!r}...", self._stream_name)
        streams = resolve_byprop("name", self._stream_name, 1, 10)
        if not streams:
            logger.error(
                "No LSL stream {!r} found after 10 s timeout",
                self._stream_name,
            )
            raise RuntimeError(
                f"No LSL stream '{self._stream_name}' found. "
                "Is the Windows streamer running?"
            )
        logger.debug(
            "Found {} LSL stream(s), using first", len(streams)
        )
        self._inlet = StreamInlet(streams[0])
        logger.info(
            "Connected to LSL stream {!r}", self._stream_name
        )

    def stop(self) -> None:
        if self._inlet:
            logger.debug("Closing LSL inlet")
            self._inlet.close_stream()
            self._inlet = None
            logger.info("LSL inlet closed")

    def pull_chunk(
        self, timeout: float = 0.0
    ) -> np.ndarray | None:
        """Pull available samples. Returns (n_samples, channels) or None."""
        if self._inlet is None:
            return None
        chunk, timestamps = self._inlet.pull_chunk(timeout=timeout)
        if not chunk:
            logger.debug("pull_chunk: no data available")
            return None
        arr = np.array(chunk, dtype=np.float32)
        logger.debug(
            "pull_chunk: got {} samples, raw shape={}",
            arr.shape[0], arr.shape,
        )
        if arr.shape[1] > self._channel_count:
            arr = arr[:, : self._channel_count]
            logger.debug(
                "Trimmed to {} channels", self._channel_count
            )
        elif arr.shape[1] < self._channel_count:
            pad = np.zeros(
                (arr.shape[0], self._channel_count - arr.shape[1]),
                dtype=np.float32,
            )
            arr = np.hstack([arr, pad])
            logger.debug(
                "Padded from {} to {} channels",
                arr.shape[1] - pad.shape[1], self._channel_count,
            )
        return arr
