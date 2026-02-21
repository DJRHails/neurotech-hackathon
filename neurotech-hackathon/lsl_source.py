"""LSL receiver source node for gpype pipeline.

Receives EEG data from a Lab Streaming Layer outlet (e.g. from
a Windows machine running the Unicorn Hybrid Black) and feeds it
into the local gpype pipeline.
"""

import logging
import threading
import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from gpype.backend.core.o_port import OPort
from gpype.backend.sources.base.fixed_rate_source import (
    FixedRateSource,
)
from gpype.common.constants import Constants

log = logging.getLogger(__name__)

OUT_PORT = Constants.Defaults.PORT_OUT


class LSLSource(FixedRateSource):
    """Receive EEG data from an LSL stream on the network."""

    DEFAULT_STREAM_NAME = "unicorn_eeg"

    def __init__(
        self,
        stream_name: str | None = None,
        sampling_rate: float = 250.0,
        channel_count: int = 8,
        frame_size: int = 4,
        **kwargs,
    ) -> None:
        if stream_name is None:
            stream_name = self.DEFAULT_STREAM_NAME
        self._stream_name = stream_name

        output_ports = kwargs.pop(
            "output_ports", [OPort.Configuration()]
        )
        decimation_factor = kwargs.pop(
            "decimation_factor", frame_size
        )

        FixedRateSource.__init__(
            self,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            decimation_factor=decimation_factor,
            output_ports=output_ports,
            **kwargs,
        )

        self._inlet: StreamInlet | None = None
        self._buffer = np.zeros(
            (frame_size, channel_count),
            dtype=Constants.DATA_TYPE,
        )

    def start(self) -> None:
        """Resolve the LSL stream and start receiving."""
        log.info(
            "Resolving LSL stream '%s'...", self._stream_name
        )
        streams = resolve_byprop("name", self._stream_name, 1, 10)
        if not streams:
            raise RuntimeError(
                f"No LSL stream '{self._stream_name}' found. "
                "Is the Windows streamer running?"
            )
        self._inlet = StreamInlet(streams[0])
        log.info("Connected to LSL stream '%s'", self._stream_name)
        FixedRateSource.start(self)

    def stop(self) -> None:
        """Close the LSL inlet."""
        FixedRateSource.stop(self)
        if self._inlet:
            self._inlet.close_stream()
            self._inlet = None

    def step(
        self, data: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray] | None:
        if not self.is_decimation_step():
            return None

        if self._inlet is None:
            return None

        config = self.config
        frame_size = config[self.Configuration.Keys.FRAME_SIZE][0]
        ch_count = config[self.Configuration.Keys.CHANNEL_COUNT][0]

        chunk, _ = self._inlet.pull_chunk(
            max_samples=frame_size, timeout=0.1
        )
        if not chunk or len(chunk) < frame_size:
            return None

        # Take exactly frame_size samples (discard excess)
        arr = np.array(
            chunk[:frame_size], dtype=Constants.DATA_TYPE
        )
        # Trim or pad channels to match expected count
        if arr.shape[1] > ch_count:
            arr = arr[:, :ch_count]
        elif arr.shape[1] < ch_count:
            pad = np.zeros(
                (frame_size, ch_count - arr.shape[1]),
                dtype=Constants.DATA_TYPE,
            )
            arr = np.hstack([arr, pad])

        return {OUT_PORT: arr}
