"""SSVEP frequency detector node.

Receives FFT magnitude data, compares power at two target
frequencies (with harmonic boost), and classifies which stimulus
the user is attending to. Outputs a 2-channel signal of
[power_left, power_right] for downstream visualization.
"""

import logging
from collections.abc import Callable

import numpy as np

from gpype.backend.core.io_node import IONode
from gpype.backend.core.i_port import IPort
from gpype.backend.core.o_port import OPort
from gpype.common.constants import Constants

from ssvep_stimulus import DETECT_LEFT, DETECT_NONE, DETECT_RIGHT

log = logging.getLogger(__name__)

# Target frequencies (must match SSVEPStimulus config)
FREQ_LEFT = 10   # Hz
FREQ_RIGHT = 15  # Hz

# Occipital channels: PO7, Oz, PO8 (0-indexed)
DEFAULT_CHANNELS = [5, 6, 7]

# Classification threshold: ratio of max/min power
POWER_RATIO_THRESHOLD = 1.3

# Weight of 2nd harmonic relative to fundamental
HARMONIC_WEIGHT = 0.5

_LABELS = {
    DETECT_NONE: "NONE",
    DETECT_LEFT: "LEFT",
    DETECT_RIGHT: "RIGHT",
}


class SSVEPDetector(IONode):
    """Classifies SSVEP from FFT output.

    Outputs 2-channel power signal [left, right] for visualization.
    """

    def __init__(
        self,
        on_detect: Callable[[int], None],
        channels: list[int] | None = None,
    ) -> None:
        input_ports = [IPort.Configuration(name="in")]
        output_ports = [OPort.Configuration(name="out")]
        IONode.__init__(
            self,
            input_ports=input_ports,
            output_ports=output_ports,
        )
        self._on_detect = on_detect
        self._channels = channels or DEFAULT_CHANNELS
        self._last_result = DETECT_NONE

    def setup(
        self,
        data: dict[str, np.ndarray],
        port_context_in: dict[str, dict],
    ) -> dict[str, dict]:
        port_context_out = super().setup(data, port_context_in)
        port_context_out["out"][Constants.Keys.CHANNEL_COUNT] = 2
        return port_context_out

    def step(
        self, data: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        spectrum = data.get("in")
        if spectrum is None:
            return {}

        # spectrum shape: (freq_bins, channel_count)
        # At 1 Hz resolution (250 Hz / 250 samples), bin = freq
        n_bins = spectrum.shape[0]

        power_left = self._band_power(spectrum, FREQ_LEFT, n_bins)
        power_right = self._band_power(
            spectrum, FREQ_RIGHT, n_bins
        )

        result = self._classify(power_left, power_right)

        if result != self._last_result:
            log.info("SSVEP detection: %s", _LABELS[result])
            self._last_result = result

        self._on_detect(result)

        return {
            "out": np.array(
                [[power_left, power_right]], dtype=np.float32
            )
        }

    def _classify(
        self, power_left: float, power_right: float
    ) -> int:
        if power_left == 0 and power_right == 0:
            return DETECT_NONE

        mn = min(power_left, power_right)
        mx = max(power_left, power_right)
        if mn == 0 or mx / mn < POWER_RATIO_THRESHOLD:
            return DETECT_NONE

        if power_left > power_right:
            return DETECT_LEFT
        return DETECT_RIGHT

    def _band_power(
        self,
        spectrum: np.ndarray,
        freq: int,
        n_bins: int,
    ) -> float:
        """Sum power at fundamental + weighted 2nd harmonic."""
        chans = [
            c for c in self._channels
            if c < spectrum.shape[1]
        ]
        if not chans:
            chans = list(range(spectrum.shape[1]))

        power = 0.0
        if freq < n_bins:
            power += float(
                np.mean(spectrum[freq, chans]) ** 2
            )

        harmonic = freq * 2
        if harmonic < n_bins:
            power += HARMONIC_WEIGHT * float(
                np.mean(spectrum[harmonic, chans]) ** 2
            )

        return power
