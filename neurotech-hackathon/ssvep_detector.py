"""SSVEP frequency detector.

Receives FFT magnitude data, compares power at two target
frequencies (with harmonic boost), and classifies which stimulus
the user is attending to. Outputs a 4-channel signal of
[power_left, power_right, gt_sin, gt_cos] for visualization.
"""

import time as _time
from collections.abc import Callable
from math import cos, pi, sin

import numpy as np
from loguru import logger

from ssvep_stimulus import DETECT_LEFT, DETECT_NONE, DETECT_RIGHT

# Target frequencies (must match SSVEPStimulus config)
FREQ_LEFT = 10   # Hz
FREQ_RIGHT = 15  # Hz

# Occipital channels: PO7, Oz, PO8 (0-indexed)
DEFAULT_CHANNELS = [5, 6, 7]

# Classification threshold: ratio of max/min power
POWER_RATIO_THRESHOLD = 1.3

# +/-BANDWIDTH Hz around each target frequency
BANDWIDTH = 2

# Weight of 2nd harmonic relative to fundamental
HARMONIC_WEIGHT = 0.5

# Interpolation samples per FFT step for smooth scope rendering
INTERP_SAMPLES = 20

_LABELS = {
    DETECT_NONE: "NONE",
    DETECT_LEFT: "LEFT",
    DETECT_RIGHT: "RIGHT",
}


class SSVEPDetector:
    """Classifies SSVEP from FFT output.

    Outputs 4-channel power signal for visualization.
    """

    def __init__(
        self,
        on_detect: Callable[[int], None],
        channels: list[int] | None = None,
    ) -> None:
        self._on_detect = on_detect
        self._channels = channels or DEFAULT_CHANNELS
        self._last_result = DETECT_NONE
        self._t0 = _time.monotonic()
        self._prev = np.zeros(4, dtype=np.float32)
        logger.debug(
            "SSVEPDetector created | channels={} threshold={}",
            self._channels, POWER_RATIO_THRESHOLD,
        )

    def process(
        self, spectrum: np.ndarray
    ) -> np.ndarray:
        """Process FFT spectrum, return (INTERP_SAMPLES, 4) array."""
        n_bins = spectrum.shape[0]
        logger.debug(
            "process: spectrum shape={} n_bins={}",
            spectrum.shape, n_bins,
        )

        power_left = self._band_power(spectrum, FREQ_LEFT, n_bins)
        power_right = self._band_power(
            spectrum, FREQ_RIGHT, n_bins
        )
        logger.debug(
            "Band power | left={:.4f} right={:.4f}",
            power_left, power_right,
        )

        result = self._classify(power_left, power_right)

        if result != self._last_result:
            logger.info(
                "SSVEP detection changed: {} -> {}",
                _LABELS[self._last_result], _LABELS[result],
            )
            self._last_result = result
        else:
            logger.debug("SSVEP steady: {}", _LABELS[result])

        self._on_detect(result)

        # Ground truth: smooth sine/cosine scaled to +/-40
        t = _time.monotonic() - self._t0
        phase = 2.0 * pi * t / 10.0  # 10s period
        gt_sin = 40.0 * sin(phase)
        gt_cos = 40.0 * cos(phase)

        # Scale power to similar range as scope (+/-50 uV)
        scale = 40.0 / max(power_left, power_right, 1e-9)

        curr = np.array(
            [power_left * scale, power_right * scale,
             gt_sin, gt_cos],
            dtype=np.float32,
        )

        # Linearly interpolate from previous to current
        t_interp = np.linspace(0, 1, INTERP_SAMPLES)
        out = (
            self._prev[None, :] * (1 - t_interp[:, None])
            + curr[None, :] * t_interp[:, None]
        )
        self._prev = curr

        logger.debug(
            "process output shape={} scale={:.2f}", out.shape, scale
        )
        return out

    def _classify(
        self, power_left: float, power_right: float
    ) -> int:
        if power_left == 0 and power_right == 0:
            return DETECT_NONE

        mn = min(power_left, power_right)
        mx = max(power_left, power_right)
        ratio = mx / mn if mn > 0 else float("inf")
        logger.debug(
            "classify | min={:.4f} max={:.4f} ratio={:.2f}",
            mn, mx, ratio,
        )
        if mn == 0 or ratio < POWER_RATIO_THRESHOLD:
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
        """Sum power in +/-BANDWIDTH band around fundamental + harmonic."""
        chans = [
            c for c in self._channels
            if c < spectrum.shape[1]
        ]
        if not chans:
            chans = list(range(spectrum.shape[1]))

        power = 0.0
        lo = max(0, freq - BANDWIDTH)
        hi = min(n_bins, freq + BANDWIDTH + 1)
        if lo < hi:
            band = spectrum[lo:hi][:, chans]
            power += float(np.mean(band) ** 2)
            logger.debug(
                "band_power {}Hz fundamental bins[{}:{}] "
                "power={:.6f}",
                freq, lo, hi, float(np.mean(band) ** 2),
            )

        harmonic = freq * 2
        lo_h = max(0, harmonic - BANDWIDTH)
        hi_h = min(n_bins, harmonic + BANDWIDTH + 1)
        if lo_h < hi_h:
            band_h = spectrum[lo_h:hi_h][:, chans]
            h_power = HARMONIC_WEIGHT * float(
                np.mean(band_h) ** 2
            )
            power += h_power
            logger.debug(
                "band_power {}Hz harmonic bins[{}:{}] "
                "weighted={:.6f}",
                freq, lo_h, hi_h, h_power,
            )

        return power
