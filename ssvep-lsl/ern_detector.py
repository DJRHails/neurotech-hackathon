"""Threshold-based Error-Related Negativity (ERN) detector.

Monitors frontal EEG channels for brief negative deflections
characteristic of ERN events. Uses baseline-corrected amplitude
with a refractory period to avoid double-counting.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ERNHit:
    """A detected ERN event."""

    timestamp: float
    amplitude: float
    channel: int


class ERNDetector:
    """Sliding-window ERN detector for frontal EEG channels.

    Args:
        s_rate: Sampling rate in Hz.
        frontal_channels: Channel indices to monitor (default 0-2
            for Fz/FCz/Cz).
        baseline_ms: Duration of baseline window in ms.
        detection_window_ms: Window in which the negative peak
            must occur after baseline, in ms.
        threshold_uv: Negative amplitude threshold in uV
            (e.g., -5.0 means deflections below -5 uV trigger).
        refractory_ms: Minimum interval between detections in ms.
    """

    def __init__(
        self,
        s_rate: int,
        frontal_channels: list[int] | None = None,
        baseline_ms: float = 500.0,
        detection_window_ms: float = 150.0,
        threshold_uv: float = -5.0,
        refractory_ms: float = 500.0,
    ) -> None:
        self._s_rate = s_rate
        self._channels = frontal_channels or [0, 1, 2]
        self._baseline_len = int(baseline_ms * s_rate / 1000)
        self._detect_len = int(detection_window_ms * s_rate / 1000)
        self._window_len = self._baseline_len + self._detect_len
        self._threshold = threshold_uv
        self._refractory = refractory_ms / 1000.0

        self._buffer: np.ndarray = np.empty((0, len(self._channels)))
        self._last_hit_time = -1.0
        self._total_hits = 0

    @property
    def total_hits(self) -> int:
        return self._total_hits

    @property
    def window_samples(self) -> int:
        """Minimum samples needed before detection can run."""
        return self._window_len

    def detect(
        self,
        data: np.ndarray,
        timestamp: float,
    ) -> list[ERNHit]:
        """Check filtered EEG data for ERN-like deflections.

        Args:
            data: Filtered EEG, shape (n_samples, n_channels_total).
                Only frontal_channels columns are used.
            timestamp: Wall-clock time of the last sample.

        Returns:
            List of ERNHit for each detection in this chunk.
        """
        frontal = data[:, self._channels]
        self._buffer = np.concatenate(
            (self._buffer, frontal), axis=0
        )

        # Keep only what we need (window + some margin)
        max_keep = self._window_len * 2
        if self._buffer.shape[0] > max_keep:
            self._buffer = self._buffer[-max_keep:]

        if self._buffer.shape[0] < self._window_len:
            return []

        hits: list[ERNHit] = []
        window = self._buffer[-self._window_len:]
        baseline = window[: self._baseline_len]
        detection = window[self._baseline_len:]

        baseline_mean = baseline.mean(axis=0)
        corrected = detection - baseline_mean

        for ch_idx, ch in enumerate(self._channels):
            min_val = float(corrected[:, ch_idx].min())
            if min_val < self._threshold:
                if timestamp - self._last_hit_time < self._refractory:
                    continue
                self._last_hit_time = timestamp
                self._total_hits += 1
                hits.append(ERNHit(
                    timestamp=timestamp,
                    amplitude=min_val,
                    channel=ch,
                ))
                break  # One hit per time window

        return hits
