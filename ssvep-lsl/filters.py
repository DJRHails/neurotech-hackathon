"""Shared EEG filter chain for real-time IIR filtering.

Provides bandpass + notch filtering suitable for continuous
EEG streams processed chunk-by-chunk with maintained state.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def build_filter(
    fs: float,
    n_ch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the standard EEG filter chain.

    Bandpass 1-45 Hz, notch 50 Hz, notch 60 Hz.
    Returns (sos, zi) with zi shaped for n_ch channels.

    Args:
        fs: Sampling frequency in Hz.
        n_ch: Number of EEG channels.
    """
    bp = butter(2, [1.0, 45.0], btype="band", fs=fs, output="sos")
    n50 = butter(2, [49.0, 51.0], btype="bandstop", fs=fs, output="sos")
    n60 = butter(2, [59.0, 61.0], btype="bandstop", fs=fs, output="sos")
    sos = np.vstack([bp, n50, n60])
    zi_single = sosfilt_zi(sos)  # (n_sections, 2)
    zi = np.repeat(zi_single[:, :, np.newaxis], n_ch, axis=2)
    return sos, zi


def apply_filter(
    data: np.ndarray,
    sos: np.ndarray,
    zi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply IIR filter chain to a chunk of data.

    Args:
        data: EEG samples, shape (n_samples, n_ch).
        sos: Second-order sections from build_filter.
        zi: Filter state from build_filter or previous call.

    Returns:
        (filtered_data, updated_zi) tuple.
    """
    filtered, zi = sosfilt(sos, data, axis=0, zi=zi)
    return filtered.astype(np.float32), zi
