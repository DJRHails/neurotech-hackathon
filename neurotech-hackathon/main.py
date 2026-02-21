"""SSVEP Brain-Computer Interface with Unicorn Hybrid Black.

Sources (pick one):
  - Default on macOS: simulated sine source (10 Hz)
  - --lsl flag: receive from LSL stream (cross-machine)

Usage:
    uv run python main.py          # simulated source
    uv run python main.py --lsl    # receive from Windows LSL
"""

import argparse
import math
import os
import sys
import time as _time
from pathlib import Path

# Load .env before any imports that need PYLSL_LIB
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import numpy as np
import pyqtgraph as pg
from images import Image, Video, generate_images
from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QMainWindow,
    QWidget,
)
from scipy.signal import butter, sosfilt, sosfilt_zi
from ssvep_detector import SSVEPDetector
from ssvep_stimulus import SSVEPStimulus

SAMPLING_RATE = 250.0
CHANNEL_COUNT = 8
FFT_WINDOW = 250
FFT_OVERLAP = 0.5
PROCESS_HZ = 25
PROCESS_MS = 1000 // PROCESS_HZ


# --- Signal processing ---


class SignalProcessor:
    """Real-time IIR filter chain + rolling FFT."""

    def __init__(
        self,
        fs: float = SAMPLING_RATE,
        n_ch: int = CHANNEL_COUNT,
    ) -> None:
        self._fs = fs
        self._n_ch = n_ch

        # Bandpass 1-45 Hz, order 2
        bp = butter(2, [1.0, 45.0], btype="band", fs=fs, output="sos")
        # Notch 50 Hz (bandstop 49-51)
        n50 = butter(2, [49.0, 51.0], btype="bandstop", fs=fs, output="sos")
        # Notch 60 Hz (bandstop 59-61)
        n60 = butter(2, [59.0, 61.0], btype="bandstop", fs=fs, output="sos")

        self._sos = np.vstack([bp, n50, n60])
        zi_single = sosfilt_zi(self._sos)  # (n_sections, 2)
        # Broadcast to (n_sections, 2, n_ch)
        self._zi = np.repeat(
            zi_single[:, :, np.newaxis], n_ch, axis=2
        )

        # Rolling FFT buffer
        self._fft_buf = np.zeros(
            (FFT_WINDOW, n_ch), dtype=np.float32
        )
        self._fft_pos = 0
        self._fft_ready = False
        hop = int(FFT_WINDOW * (1 - FFT_OVERLAP))
        self._fft_hop = hop
        self._fft_since_last = 0
        self._hamming = np.hamming(FFT_WINDOW).astype(
            np.float32
        )
        logger.debug(
            "SignalProcessor created | fs={} channels={} "
            "sos_sections={} fft_window={} fft_hop={}",
            fs, n_ch, self._sos.shape[0], FFT_WINDOW, hop,
        )

    def filter(self, data: np.ndarray) -> np.ndarray:
        """Apply IIR filter chain. data: (n_samples, n_ch)."""
        logger.debug(
            "filter input shape={} dtype={}", data.shape, data.dtype
        )
        filtered, self._zi = sosfilt(
            self._sos, data, axis=0, zi=self._zi
        )
        out = filtered.astype(np.float32)
        logger.debug(
            "filter output range=[{:.2f}, {:.2f}]",
            float(out.min()), float(out.max()),
        )
        return out

    def push_fft(
        self, data: np.ndarray
    ) -> np.ndarray | None:
        """Push filtered data into rolling FFT buffer.

        Returns (n_bins, n_ch) magnitude spectrum when a new
        FFT frame is ready, else None.
        """
        n = data.shape[0]
        result = None

        for i in range(n):
            self._fft_buf[self._fft_pos] = data[i]
            self._fft_pos = (self._fft_pos + 1) % FFT_WINDOW
            self._fft_since_last += 1

            if not self._fft_ready:
                if self._fft_pos == 0:
                    self._fft_ready = True
                    self._fft_since_last = 0
                    result = self._compute_fft()
                    logger.debug("FFT buffer filled, first frame computed")
            elif self._fft_since_last >= self._fft_hop:
                self._fft_since_last = 0
                result = self._compute_fft()

        if result is not None:
            logger.debug(
                "push_fft: new spectrum shape={} "
                "peak_mag={:.2f}",
                result.shape, float(result.max()),
            )
        return result

    def _compute_fft(self) -> np.ndarray:
        # Reorder circular buffer to contiguous
        buf = np.roll(
            self._fft_buf, -self._fft_pos, axis=0
        )
        windowed = buf * self._hamming[:, None]
        spectrum = np.abs(np.fft.rfft(windowed, axis=0))
        return spectrum.astype(np.float32)


# --- Simulated EEG source ---


class SimulatedSource:
    """Generates sine + noise EEG at 250 Hz, wall-clock paced."""

    def __init__(
        self,
        fs: float = SAMPLING_RATE,
        n_ch: int = CHANNEL_COUNT,
        amplitude: float = 50.0,
        noise: float = 2.0,
    ) -> None:
        self._fs = fs
        self._n_ch = n_ch
        self._amplitude = amplitude
        self._noise = noise
        self.signal_frequency = 10.0
        self._phase = 0.0
        self._t0: float | None = None
        self._samples_emitted = 0
        logger.debug(
            "SimulatedSource created | fs={} channels={} "
            "amplitude={} noise={}",
            fs, n_ch, amplitude, noise,
        )

    def start(self) -> None:
        self._t0 = _time.monotonic()
        self._samples_emitted = 0
        self._phase = 0.0
        logger.info(
            "SimulatedSource started | freq={:.1f}Hz",
            self.signal_frequency,
        )

    def stop(self) -> None:
        logger.info(
            "SimulatedSource stopped | total_samples={}",
            self._samples_emitted,
        )
        self._t0 = None

    def pull_chunk(self) -> np.ndarray | None:
        """Return samples accumulated since last call."""
        if self._t0 is None:
            return None
        elapsed = _time.monotonic() - self._t0
        total = int(elapsed * self._fs)
        n = total - self._samples_emitted
        if n <= 0:
            return None
        self._samples_emitted = total

        t = np.arange(n, dtype=np.float32) / self._fs
        phase_inc = (
            2.0 * np.pi * self.signal_frequency * t
            + self._phase
        )
        self._phase = float(phase_inc[-1]) if n > 0 else self._phase
        sine = self._amplitude * np.sin(phase_inc)

        data = np.tile(sine[:, None], (1, self._n_ch))
        data += self._noise * np.random.randn(
            n, self._n_ch
        ).astype(np.float32)
        logger.debug(
            "pull_chunk: {} samples, freq={:.1f}Hz, "
            "elapsed={:.3f}s",
            n, self.signal_frequency, elapsed,
        )
        return data


# --- Pyqtgraph scope widgets ---


class TimeSeriesScope(pg.PlotWidget):
    """Circular-buffer time series with channel stacking."""

    def __init__(
        self,
        n_ch: int = CHANNEL_COUNT,
        fs: float = SAMPLING_RATE,
        time_window: float = 5.0,
    ) -> None:
        super().__init__(title="EEG")
        self._n_ch = n_ch
        self._fs = fs
        buf_len = int(fs * time_window)
        self._buf = np.zeros(
            (buf_len, n_ch), dtype=np.float32
        )
        self._pos = 0
        self._filled = False

        self._t = np.linspace(
            -time_window, 0, buf_len
        )

        self._curves = []
        for i in range(n_ch):
            pen = pg.intColor(i, hues=n_ch)
            curve = self.plot(pen=pen)
            self._curves.append(curve)

        self.setLabel("bottom", "Time", "s")
        self.setLabel("left", "Amplitude", "uV")
        self.enableAutoRange(axis="y")
        logger.debug(
            "TimeSeriesScope created | channels={} "
            "buf_len={} window={}s",
            n_ch, buf_len, time_window,
        )

    def push(self, data: np.ndarray) -> None:
        """Append (n_samples, n_ch) data."""
        n = data.shape[0]
        buf_len = self._buf.shape[0]
        if n >= buf_len:
            self._buf[:] = data[-buf_len:]
            self._pos = 0
            self._filled = True
        else:
            end = self._pos + n
            if end <= buf_len:
                self._buf[self._pos:end] = data
            else:
                first = buf_len - self._pos
                self._buf[self._pos:] = data[:first]
                self._buf[:end - buf_len] = data[first:]
                self._filled = True
            self._pos = end % buf_len

    def refresh(self) -> None:
        if self._filled:
            buf = np.roll(self._buf, -self._pos, axis=0)
        else:
            buf = self._buf[:self._pos]
            if buf.shape[0] == 0:
                return

        spacing = 100.0
        for i in range(self._n_ch):
            offset = (self._n_ch - 1 - i) * spacing
            y = buf[:, i] + offset
            t = self._t[-len(y):]
            self._curves[i].setData(t, y)


class SpectrumScope(pg.PlotWidget):
    """FFT magnitude spectrum display."""

    def __init__(
        self,
        fs: float = SAMPLING_RATE,
        n_ch: int = CHANNEL_COUNT,
    ) -> None:
        super().__init__(title="Spectrum")
        self._fs = fs
        self._n_ch = n_ch
        self._spectrum: np.ndarray | None = None

        self._curves = []
        for i in range(n_ch):
            pen = pg.intColor(i, hues=n_ch)
            curve = self.plot(pen=pen)
            self._curves.append(curve)

        self.setLabel("bottom", "Frequency", "Hz")
        self.setLabel("left", "Magnitude")
        self.setXRange(0, 50)
        self.enableAutoRange(axis="y")
        logger.debug(
            "SpectrumScope created | fs={} channels={}", fs, n_ch
        )

    def set_spectrum(self, spectrum: np.ndarray) -> None:
        """Set new spectrum data (n_bins, n_ch)."""
        self._spectrum = spectrum

    def refresh(self) -> None:
        if self._spectrum is None:
            return
        n_bins = self._spectrum.shape[0]
        freqs = np.linspace(
            0, self._fs / 2, n_bins
        )
        for i in range(min(self._n_ch, self._spectrum.shape[1])):
            self._curves[i].setData(
                freqs, self._spectrum[:, i]
            )


class SSVEPPowerScope(pg.PlotWidget):
    """Labeled 4-channel power scope for SSVEP detector output."""

    _LABELS = ["10 Hz", "15 Hz", "GT sin", "GT cos"]
    _COLORS = ["#EF4444", "#3B82F6", "#A3A3A3", "#737373"]

    def __init__(
        self,
        time_window: float = 10.0,
        update_rate: float = PROCESS_HZ,
    ) -> None:
        super().__init__(title="SSVEP Power")
        buf_len = int(update_rate * time_window)
        self._buf = np.zeros((buf_len, 4), dtype=np.float32)
        self._pos = 0
        self._filled = False

        self._t = np.linspace(
            -time_window, 0, buf_len
        )

        self._curves = []
        self.addLegend()
        for i, (label, color) in enumerate(
            zip(self._LABELS, self._COLORS, strict=True)
        ):
            pen = pg.mkPen(color, width=2)
            curve = self.plot(pen=pen, name=label)
            self._curves.append(curve)

        self.setLabel("bottom", "Time", "s")
        self.setLabel("left", "Power", "a.u.")
        self.enableAutoRange(axis="y")
        logger.debug(
            "SSVEPPowerScope created | buf_len={} window={}s",
            buf_len, time_window,
        )

    def push(self, data: np.ndarray) -> None:
        """Append (n_samples, 4) data."""
        n = data.shape[0]
        buf_len = self._buf.shape[0]
        if n >= buf_len:
            self._buf[:] = data[-buf_len:]
            self._pos = 0
            self._filled = True
        else:
            end = self._pos + n
            if end <= buf_len:
                self._buf[self._pos:end] = data
            else:
                first = buf_len - self._pos
                self._buf[self._pos:] = data[:first]
                self._buf[:end - buf_len] = data[first:]
                self._filled = True
            self._pos = end % buf_len

    def refresh(self) -> None:
        if self._filled:
            buf = np.roll(self._buf, -self._pos, axis=0)
        else:
            buf = self._buf[:self._pos]
            if buf.shape[0] == 0:
                return
        t = self._t[-buf.shape[0]:]
        for i in range(4):
            self._curves[i].setData(t, buf[:, i])


# --- Main application ---


def _make_source(use_lsl: bool):
    """Create EEG source based on flags."""
    if use_lsl:
        logger.info("Creating LSL source")
        from lsl_source import LSLSource
        return LSLSource(stream_name="lock_in_eeg_processed")
    logger.info("Creating simulated source")
    return SimulatedSource()


def main() -> None:
    parser = argparse.ArgumentParser(description="SSVEP BCI")
    parser.add_argument(
        "--lsl",
        action="store_true",
        help="Receive EEG via LSL instead of simulated source",
    )
    parser.add_argument(
        "--prompt",
        nargs=2,
        metavar=("A", "B"),
        default=[
            "a calm blue ocean wave",
            "a bright red fire flame",
        ],
        help="Two image prompts for Grok generation",
    )
    args = parser.parse_args()

    logger.info(
        "Starting SSVEP BCI | lsl={} prompts={!r}",
        args.lsl, args.prompt,
    )

    qapp = QApplication(sys.argv)

    # --- Stimulus window ---
    video_path = Path(__file__).parent / "assets" / "subway_surfers.mp4"
    logger.debug("Checking video path: {}", video_path)
    generated = generate_images(args.prompt[0], args.prompt[1])

    stimuli: list[Video | Image] = []
    if video_path.exists():
        logger.info("Loading video stimulus: {}", video_path)
        stimuli.append(Video(video_path))
    elif generated:
        stimuli.append(generated[0])
    if generated:
        stimuli.append(generated[1])

    logger.debug("Stimulus count: {}", len(stimuli))
    stimulus = SSVEPStimulus(stimuli=stimuli or None)
    stimulus.setGeometry(100, 100, 700, 500)

    # --- Scope window ---
    scope_win = QMainWindow()
    scope_win.setWindowTitle("SSVEP Scopes")
    scope_win.setGeometry(820, 100, 900, 600)

    central = QWidget()
    scope_win.setCentralWidget(central)
    grid = QGridLayout(central)

    eeg_scope = TimeSeriesScope()
    spectrum_scope = SpectrumScope()
    ssvep_scope = SSVEPPowerScope()

    # EEG scope spans top row
    grid.addWidget(eeg_scope, 0, 0, 1, 2)
    grid.addWidget(spectrum_scope, 1, 0)
    grid.addWidget(ssvep_scope, 1, 1)
    logger.debug("Scope window layout built")

    # --- Processing pipeline ---
    source = _make_source(use_lsl=args.lsl)
    processor = SignalProcessor()
    detector = SSVEPDetector(on_detect=stimulus.set_detection)

    # Frequency sweep for simulated source
    sweep_period = 10.0
    t0 = _time.monotonic()
    _tick = 0

    def _process() -> None:
        nonlocal _tick
        _tick += 1
        chunk = source.pull_chunk()
        if chunk is None:
            logger.debug("tick {}: no data from source", _tick)
            return

        logger.debug(
            "tick {}: chunk shape={}", _tick, chunk.shape
        )
        filtered = processor.filter(chunk)
        eeg_scope.push(filtered)

        spectrum = processor.push_fft(filtered)
        if spectrum is not None:
            spectrum_scope.set_spectrum(spectrum)
            power = detector.process(spectrum)
            ssvep_scope.push(power)
            logger.debug(
                "tick {}: FFT + detector ran", _tick
            )

        # Sweep frequency if simulated
        if isinstance(source, SimulatedSource):
            t = _time.monotonic() - t0
            phase = 2.0 * math.pi * t / sweep_period
            new_freq = 12.5 + 2.5 * math.sin(phase)
            source.signal_frequency = new_freq
            logger.debug(
                "tick {}: sweep freq={:.2f}Hz", _tick, new_freq
            )

        eeg_scope.refresh()
        spectrum_scope.refresh()
        ssvep_scope.refresh()

    process_timer = QTimer()
    process_timer.timeout.connect(_process)

    # Start everything
    logger.info(
        "Starting pipeline | process_hz={} process_ms={}",
        PROCESS_HZ, PROCESS_MS,
    )
    source.start()
    stimulus.show()
    stimulus.start()
    scope_win.show()
    process_timer.start(PROCESS_MS)

    logger.info("Entering Qt event loop")
    try:
        qapp.exec()
    finally:
        logger.info("Shutting down")
        process_timer.stop()
        stimulus.stop()
        source.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
