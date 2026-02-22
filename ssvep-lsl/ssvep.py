"""SSVEP visual stimulation with real-time CCA classification."""

import time
from threading import Lock

import numpy as np
from loguru import logger
from psychopy import visual
from psychopy_visionscience.radial import RadialStim

from analysis import CCAAnalysis

SCORE_THRESHOLD = 0.1


class CheckerBoard:
    """Flickering radial checkerboard stimulus.

    Args:
        window: Psychopy window.
        size: Stimulus size (width, height) in norm units.
        position: Screen position (x, y) in norm units.
        n_frame: Frames per half-cycle (freq = refresh_rate / n_frame).
        log_time: Record toggle timestamps for statistics.
    """

    def __init__(
        self,
        window: visual.Window,
        size: tuple[float, float],
        position: tuple[float, float],
        n_frame: int,
        *,
        log_time: bool = False,
    ) -> None:
        self._window = window
        self._fr_rate = n_frame
        self._fr_counter = n_frame
        pattern = np.ones((4, 4))
        pattern[::2, ::2] *= -1
        pattern[1::2, 1::2] *= -1
        stim_kwargs = {
            "win": window,
            "pos": position,
            "size": size,
            "radialCycles": 1,
            "texRes": 256,
            "opacity": 1,
        }
        self._stim1 = RadialStim(tex=pattern, **stim_kwargs)
        self._stim2 = RadialStim(tex=pattern * -1, **stim_kwargs)
        self._toggle = False
        self.log_time = log_time
        self.toggle_times: list[float] = []

    def draw(self) -> None:
        """Draw one frame, toggling at the configured rate."""
        stim = self._stim1 if self._toggle else self._stim2
        stim.draw()
        self._fr_counter -= 1
        if self._fr_counter == 0:
            if self.log_time:
                self.toggle_times.append(time.time())
            self._fr_counter = self._fr_rate
            self._toggle = not self._toggle

    def get_statistics(self) -> tuple[float, float]:
        """Return (mean, std) of toggle intervals in seconds."""
        assert self.log_time, "Time logging not enabled."
        diffs = np.diff(np.array(self.toggle_times))
        return float(diffs.mean()), float(diffs.std())


class SSVEPRealTime:
    """Real-time SSVEP experiment with CCA classification.

    EEG data is fed via `push_eeg()` from a background LSL
    reader thread. Classification runs on each psychopy frame
    when enough data has accumulated in the sliding window.

    Args:
        frame_rates: Frames per half-cycle for each target.
        positions: Screen positions for each target.
        labels: Unicode labels shown on detection.
        signal_len: CCA analysis window length (seconds).
        eeg_s_rate: EEG sampling rate (Hz).
        overlap: Fraction of buffer to retain after analysis.
        screen_refresh_rate: Monitor refresh rate (Hz).
    """

    def __init__(
        self,
        frame_rates: list[int],
        positions: list[tuple[float, float]],
        labels: list[str],
        signal_len: float,
        eeg_s_rate: int,
        overlap: float = 0.25,
        screen_refresh_rate: int = 60,
    ) -> None:
        self._fr_rates = frame_rates
        self._freqs = [screen_refresh_rate / fr for fr in frame_rates]
        logger.info("Target frequencies: {}", self._freqs)
        self._positions = positions
        self._labels = labels
        self._chunk_len = signal_len
        self._s_rate = eeg_s_rate
        self._overlap = overlap
        self._lock = Lock()
        self._data_buf: np.ndarray = np.array([])
        self._predicted_idx: int | None = None
        self.cca = CCAAnalysis(
            freqs=self._freqs,
            win_len=signal_len,
            s_rate=eeg_s_rate,
            n_harmonics=2,
        )
        self.win: visual.Window | None = None
        self.targets: list[CheckerBoard] = []
        self._pred_texts: list[visual.TextStim] = []

    def push_eeg(self, samples: np.ndarray) -> None:
        """Thread-safe: append EEG samples to the buffer.

        Args:
            samples: Array of shape (n_samples, n_channels).
        """
        with self._lock:
            if self._data_buf.size == 0:
                self._data_buf = samples
            else:
                self._data_buf = np.concatenate((self._data_buf, samples), axis=0)

    def run(self, duration: float) -> None:
        """Run the real-time experiment for `duration` seconds."""
        self._init_display()
        start = time.time()
        while time.time() - start < duration:
            self.win.flip()
            for stim in self.targets:
                stim.draw()
            if self._predicted_idx is not None:
                self._pred_texts[self._predicted_idx].draw()
            self._analyze()
        self.win.close()

    def show_statistics(self) -> None:
        """Print flicker timing statistics."""
        for stim in self.targets:
            avg, std = stim.get_statistics()
            logger.info("frequency: {:.2f} Hz, std: {:.6f}", 1 / avg, std)

    def _init_display(self) -> None:
        self._data_buf = np.array([])
        self.win = visual.Window(
            [800, 600],
            monitor="testMonitor",
            fullscr=True,
            screen=1,
            units="norm",
            color=[0.1, 0.1, 0.1],
        )
        self.win.recordFrameIntervals = True
        aspect = self.win.size[1] / self.win.size[0]
        stim_size = (0.6 * aspect, 0.6)
        for fr, pos, label in zip(self._fr_rates, self._positions, self._labels):
            self.targets.append(
                CheckerBoard(
                    window=self.win,
                    size=stim_size,
                    n_frame=fr,
                    position=pos,
                    log_time=True,
                )
            )
            self._pred_texts.append(
                visual.TextStim(
                    win=self.win,
                    pos=[0, 0],
                    text=label,
                    color=(1, 1, 1),
                    height=0.3,
                    colorSpace="rgb",
                    bold=True,
                )
            )

    def _analyze(self) -> None:
        """Classify when enough data is buffered."""
        with self._lock:
            if self._data_buf.size == 0:
                return
            n_needed = int(self._chunk_len * self._s_rate)
            if self._data_buf.shape[0] < n_needed:
                return
            chunk = self._data_buf[:n_needed].copy()
            keep = int(self._overlap * self._s_rate)
            self._data_buf = self._data_buf[keep:]

        scores = self.cca.apply_cca(chunk)
        logger.debug("CCA scores: {}", scores)
        if all(s < SCORE_THRESHOLD for s in scores):
            self._predicted_idx = None
        else:
            self._predicted_idx = int(np.argmax(scores))
            logger.info("  -> Target {}", self._predicted_idx)
