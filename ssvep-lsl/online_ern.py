"""Passive ERN (Error-Related Negativity) monitor with LSL.

Receives EEG from an LSL stream, applies bandpass + notch filtering,
runs threshold-based ERN detection on frontal channels, and displays
a live PsychoPy window with real-time EEG traces and hit indicators.

Usage:
    uv run python online_ern.py -s "lock_in_eeg_processed"
    uv run python online_ern.py -s None   # display-only, no EEG
"""

import argparse
import threading
import time
from collections import deque

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from ern_detector import ERNDetector
from filters import apply_filter, build_filter
from psychopy import core, visual
from pylsl import StreamInlet, resolve_byprop

TRACE_SECONDS = 3
N_TRACE_CHANNELS = 3  # Frontal channels to display


def _lsl_reader(
    inlet: StreamInlet,
    push_fn: callable,
    stop: threading.Event,
    channels: list[int] | None = None,
) -> None:
    """Background thread: pull LSL chunks and push to monitor."""
    while not stop.is_set():
        chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=32)
        if chunk:
            samples = np.array(chunk, dtype=np.float32)
            if channels is not None:
                samples = samples[:, channels]
            push_fn(samples)


class ERNMonitor:
    """Live ERN monitor with PsychoPy display.

    Shows real-time EEG traces for frontal channels and flashes
    an indicator when an ERN is detected.

    Args:
        s_rate: EEG sampling rate in Hz.
        frontal_channels: Channel indices to monitor.
        threshold_uv: ERN detection threshold in uV.
    """

    def __init__(
        self,
        s_rate: int,
        frontal_channels: list[int] | None = None,
        threshold_uv: float = -5.0,
    ) -> None:
        self._s_rate = s_rate
        self._channels = frontal_channels or [0, 1, 2]
        self._lock = threading.Lock()
        self._raw_buf: np.ndarray = np.empty((0, 0))
        self._use_filter = True

        self._detector = ERNDetector(
            s_rate=s_rate,
            frontal_channels=self._channels,
            threshold_uv=threshold_uv,
        )

        # Display state
        self._trace_buf: deque[np.ndarray] = deque(
            maxlen=TRACE_SECONDS * s_rate,
        )
        self._flash_until = 0.0
        self._hit_count = 0

    def enable_filter(self, n_ch: int) -> None:
        """Initialise the IIR filter chain."""
        self._sos, self._zi = build_filter(float(self._s_rate), n_ch)
        self._use_filter = True

    def disable_filter(self) -> None:
        self._use_filter = False

    def push_eeg(self, samples: np.ndarray) -> None:
        """Thread-safe: accept new EEG samples from LSL reader."""
        with self._lock:
            if self._raw_buf.size == 0:
                self._raw_buf = samples
            else:
                self._raw_buf = np.concatenate(
                    (self._raw_buf, samples), axis=0
                )

    def run(self, duration: float) -> None:
        """Run the live monitor for `duration` seconds."""
        win = visual.Window(
            [800, 600],
            monitor="testMonitor",
            fullscr=True,
            screen=1,
            units="norm",
            color=[0.1, 0.1, 0.1],
        )

        title_text = visual.TextStim(
            win, text="ERN Monitor",
            pos=(0, 0.9), height=0.06,
            color="white",
        )
        counter_text = visual.TextStim(
            win, text="Hits: 0",
            pos=(0.7, 0.9), height=0.06,
            color="white",
        )
        flash_circle = visual.Circle(
            win, radius=0.08, pos=(0, 0),
            fillColor="red", lineColor="red",
            opacity=0.0,
        )
        status_text = visual.TextStim(
            win, text="Waiting for data...",
            pos=(0, -0.05), height=0.05,
            color="gray",
        )

        ch_labels = [f"ch{c}" for c in self._channels]
        trace_colors = ["#4FC3F7", "#81C784", "#FFB74D"]
        trace_lines = []
        y_positions = [0.4, 0.0, -0.4]
        for i, (label, color, y_pos) in enumerate(
            zip(ch_labels, trace_colors, y_positions)
        ):
            lbl = visual.TextStim(
                win, text=label,
                pos=(-0.9, y_pos), height=0.04,
                color=color, anchorHoriz="left",
            )
            trace_lines.append({
                "label": lbl,
                "color": color,
                "y_center": y_pos,
                "shape": None,
            })

        clock = core.Clock()
        start = time.time()

        while time.time() - start < duration:
            self._process_chunk()

            # Draw traces
            trace_data = list(self._trace_buf)
            if trace_data:
                arr = np.array(trace_data)
                n_pts = arr.shape[0]
                x_coords = np.linspace(-0.85, 0.85, n_pts)

                for i, ch in enumerate(self._channels):
                    if ch < arr.shape[1]:
                        y_data = arr[:, ch]
                        y_range = max(
                            np.ptp(y_data), 1e-6
                        )
                        y_norm = (
                            (y_data - y_data.mean())
                            / y_range * 0.25
                        )
                        y_screen = y_norm + y_positions[i]

                        vertices = np.column_stack(
                            (x_coords, y_screen)
                        )
                        trace_lines[i]["shape"] = visual.ShapeStim(
                            win,
                            vertices=vertices.tolist(),
                            lineColor=trace_colors[i],
                            lineWidth=1.5,
                            closeShape=False,
                        )

            for tl in trace_lines:
                tl["label"].draw()
                if tl["shape"] is not None:
                    tl["shape"].draw()

            # Flash indicator
            now = time.time()
            if now < self._flash_until:
                flash_circle.opacity = 0.8
            else:
                flash_circle.opacity = 0.0
            flash_circle.draw()

            counter_text.text = f"Hits: {self._hit_count}"
            if not self._trace_buf:
                status_text.draw()

            title_text.draw()
            counter_text.draw()
            win.flip()

            if "escape" in psychopy_get_keys():
                break

        win.close()

    def _process_chunk(self) -> None:
        """Pull buffered data, filter, detect, update display."""
        with self._lock:
            if self._raw_buf.size == 0:
                return
            chunk = self._raw_buf.copy()
            self._raw_buf = np.empty((0, chunk.shape[1]))

        if self._use_filter:
            chunk, self._zi = apply_filter(chunk, self._sos, self._zi)

        for row in chunk:
            self._trace_buf.append(row)

        hits = self._detector.detect(chunk, time.time())
        for hit in hits:
            self._hit_count += 1
            self._flash_until = time.time() + 0.3
            print(
                f"[ERN] Hit #{self._hit_count} "
                f"t={hit.timestamp:.2f} "
                f"ch={hit.channel} "
                f"amp={hit.amplitude:.1f} uV"
            )


def psychopy_get_keys() -> list[str]:
    """Wrap psychopy event.getKeys for testability."""
    from psychopy import event

    return event.getKeys()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Passive ERN monitor via LSL"
    )
    parser.add_argument(
        "-s", "--stream",
        default="lock_in_eeg_processed",
        help="LSL stream name, or 'None' for display-only mode",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int, default=120,
        help="Duration in seconds (default: 120)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float, default=-5.0,
        help="ERN detection threshold in uV (default: -5.0)",
    )
    parser.add_argument(
        "-c", "--channels",
        nargs="*", type=int,
        default=list(range(8)),
        help="Channel indices to use (default: 0-7)",
    )
    args = parser.parse_args()

    use_lsl = args.stream.lower() != "none"

    if use_lsl:
        print(f"Resolving LSL stream '{args.stream}'...")
        streams = resolve_byprop(
            "name", args.stream, minimum=1, timeout=10
        )
        if not streams:
            raise RuntimeError(
                f"No LSL stream '{args.stream}' found. "
                "Is the EEG source running?"
            )
        inlet = StreamInlet(streams[0])
        info = inlet.info()
        s_rate = int(info.nominal_srate())
        total_ch = info.channel_count()
        channels = args.channels
        bad = [c for c in channels if c >= total_ch]
        if bad:
            raise RuntimeError(
                f"Channels {bad} out of range "
                f"(stream has {total_ch})"
            )
        n_ch = len(channels)
        print(
            f"Connected: {info.name()}, "
            f"{total_ch} ch @ {s_rate} Hz "
            f"(using {n_ch}: {channels})"
        )
    else:
        s_rate = 250
        channels = args.channels
        n_ch = len(channels)
        print("Display-only mode (no LSL stream)")

    monitor = ERNMonitor(
        s_rate=s_rate,
        threshold_uv=args.threshold,
    )

    if use_lsl:
        monitor.enable_filter(n_ch)
        print(
            "Filter: bandpass 1-45 Hz, notch 50+60 Hz"
        )

    stop = threading.Event()
    if use_lsl:
        reader = threading.Thread(
            target=_lsl_reader,
            args=(
                inlet, monitor.push_eeg,
                stop, channels,
            ),
            daemon=True,
        )
        reader.start()

    try:
        monitor.run(args.duration)
    finally:
        stop.set()
        if use_lsl:
            reader.join(timeout=2)
            inlet.close_stream()
        print(
            f"\nSession complete. "
            f"Total ERN hits: {monitor._hit_count}"
        )


if __name__ == "__main__":
    main()
