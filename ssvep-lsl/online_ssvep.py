"""Real-time SSVEP experiment with LSL data acquisition.

Receives EEG from any LSL stream and runs the CCA-based
SSVEP classifier in real-time with psychopy visual stimulation.

Usage:
    uv run python online_ssvep.py -s "lock_in_eeg_processed" -d 120
"""

import argparse
import threading

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from pylsl import StreamInlet, resolve_byprop
from ssvep import SSVEPRealTime


def _lsl_reader(
    inlet: StreamInlet,
    experiment: SSVEPRealTime,
    stop: threading.Event,
) -> None:
    """Background thread: pull LSL chunks into experiment."""
    while not stop.is_set():
        chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=32)
        if chunk:
            experiment.push_eeg(np.array(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time SSVEP via LSL")
    parser.add_argument(
        "-s",
        "--stream",
        default="lock_in_eeg_processed",
        help="LSL stream name (default: lock_in_eeg_processed)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=120,
        help="Duration in seconds (default: 120)",
    )
    args = parser.parse_args()

    print(f"Resolving LSL stream '{args.stream}'...")
    streams = resolve_byprop("name", args.stream, minimum=1, timeout=10)
    if not streams:
        raise RuntimeError(
            f"No LSL stream '{args.stream}' found. Is the EEG source running?"
        )

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    s_rate = int(info.nominal_srate())
    print(f"Connected: {info.name()}, {info.channel_count()} ch @ {s_rate} Hz")

    # 4 targets at screen corners
    # At 60 Hz refresh: 12, 10, 8.57, 7.5 Hz
    target_pos = [
        (-0.6, -0.6),
        (-0.6, 0.6),
        (0.6, 0.6),
        (0.6, -0.6),
    ]
    target_labels = ["\u2199", "\u2196", "\u2197", "\u2198"]
    fr_rates = [5, 6, 7, 8]

    experiment = SSVEPRealTime(
        frame_rates=fr_rates,
        positions=target_pos,
        labels=target_labels,
        signal_len=3,
        eeg_s_rate=s_rate,
        overlap=0.2,
        screen_refresh_rate=60,
    )

    stop = threading.Event()
    reader = threading.Thread(
        target=_lsl_reader,
        args=(inlet, experiment, stop),
        daemon=True,
    )
    reader.start()

    try:
        experiment.run(args.duration)
        experiment.show_statistics()
    finally:
        stop.set()
        reader.join(timeout=2)
        inlet.close_stream()


if __name__ == "__main__":
    main()
