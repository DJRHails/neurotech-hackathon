"""Real-time SSVEP experiment with LSL data acquisition.

Receives EEG from any LSL stream and runs the CCA-based
SSVEP classifier in real-time with psychopy visual stimulation.

Usage:
    uv run python online_ssvep.py -s "lock_in_eeg_processed" -d 120
    uv run python online_ssvep.py -s None -d 30  # stimulus only, no EEG
"""

import argparse
import threading

import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from pylsl import StreamInlet, resolve_byprop  # noqa: E402

from ssvep import SSVEPRealTime  # noqa: E402

TARGET_POS = [(-0.6, -0.6), (-0.6, 0.6), (0.6, 0.6), (0.6, -0.6)]
TARGET_LABELS = ["\u2199", "\u2196", "\u2197", "\u2198"]
FRAME_RATES = [5, 6, 7, 8]


def _lsl_reader(
    inlet: StreamInlet,
    experiment: SSVEPRealTime,
    stop: threading.Event,
    channels: list[int] | None = None,
) -> None:
    """Background thread: pull LSL chunks into experiment."""
    while not stop.is_set():
        chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=32)
        if chunk:
            samples = np.array(chunk)
            if channels is not None:
                samples = samples[:, channels]
            experiment.push_eeg(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time SSVEP via LSL")
    parser.add_argument(
        "-s",
        "--stream",
        default="lock_in_eeg_processed",
        help="LSL stream name, or 'None' for stimulus-only mode",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=120,
        help="Duration in seconds (default: 120)",
    )
    parser.add_argument(
        "-c",
        "--channels",
        nargs="*",
        type=int,
        default=list(range(8)),
        help="Channel indices to use (default: 0-7)",
    )
    args = parser.parse_args()

    # At 60 Hz refresh: 12, 10, 8.57, 7.5 Hz
    use_lsl = args.stream.lower() != "none"

    if use_lsl:
        logger.info("Resolving LSL stream '{}'...", args.stream)
        streams = resolve_byprop("name", args.stream, minimum=1, timeout=10)
        if not streams:
            msg = f"No LSL stream '{args.stream}' found. Is the EEG source running?"
            raise RuntimeError(msg)
        inlet = StreamInlet(streams[0])
        info = inlet.info()
        s_rate = int(info.nominal_srate())
        total_ch = info.channel_count()
        channels = args.channels
        bad = [c for c in channels if c >= total_ch]
        if bad:
            msg = (
                f"Channels {bad} out of range "
                f"(stream has {total_ch})"
            )
            raise RuntimeError(msg)
        logger.info(
            "Connected: {}, {} ch @ {} Hz "
            "(using {}: {})",
            info.name(), total_ch, s_rate,
            len(channels), channels,
        )
    else:
        s_rate = 250
        channels = args.channels
        logger.info("Stimulus-only mode (no LSL stream)")

    experiment = SSVEPRealTime(
        frame_rates=FRAME_RATES,
        positions=TARGET_POS,
        labels=TARGET_LABELS,
        signal_len=3,
        eeg_s_rate=s_rate,
        overlap=0.2,
        screen_refresh_rate=60,
    )

    stop = threading.Event()
    if use_lsl:
        reader = threading.Thread(
            target=_lsl_reader,
            args=(inlet, experiment, stop, channels),
            daemon=True,
        )
        reader.start()

    try:
        experiment.run(args.duration)
        experiment.show_statistics()
    finally:
        stop.set()
        if use_lsl:
            reader.join(timeout=2)
            inlet.close_stream()


if __name__ == "__main__":
    main()
