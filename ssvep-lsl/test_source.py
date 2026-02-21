"""Synthetic EEG source over LSL for testing.

Generates a multi-channel signal with a configurable dominant
frequency (default 10 Hz) plus noise, streamed at 250 Hz.

Usage:
    uv run python test_source.py              # 10 Hz sine
    uv run python test_source.py --freq 15    # 15 Hz sine
"""

import argparse
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from pylsl import StreamInfo, StreamOutlet

STREAM_NAME = "lock_in_eeg_processed"
SRATE = 250
N_CHANNELS = 8
CHUNK_SIZE = 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic EEG LSL source"
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="Dominant frequency in Hz (default: 10)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=50.0,
        help="Signal amplitude in uV (default: 50)",
    )
    args = parser.parse_args()

    info = StreamInfo(
        name=STREAM_NAME,
        type="EEG",
        channel_count=N_CHANNELS,
        nominal_srate=SRATE,
        source_id="test_ssvep_source",
    )
    outlet = StreamOutlet(info, chunk_size=CHUNK_SIZE)
    print(
        f"Streaming '{STREAM_NAME}': "
        f"{N_CHANNELS} ch @ {SRATE} Hz, "
        f"freq={args.freq} Hz, amp={args.amplitude} uV"
    )
    print("Ctrl+C to stop")

    dt = 1.0 / SRATE
    t = 0.0
    try:
        while True:
            chunk = []
            for _ in range(CHUNK_SIZE):
                signal = args.amplitude * np.sin(
                    2 * np.pi * args.freq * t
                )
                noise = 2.0 * np.random.randn(N_CHANNELS)
                sample = noise.copy()
                # Put stronger signal on occipital channels
                sample[5:8] += signal
                chunk.append(sample.tolist())
                t += dt
            outlet.push_chunk(chunk)
            time.sleep(CHUNK_SIZE * dt)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
