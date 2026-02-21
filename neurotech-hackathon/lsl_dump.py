"""Dump raw LSL stream data to CSV."""

import csv
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from pylsl import resolve_byprop, StreamInlet

STREAM_NAME = sys.argv[1] if len(sys.argv) > 1 else "lock_in_eeg_processed"
OUTPUT = f"lsl_dump_{int(time.time())}.csv"


def main() -> None:
    print(f"Resolving LSL stream '{STREAM_NAME}'...")
    streams = resolve_byprop("name", STREAM_NAME, minimum=1, timeout=10)
    if not streams:
        print(f"No stream '{STREAM_NAME}' found.")
        sys.exit(1)

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    n_ch = info.channel_count()
    sr = info.nominal_srate()
    print(
        f"Connected: {info.name()}, "
        f"{n_ch} channels @ {sr} Hz"
    )
    print(f"Writing to {OUTPUT}  (Ctrl+C to stop)")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp"] + [f"ch{i}" for i in range(n_ch)]
        )
        count = 0
        try:
            while True:
                sample, ts = inlet.pull_sample(timeout=1.0)
                if sample is None:
                    continue
                writer.writerow([ts] + sample)
                count += 1
                if count % 250 == 0:
                    f.flush()
                    print(f"  {count} samples...")
        except KeyboardInterrupt:
            print(f"\nDone. {count} samples written to {OUTPUT}")


if __name__ == "__main__":
    main()
