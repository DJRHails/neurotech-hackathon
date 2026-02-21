"""Windows EEG streamer: Unicorn Hybrid Black -> LSL.

Run this on the Windows machine connected to the Unicorn Hybrid
Black headset. It streams raw EEG data over the network via Lab
Streaming Layer (LSL). The macOS machine receives it using
`main.py --lsl`.

Usage:
    python windows_stream.py
"""

import gpype as gp


def main() -> None:
    app = gp.MainApp(
        caption="Unicorn LSL Streamer",
        grid_size=[1, 1],
    )

    pipeline = gp.Pipeline()

    source = gp.HybridBlack()
    lsl_out = gp.LSLSender(stream_name="unicorn_eeg")

    pipeline.connect(source, lsl_out)

    # Monitor scope so you can see the signal on Windows too
    scope = gp.TimeSeriesScope(time_window=5)
    pipeline.connect(source, scope)
    app.add_widget(scope)

    app.run(pipeline)


if __name__ == "__main__":
    main()
