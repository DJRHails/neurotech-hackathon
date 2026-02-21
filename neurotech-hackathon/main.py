"""SSVEP Brain-Computer Interface with Unicorn Hybrid Black.

Sources (pick one):
  - Default on macOS: simulated Generator (10 Hz sine)
  - Default on Windows: Unicorn Hybrid Black (direct USB)
  - --lsl flag: receive from LSL stream (cross-machine)

Usage:
    uv run python main.py          # local source
    uv run python main.py --lsl    # receive from Windows LSL
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env before any imports that need PYLSL_LIB
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from PySide6.QtWidgets import QApplication

import gpype as gp

from images import Image, Video, generate_images
from ssvep_detector import SSVEPDetector
from ssvep_stimulus import SSVEPStimulus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def _make_source(use_lsl: bool):
    """Create EEG source based on flags and platform."""
    if use_lsl:
        from lsl_source import LSLSource

        return LSLSource(stream_name="unicorn_eeg")
    if sys.platform == "win32":
        return gp.HybridBlack()
    return gp.Generator(
        sampling_rate=250.0,
        channel_count=8,
        frame_size=1,
        signal_frequency=10.0,
        signal_amplitude=50.0,
        noise_amplitude=2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSVEP BCI"
    )
    parser.add_argument(
        "--lsl",
        action="store_true",
        help="Receive EEG via LSL instead of local device",
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

    qapp = QApplication([])

    stim_app = gp.MainApp(
        caption="SSVEP Stimulus",
        grid_size=[1, 1],
        app=qapp,
        position=[100, 100, 700, 500],
    )
    scope_app = gp.MainApp(
        caption="SSVEP Scopes",
        grid_size=[2, 2],
        app=qapp,
        position=[820, 100, 900, 600],
    )

    pipeline = gp.Pipeline()

    # Source
    source = _make_source(use_lsl=args.lsl)

    # Filter chain: Bandpass -> Notch 50 Hz -> Notch 60 Hz
    bandpass = gp.Bandpass(f_lo=1.0, f_hi=45.0, order=2)
    notch50 = gp.Bandstop(f_lo=49.0, f_hi=51.0, order=2)
    notch60 = gp.Bandstop(f_lo=59.0, f_hi=61.0, order=2)

    pipeline.connect(source, bandpass)
    pipeline.connect(bandpass, notch50)
    pipeline.connect(notch50, notch60)

    # FFT: 250-sample window = 1 Hz resolution at 250 Hz SR
    fft = gp.FFT(
        window_size=250, overlap=0.5, window_function="hamming"
    )
    pipeline.connect(notch60, fft)

    # Debug scopes
    eeg_scope = gp.TimeSeriesScope(time_window=5)
    pipeline.connect(notch60, eeg_scope)

    spectrum_scope = gp.SpectrumScope()
    pipeline.connect(fft, spectrum_scope)

    # SSVEP stimulus + detector
    video_path = (
        Path(__file__).parent / "assets" / "subway_surfers.mp4"
    )
    generated = generate_images(
        args.prompt[0], args.prompt[1]
    )

    stimuli: list[Video | Image] = []
    if video_path.exists():
        log.info("Loading video stimulus: %s", video_path)
        stimuli.append(Video(video_path))
    elif generated:
        stimuli.append(generated[0])
    if generated:
        stimuli.append(generated[1])

    stimulus = SSVEPStimulus(
        stimuli=stimuli or None
    )
    detector = SSVEPDetector(
        on_detect=stimulus.set_detection
    )
    pipeline.connect(fft, detector)

    # SSVEP power scope: 2-channel [left, right] correlation
    ssvep_scope = gp.TimeSeriesScope(time_window=10)
    pipeline.connect(detector, ssvep_scope)

    # Stimulus window: full-size flickering images
    stim_app.add_widget(stimulus, grid_positions=[1])

    # Scope window: EEG top, spectrum + SSVEP power bottom
    scope_app.add_widget(
        eeg_scope, grid_positions=[1, 2]
    )
    scope_app.add_widget(spectrum_scope, grid_positions=[3])
    scope_app.add_widget(ssvep_scope, grid_positions=[4])

    pipeline.start()
    try:
        # Show + start stimulus window manually
        stim_app._window.show()
        for w in stim_app._widgets:
            w.run()
        # Scope window enters Qt event loop
        scope_app.run()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
