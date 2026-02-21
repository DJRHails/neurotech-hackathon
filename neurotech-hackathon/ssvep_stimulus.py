"""SSVEP flickering stimulus widget.

Displays two images side-by-side, each flickering at a different
frequency using frame-counted pixmap swaps. Receives detection
feedback from SSVEPDetector via a thread-safe callback.
"""

import threading

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from gpype.frontend.widgets.base.widget import Widget

from images import make_placeholder

# Detection result constants
DETECT_NONE = 0
DETECT_LEFT = 1
DETECT_RIGHT = 2

# Flicker config: (frequency_hz, half_cycle_frames) at 60 FPS
# 10 Hz -> 3 frames on, 3 off (period = 6 frames = 100 ms)
# 15 Hz -> 2 frames on, 2 off (period = 4 frames = 66.7 ms)
STIMULI = [
    {"freq": 10, "half_cycle": 3},
    {"freq": 15, "half_cycle": 2},
]

_BORDER_NONE = "border: 3px solid transparent;"
_BORDER_DETECTED = "border: 3px solid #22C55E;"


class SSVEPStimulus(Widget):
    """Two-image SSVEP stimulus with frame-counted flicker."""

    def __init__(self) -> None:
        container = QWidget()
        Widget.__init__(
            self,
            widget=container,
            name="SSVEP Stimulus",
            layout=QVBoxLayout,
        )

        self._frame = 0
        self._lock = threading.Lock()
        self._detection = DETECT_NONE

        # Build image pixmaps and black masks
        self._pixmaps: list[QPixmap] = []
        self._blacks: list[QPixmap] = []
        self._labels: list[QLabel] = []
        self._frames: list[QLabel] = []

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for i in range(2):
            pixmap = make_placeholder(i)
            self._pixmaps.append(pixmap)

            black = QPixmap(pixmap.size())
            black.fill(Qt.GlobalColor.black)
            self._blacks.append(black)

            frame_label = QLabel()
            frame_label.setStyleSheet(_BORDER_NONE)
            frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._frames.append(frame_label)

            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._labels.append(img_label)

            inner = QVBoxLayout(frame_label)
            inner.setContentsMargins(4, 4, 4, 4)
            inner.addWidget(img_label)

            freq_label = QLabel(f"{STIMULI[i]['freq']} Hz")
            freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(freq_label)

            row_layout.addWidget(frame_label)

        self._status = QLabel("Detection: ---")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._layout.addWidget(row)
        self._layout.addWidget(self._status)

    def set_detection(self, result: int) -> None:
        """Thread-safe setter called from SSVEPDetector."""
        with self._lock:
            self._detection = result

    def _update(self) -> None:
        self._frame += 1

        # Flicker each image via pixmap swap
        for i, stim in enumerate(STIMULI):
            hc = stim["half_cycle"]
            visible = (self._frame % (2 * hc)) < hc
            if visible:
                self._labels[i].setPixmap(self._pixmaps[i])
            else:
                self._labels[i].setPixmap(self._blacks[i])

        # Read detection and update borders
        with self._lock:
            det = self._detection

        for i in range(2):
            detected = (
                (det == DETECT_LEFT and i == 0)
                or (det == DETECT_RIGHT and i == 1)
            )
            style = _BORDER_DETECTED if detected else _BORDER_NONE
            self._frames[i].setStyleSheet(style)

        labels = {
            DETECT_NONE: "Detection: ---",
            DETECT_LEFT: "Detection: LEFT (A)",
            DETECT_RIGHT: "Detection: RIGHT (B)",
        }
        self._status.setText(labels.get(det, "Detection: ---"))
