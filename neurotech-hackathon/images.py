"""Stimulus sources: static images and video frame providers."""

from __future__ import annotations

import functools
import hashlib
import inspect
import io
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import cv2
import diskcache
import fal_client
import httpx
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from PySide6.QtGui import QImage, QPixmap

T = TypeVar("T")
SIZE = 300
_FAL_MODEL = "xai/grok-imagine-image"
_CACHE_DIR = Path(__file__).parent / ".cache" / "images"
_cache = diskcache.Cache(str(_CACHE_DIR))

log = logging.getLogger(__name__)

Stimulus = "Video | Image"


def cached(
    cache: diskcache.Cache,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Disk-cache decorator keyed on function source + args."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        src_hash = hashlib.md5(
            inspect.getsource(func).encode()
        ).hexdigest()

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            arg_key = json.dumps(
                [args, kwargs], sort_keys=True
            )
            key = hashlib.md5(
                f"{func.__name__}{src_hash}{arg_key}".encode()
            ).hexdigest()
            result = cache.get(key)
            if result is not None:
                log.info("[CACHE HIT] %s", func.__name__)
                return result  # type: ignore[return-value]
            log.info("[CACHE MISS] %s", func.__name__)
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result  # type: ignore[return-value]

        return wrapper

    return decorator


# ── Helpers ──────────────────────────────────────────────

_COLORS = [
    ("#2563EB", "#DBEAFE", "A"),  # Blue
    ("#DC2626", "#FEE2E2", "B"),  # Red
]


def _pil_to_qpixmap(img: PILImage.Image) -> QPixmap:
    data = img.convert("RGBA").tobytes()
    qimg = QImage(
        data,
        img.width,
        img.height,
        QImage.Format.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimg.copy())


# ── Stimulus types ───────────────────────────────────────


class Image:
    """Static image stimulus (single QPixmap)."""

    def __init__(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap

    def advance(self) -> None:
        pass

    @property
    def current(self) -> QPixmap:
        return self._pixmap


class Video:
    """Video stimulus — decodes frames on-the-fly.

    Loops back to the start when the video ends. Decodes one
    frame per ``advance()`` call so memory stays constant.
    """

    def __init__(
        self, path: str | Path, size: int = SIZE
    ) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video: {path}"
            )
        self._size = size
        self._current: QPixmap = QPixmap()
        self.advance()

    def advance(self) -> None:
        """Decode the next frame, looping at end."""
        ret, bgr = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bgr = self._cap.read()
            if not ret:
                return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (self._size, self._size)
        )
        h, w, ch = rgb.shape
        data = rgb.tobytes()
        qimg = QImage(
            data, w, h, ch * w,
            QImage.Format.Format_RGB888,
        )
        self._current = QPixmap.fromImage(qimg.copy())

    @property
    def current(self) -> QPixmap:
        """Most recently decoded frame."""
        return self._current

    def close(self) -> None:
        """Release the video capture."""
        self._cap.release()


# ── Builders ─────────────────────────────────────────────


def make_placeholder(index: int) -> Image:
    """Create a colored square with a centered label.

    Args:
        index: 0 for image A, 1 for image B.
    """
    accent, bg, label = _COLORS[index % len(_COLORS)]
    img = PILImage.new("RGB", (SIZE, SIZE), bg)
    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [4, 4, SIZE - 5, SIZE - 5], outline=accent, width=4
    )

    try:
        font = ImageFont.truetype("Arial", 80)
    except OSError:
        font = ImageFont.load_default(size=80)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((SIZE - tw) / 2, (SIZE - th) / 2 - bbox[1]),
        label,
        fill=accent,
        font=font,
    )

    return Image(_pil_to_qpixmap(img))


@cached(_cache)
def _fetch_image_bytes(prompt: str) -> bytes:
    """Call FAL AI and return raw image bytes.

    Args:
        prompt: Text description of the image to generate.

    Returns:
        Raw JPEG/PNG bytes from the API.
    """
    result = fal_client.subscribe(
        _FAL_MODEL,
        arguments={"prompt": prompt},
    )
    url = result["images"][0]["url"]
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def _generate_one(prompt: str) -> Image:
    """Generate a single image via FAL AI.

    Args:
        prompt: Text description of the image to generate.

    Returns:
        Image stimulus sized to SIZE x SIZE.
    """
    data = _fetch_image_bytes(prompt)
    img = PILImage.open(io.BytesIO(data))
    img = img.resize(
        (SIZE, SIZE), PILImage.Resampling.LANCZOS
    )
    return Image(_pil_to_qpixmap(img))


def generate_images(
    prompt_a: str, prompt_b: str
) -> tuple[Image, Image] | None:
    """Generate two stimulus images via FAL AI.

    Args:
        prompt_a: Text prompt for the left image.
        prompt_b: Text prompt for the right image.

    Returns:
        Tuple of two Image stimuli, or None on failure.
    """
    api_key = os.environ.get("FAL_KEY", "")
    if not api_key:
        log.info("FAL_KEY not set, using placeholders")
        return None

    try:
        log.info("Generating image A: %s", prompt_a)
        pix_a = _generate_one(prompt_a)
        log.info("Generating image B: %s", prompt_b)
        pix_b = _generate_one(prompt_b)
    except Exception:
        log.warning(
            "Image generation failed, using placeholders",
            exc_info=True,
        )
        return None

    return (pix_a, pix_b)
