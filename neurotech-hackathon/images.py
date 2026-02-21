"""Generate placeholder stimulus images as QPixmaps."""

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtGui import QPixmap, QImage

SIZE = 300
_COLORS = [
    ("#2563EB", "#DBEAFE", "A"),  # Blue
    ("#DC2626", "#FEE2E2", "B"),  # Red
]


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    data = img.convert("RGBA").tobytes()
    qimg = QImage(
        data, img.width, img.height, QImage.Format.Format_RGBA8888
    )
    return QPixmap.fromImage(qimg.copy())


def make_placeholder(index: int) -> QPixmap:
    """Create a colored square with a centered label.

    Args:
        index: 0 for image A, 1 for image B.
    """
    accent, bg, label = _COLORS[index % len(_COLORS)]
    img = Image.new("RGB", (SIZE, SIZE), bg)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle(
        [4, 4, SIZE - 5, SIZE - 5], outline=accent, width=4
    )

    # Centered label
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

    return _pil_to_qpixmap(img)
