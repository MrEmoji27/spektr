"""Generate ``spektr.ico`` — a spectrum in the classic theme's colours.

Run it only when you want to change the icon:

    python packaging/make_icon.py

The result is committed, so building the exe does not need Pillow. Drawn at 512
and downsampled into every icon size Windows asks for, because a 16x16 drawn
directly is unreadable and a 16x16 downsampled from 512 keeps the silhouette.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    sys.exit("this script needs Pillow:  pip install pillow")

OUT = Path(__file__).with_name("spektr.ico")
SIZES = [16, 24, 32, 48, 64, 128, 256]

BG = (13, 15, 20, 255)
#: the classic theme's three anchors — low, mid, high
LOW, MID, HIGH = (0x00, 0xFF, 0x41), (0xFF, 0xB0, 0x00), (0xFF, 0x33, 0x00)

#: bar heights as a fraction of the canvas, chosen to read as a spectrum
#: (bass-heavy, decaying, with one peak in the mids) rather than as a bar chart
HEIGHTS = [0.92, 0.66, 0.78, 0.44, 0.58, 0.32, 0.22]


def ramp(t: float) -> tuple[int, int, int]:
    """Blend low -> mid -> high in linear light, as the palette module does."""
    gamma = 2.2
    a, b, u = (LOW, MID, t * 2.0) if t < 0.5 else (MID, HIGH, (t - 0.5) * 2.0)
    u = u * u * (3.0 - 2.0 * u)                      # smoothstep, no crease
    out = []
    for ca, cb in zip(a, b):
        la, lb = (ca / 255) ** gamma, (cb / 255) ** gamma
        out.append(round((la + (lb - la) * u) ** (1 / gamma) * 255))
    return tuple(out)


def draw(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 8, fill=BG)

    pad = size * 0.14
    inner = size - pad * 2
    n = len(HEIGHTS)
    gap = inner / n * 0.28
    bar_w = (inner - gap * (n - 1)) / n
    floor = size - pad

    for i, h in enumerate(HEIGHTS):
        x0 = pad + i * (bar_w + gap)
        top = floor - inner * h
        d.rounded_rectangle(
            [x0, top, x0 + bar_w, floor],
            radius=bar_w * 0.22,
            fill=(*ramp(h), 255),
        )
        # peak marker, floating just above each bar
        cap = max(2.0, size * 0.016)
        y = top - cap * 2.4
        d.rounded_rectangle([x0, y, x0 + bar_w, y + cap], radius=cap / 2, fill=(235, 235, 235, 255))

    return img


def main() -> None:
    base = draw()
    base.save(OUT, sizes=[(s, s) for s in SIZES])
    print(f"wrote {OUT}  ({', '.join(f'{s}x{s}' for s in SIZES)})")


if __name__ == "__main__":
    main()
