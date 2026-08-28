"""Generate every icon in the project from the one logo in ``assets/spektr.png``.

Run it only when the logo changes:

    python packaging/make_icon.py

The results are committed, so neither the exe build nor the APK build needs
Pillow. What it writes:

* ``packaging/spektr.ico`` — the Windows executable and installer icon, drawn
  at 512 and downsampled into every size Windows asks for, because a 16x16
  drawn directly is unreadable and a 16x16 downsampled from 512 keeps the
  silhouette. The square is rounded off and the corners made transparent: the
  logo's own field is black, and a hard black square on a dark taskbar reads
  as a hole rather than an icon.
* ``android/.../mipmap-*/ic_launcher_foreground.png`` — the mark alone, on
  transparency, for the adaptive icon's foreground layer. The logo fades its
  bars out to black, so alpha is taken from luminance and the fade survives as
  a fade rather than becoming a black smear on a launcher's own wallpaper.
* ``android/.../mipmap-*/ic_launcher.png`` — the same thing flattened onto the
  background colour, as the legacy square. minSdk is 29 so nothing should ever
  reach for it, but a launcher that does gets an icon rather than a blank.

The adaptive icon XML, the monochrome layer and the notification icon are
hand-written vectors under ``res/`` and are not generated — they are the same
mark reduced to flat shapes, which is a drawing decision rather than a
resampling one.

Sizing follows the adaptive-icon geometry: the foreground canvas is 108 units,
of which only the central 66 is guaranteed to survive an arbitrary mask. The
mark is 1.40:1, so the widest box whose *diagonal* fits that circle is 54
units across — half the canvas. Anything wider clips its own extremes under a
circular mask, and the extremes here are the cyan bar and the end of the
underscore.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    sys.exit("this script needs Pillow:  pip install pillow")

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets" / "spektr.png"
ICO = Path(__file__).with_name("spektr.ico")
RES = ROOT / "android" / "app" / "src" / "main" / "res"

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]

#: The logo's own field, and the adaptive icon's background layer. Kept in step
#: with ``res/values/ic_launcher_background.xml`` by hand — one number.
BG = (0, 0, 0, 255)

#: Density buckets, as multiples of the 108-unit adaptive canvas at mdpi.
DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}

#: Adaptive-icon canvas, and the width the mark is drawn at inside it.
CANVAS_UNITS = 108
MARK_UNITS = 54


def load_mark() -> Image.Image:
    """The logo cropped to its ink, with the black field turned to alpha.

    Alpha comes from luminance rather than from a colour key. Every bar in the
    mark is a gradient that runs out into the background, so keying black would
    cut each one off at whatever threshold was picked and leave a hard edge
    where the design has a fade.
    """
    src = Image.open(SOURCE).convert("RGB")
    lum = src.convert("L")
    mark = src.copy()
    mark.putalpha(lum)
    box = lum.point(lambda v: 255 if v > 8 else 0).getbbox()
    if box is None:
        sys.exit(f"{SOURCE} looks empty — nothing above the noise floor to crop")
    return mark.crop(box)


def on_canvas(mark: Image.Image, canvas: int, mark_w: int) -> Image.Image:
    """The mark centred on a transparent square of ``canvas`` pixels."""
    w, h = mark.size
    scaled = mark.resize((mark_w, max(1, round(h * mark_w / w))), Image.LANCZOS)
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.alpha_composite(
        scaled, ((canvas - scaled.width) // 2, (canvas - scaled.height) // 2)
    )
    return out


def write_ico(mark: Image.Image) -> None:
    size = 512
    # A little tighter than the Android safe box: a Windows icon is never
    # masked, so the only constraint is that it not touch its own edges.
    tile = on_canvas(mark, size, round(size * 0.78))
    plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=size // 8, fill=BG
    )
    plate.alpha_composite(tile)
    plate.save(ICO, sizes=[(s, s) for s in ICO_SIZES])
    print(f"wrote {ICO.relative_to(ROOT)}  ({', '.join(f'{s}x{s}' for s in ICO_SIZES)})")


def write_android(mark: Image.Image) -> None:
    for bucket, factor in DENSITIES.items():
        canvas = round(CANVAS_UNITS * factor)
        fg = on_canvas(mark, canvas, round(MARK_UNITS * factor))
        out = RES / f"mipmap-{bucket}"
        out.mkdir(parents=True, exist_ok=True)
        fg.save(out / "ic_launcher_foreground.png")

        # The legacy square: the same mark, flattened, at the launcher's own
        # icon size rather than the adaptive canvas — 48 units, not 108.
        legacy_px = round(48 * factor)
        legacy = Image.new("RGBA", (legacy_px, legacy_px), BG)
        legacy.alpha_composite(on_canvas(mark, legacy_px, round(legacy_px * 0.82)))
        legacy.save(out / "ic_launcher.png")
        print(f"wrote {out.relative_to(ROOT)}/  ({canvas}px adaptive, {legacy_px}px legacy)")


def main() -> None:
    mark = load_mark()
    print(f"read {SOURCE.relative_to(ROOT)} — mark is {mark.width}x{mark.height}")
    write_ico(mark)
    write_android(mark)


if __name__ == "__main__":
    main()
