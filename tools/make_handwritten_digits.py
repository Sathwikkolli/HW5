"""Generate 5 hand-drawn-looking versions of each digit as 28x28 grayscale PNGs (n_m.png).

Each digit is a pen-stroke skeleton, wobbled/slanted/scaled randomly per version, drawn with a
thick pen, blurred and normalised into a 20x20 box centred by centre of mass (MNIST style).
White digit on black background. Run from the repo root: python tools/make_handwritten_digits.py
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SKELETONS = {
    0: [[(0.5, 0.05), (0.2, 0.25), (0.15, 0.55), (0.3, 0.9), (0.6, 0.95), (0.82, 0.65), (0.82, 0.3), (0.6, 0.07), (0.45, 0.05)]],
    1: [[(0.3, 0.3), (0.55, 0.05), (0.55, 0.95)]],
    2: [[(0.2, 0.25), (0.4, 0.05), (0.75, 0.15), (0.75, 0.4), (0.2, 0.95), (0.85, 0.95)]],
    3: [[(0.2, 0.12), (0.7, 0.08), (0.75, 0.3), (0.4, 0.48), (0.78, 0.65), (0.7, 0.92), (0.2, 0.9)]],
    4: [[(0.6, 0.05), (0.15, 0.65), (0.9, 0.65)], [(0.65, 0.3), (0.65, 0.97)]],
    5: [[(0.8, 0.08), (0.25, 0.08), (0.2, 0.45), (0.6, 0.4), (0.8, 0.6), (0.7, 0.9), (0.2, 0.88)]],
    6: [[(0.7, 0.08), (0.3, 0.3), (0.2, 0.65), (0.4, 0.95), (0.75, 0.8), (0.65, 0.5), (0.25, 0.6)]],
    7: [[(0.15, 0.1), (0.85, 0.1), (0.4, 0.95)]],
    8: [[(0.5, 0.5), (0.2, 0.3), (0.5, 0.05), (0.8, 0.3), (0.5, 0.5), (0.15, 0.75), (0.5, 0.97), (0.85, 0.75), (0.5, 0.5)]],
    9: [[(0.75, 0.4), (0.5, 0.5), (0.25, 0.3), (0.5, 0.05), (0.78, 0.25), (0.7, 0.6), (0.5, 0.95)]],
}


def catmull_rom(pts, n=12):
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = (np.array(q) for q in p[i - 1:i + 3])
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(np.array(p[-2]))
    return np.array(out)


def draw_digit(d, rng):
    S = 224
    img = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(img)
    slant = rng.uniform(-0.25, 0.25)
    sx, sy = rng.uniform(0.8, 1.1), rng.uniform(0.9, 1.1)
    pen = rng.randint(14, 20)
    for stroke in SKELETONS[d]:
        pts = [(x + rng.gauss(0, 0.035), y + rng.gauss(0, 0.035)) for x, y in stroke]
        line = catmull_rom(pts)
        xy = [((0.5 + (x - 0.5) * sx + slant * (0.5 - y)) * S * 0.7 + S * 0.15,
               (0.5 + (y - 0.5) * sy) * S * 0.7 + S * 0.15) for x, y in line]
        dr.line(xy, fill=255, width=pen, joint="curve")
        for x, y in xy[::4]:
            dr.ellipse([x - pen / 2, y - pen / 2, x + pen / 2, y + pen / 2], fill=255)
    img = img.filter(ImageFilter.GaussianBlur(2))
    img = img.crop(img.getbbox())
    s = 20 / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    a = np.array(img, dtype=float)
    cy, cx = (np.indices(a.shape) * a).sum(axis=(1, 2)) / a.sum()      # centre of mass
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, (round(14 - cx), round(14 - cy)))
    return canvas


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "images"
    out.mkdir(exist_ok=True)
    rng = random.Random(5831)
    for d in range(10):
        for v in range(1, 6):
            draw_digit(d, rng).save(out / f"{d}_{v}.png")
    print("wrote", len(list(out.glob("*.png"))), "images to", out)
