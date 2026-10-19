"""Generate PLACEHOLDER digit images (5 per digit) so mnist.ipynb runs out of the box.

These are rendered from handwriting-style fonts with random rotation/shift/scale,
NOT real handwriting. Replace images/ with your own 28x28 grayscale n_m.png files.
Convention: white digit on black background (same as MNIST).
"""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONTS = ["C:/Windows/Fonts/Inkfree.ttf", "C:/Windows/Fonts/segoepr.ttf",
         "C:/Windows/Fonts/LHANDW.TTF", "C:/Windows/Fonts/mvboli.ttf",
         "C:/Windows/Fonts/comic.ttf"]
out = Path(__file__).resolve().parent.parent / "images"
out.mkdir(exist_ok=True)
random.seed(5831)

for d in range(10):
    for v in range(1, 6):
        font = ImageFont.truetype(FONTS[(v - 1) % len(FONTS)], random.randint(150, 190))
        big = Image.new("L", (280, 280), 0)
        ImageDraw.Draw(big).text((140, 140), str(d), font=font, fill=255, anchor="mm")
        big = big.rotate(random.uniform(-12, 12), resample=Image.BICUBIC)
        bbox = big.getbbox()                       # crop to digit, fit in 20x20 like MNIST
        digit = big.crop(bbox)
        s = 20 / max(digit.size)
        digit = digit.resize((max(1, round(digit.width * s)), max(1, round(digit.height * s))), Image.LANCZOS)
        img = Image.new("L", (28, 28), 0)
        img.paste(digit, ((28 - digit.width) // 2 + random.randint(-1, 1),
                          (28 - digit.height) // 2 + random.randint(-1, 1)))
        img.save(out / f"{d}_{v}.png")
print("wrote", len(list(out.glob('*.png'))), "images to", out)
