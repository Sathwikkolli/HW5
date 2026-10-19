"""Builds mnist.ipynb and rock-paper-scissors.ipynb (run from the repo root)."""
import json
from pathlib import Path


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


mnist = nb([
("markdown", """
# Problem 2: MNIST class and testing with my own digits
Forward propagation of a 3-layer network (784-50-100-10, sigmoid, sigmoid, softmax) using the
pre-trained weights in `mnist/sample_weight.pkl`.
Test images live in `images/` and are named `n_m.png` (digit `n`, version `m`), 28x28 grayscale,
white digit on black background (MNIST convention).
"""),
("code", """
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
"""),
("code", '''
class Mnist:
    """Forward propagation (inference) for the pre-trained MNIST network."""

    def __init__(self, model_file="mnist/sample_weight.pkl"):
        with open(model_file, "rb") as f:
            self.network = pickle.load(f)

    # --- activation functions
    @staticmethod
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def softmax(a):
        c = np.max(a)                 # subtract max for numerical stability
        e = np.exp(a - c)
        return e / np.sum(e)

    # --- forward propagation
    def predict(self, x):
        w1, w2, w3 = self.network["W1"], self.network["W2"], self.network["W3"]
        b1, b2, b3 = self.network["b1"], self.network["b2"], self.network["b3"]
        z1 = self.sigmoid(np.dot(x, w1) + b1)
        z2 = self.sigmoid(np.dot(z1, w2) + b2)
        return self.softmax(np.dot(z2, w3) + b3)

    # --- image handling
    @staticmethod
    def load_image(path):
        """Open as 28x28 grayscale; return (2-D uint8 array for display, flat float vector in [0,1])."""
        img = Image.open(path).convert("L")
        if img.size != (28, 28):
            img = img.resize((28, 28))
        arr = np.array(img)
        return arr, arr.reshape(-1) / 255.0

    def classify(self, path, show=True):
        """Classify `path` (named n_m.png); show the input with imshow; print Success/Fail."""
        name = os.path.basename(path)
        expected = int(name.split("_")[0])
        arr, x = self.load_image(path)
        result = int(np.argmax(self.predict(x)))

        if show:
            plt.figure(figsize=(1.6, 1.6))
            plt.imshow(arr, cmap="gray")
            plt.title(name, fontsize=9)
            plt.axis("off")
            plt.show()

        if result == expected:
            print(f"Success: Image {name} is for digit {expected} is recognized as {result}.")
        else:
            print(f"Fail: Image {name} is for digit {expected} but the inference result is {result}.")
        return result == expected
'''),
("markdown", "## Sanity check on an MNIST test image"),
("code", """
import gzip
with gzip.open("mnist/t10k-images-idx3-ubyte.gz", "rb") as f:
    X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
with gzip.open("mnist/t10k-labels-idx1-ubyte.gz", "rb") as f:
    y_test = np.frombuffer(f.read(), np.uint8, offset=8)

mnist = Mnist()
n = 100
y = mnist.predict(X_test[n] / 255.0)
print("label:", y_test[n], "prediction:", np.argmax(y), "certainty: %.3f" % np.max(y))
"""),
("markdown", "## My digit images (5 per digit)"),
("code", """
results = []
for digit in range(10):
    for ver in range(1, 6):
        results.append(mnist.classify(f"images/{digit}_{ver}.png"))
print(f"\\nAccuracy on my images: {sum(results)}/{len(results)} = {np.mean(results):.0%}")
"""),
])

rps = nb([
("markdown", """
# Problem 1: Rock-Paper-Scissors with a Teachable Machine model
**Setup (one time)**
1. Train an *Image Project* with 3 classes (Rock, Paper, Scissors) at https://teachablemachine.withgoogle.com/
2. *Export Model -> Tensorflow -> Keras -> Download my model*. Unzip; put `keras_model.h5` and `labels.txt` in `model/`.
3. `pip install tensorflow pillow numpy matplotlib` (if loading fails on newer TensorFlow also `pip install tf_keras`; handled below).
4. Put a new image in `test_images/` (e.g. `test_images/rock.jpg`) and run the last cell.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

MODEL_FILE = "model/keras_model.h5"
LABELS_FILE = "model/labels.txt"
"""),
("code", """
def load_model(path):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:      # Keras 3 can't read some Teachable Machine layers
        print("tf.keras failed (", type(e).__name__, ") - retrying with tf_keras")
        import tf_keras
        return tf_keras.models.load_model(path, compile=False)

model = load_model(MODEL_FILE)

# labels.txt lines look like "0 Rock" -> keep the name only
with open(LABELS_FILE) as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f if line.strip()]
print(class_names)
model.summary()
"""),
("markdown", """
## Inference (forward propagation)
Teachable Machine preprocessing: resize + center-crop to 224x224, RGB, pixels scaled to [-1, 1].
`model.predict` runs the forward pass and returns softmax probabilities per class.
"""),
("code", """
def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    x = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
    return img, x[np.newaxis, ...]            # shape (1, 224, 224, 3)

def classify(path):
    img, x = preprocess(path)
    probs = model.predict(x, verbose=0)[0]
    k = int(np.argmax(probs))
    plt.imshow(img); plt.axis("off")
    plt.title(f"{class_names[k]} ({probs[k]:.1%})"); plt.show()
    for name, p in zip(class_names, probs):
        print(f"{name:10s} {p:.4f}")
    print("Prediction:", class_names[k])
    return class_names[k]

classify("test_images/rock.jpg")     # change to your new image
"""),
])

for name, book in (("mnist.ipynb", mnist), ("rock-paper-scissors.ipynb", rps)):
    Path(name).write_text(json.dumps(book, indent=1), encoding="utf-8")
print("built")
