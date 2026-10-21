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
3. `pip install tensorflow h5py pillow numpy matplotlib` (the loader below works around a Keras 3 incompatibility with Teachable Machine files).
4. Put a new image in `test_images/` (e.g. `test_images/test_1.png`) and run the last cell.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

MODEL_FILE = "model/keras_model.h5"
LABELS_FILE = "model/labels.txt"
"""),
("code", """
import json
import h5py
import tensorflow as tf

class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *a, **k):
        k.pop("groups", None); super().__init__(*a, **k)

def _key(layer, v):
    n = v.path.split("/")[-1]
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D) and n == "kernel":
        n = "depthwise_kernel"
    return n + ":0"

def _first(t):
    return t[0] if isinstance(t, (list, tuple)) else t

def load_tm_model(path):
    # Rebuilds the Teachable Machine model (MobileNetV2 backbone + 2 dense layers) and copies the weights
    # from keras_model.h5. Keras 3 cannot open these .h5 files directly.
    with h5py.File(path, "r") as f:
        cfg = json.loads(f.attrs["model_config"])
        seqs = {l["config"]["name"]: l["config"]["layers"] for l in cfg["config"]["layers"] if l["class_name"] == "Sequential"}
        backbone = tf.keras.Model.from_config(
            [l for l in seqs["sequential_1"] if l["class_name"] in ("Functional", "Model")][0]["config"],
            custom_objects={"DepthwiseConv2D": DepthwiseConv2D})
        w = f["model_weights"]
        for layer in backbone.layers:
            if layer.weights:
                g = w["sequential_1"][layer.name]
                layer.set_weights([np.asarray(g[_key(layer, v)]) for v in layer.weights])
        x = tf.keras.layers.GlobalAveragePooling2D()(_first(backbone.output))
        for l in seqs["sequential_3"]:
            if l["class_name"] != "Dense":
                continue
            c = l["config"]
            dense = tf.keras.layers.Dense(c["units"], activation=c["activation"], use_bias=c["use_bias"], name=c["name"])
            x = dense(x)
            g = w["sequential_3"][c["name"]]
            dense.set_weights([np.asarray(g[v.path.split("/")[-1] + ":0"]) for v in dense.weights])
    return tf.keras.Model(_first(backbone.input), x)

model = load_tm_model(MODEL_FILE)

# labels.txt lines look like "0 Rock" -> keep the name only
with open(LABELS_FILE) as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f if line.strip()]
print(class_names)
print("input:", model.input_shape, "output:", model.output_shape)
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

classify("test_images/test_1.png")     # change to your new image
"""),
("markdown", "## All held-out test images"),
("code", """
import glob
for f in sorted(glob.glob("test_images/*.png") + glob.glob("test_images/*.jpg")):
    print(f)
    classify(f)
"""),
])

for name, book in (("mnist.ipynb", mnist), ("rock-paper-scissors.ipynb", rps)):
    Path(name).write_text(json.dumps(book, indent=1), encoding="utf-8")
print("built")
