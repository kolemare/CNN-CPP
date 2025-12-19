#!/usr/bin/env python3
import os
import glob
import json
import argparse

import cv2
import numpy as np
import onnxruntime as ort


def softmax(x):
    x = x.astype(np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_img(path, h, w):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Can't read: " + path)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None, ...].astype(np.float32)


def is_probs(y):
    if y.ndim != 1:
        return False
    s = float(y.sum())
    return abs(s - 1.0) < 1e-2 and y.min() >= -1e-3 and y.max() <= 1.0 + 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--w", type=int, default=32)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "model.onnx")
    map_path = os.path.join(base, "class_mapping.json")

    with open(map_path, "r", encoding="utf-8") as f:
        name_to_index = json.load(f)["name_to_index"]

    idx_to_name = {int(v): k for k, v in name_to_index.items()}

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(args.folder, ext))
    files.sort()

    for path in files:
        x = load_img(path, args.h, args.w)
        y = np.squeeze(sess.run(None, {in_name: x})[0]).astype(np.float32)

        p = y if is_probs(y) else softmax(y)
        pred = int(np.argmax(p))

        print(f"{os.path.basename(path)} -> {pred} ({idx_to_name.get(pred,'?')}) {float(p[pred]):.4f}")


if __name__ == "__main__":
    main()
