"""Exercise 3: turn a camera frame into the tensor the model expects.

Run:  python3 ex3_preprocess.py
"""
import cv2
import numpy as np

import checks
import config
from backends import read_frame


def letterbox(img, size=config.INPUT_SIZE):
    """Resize img to fit inside a size x size square without stretching it.

    Matches the C++ (yolov5.cpp): the resized image goes in the TOP-LEFT corner and the
    rest is black. Returns (canvas, scale), where scale = resized size / original size.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
    return canvas, scale


def to_tensor(canvas):
    """640x640x3 uint8 BGR image -> (1, 3, 640, 640) float32 RGB tensor with values 0..1."""
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)  # height, width, channel -> channel, height, width
    return np.ascontiguousarray(chw[None].astype(np.float32) / 255.0)


def preprocess(img):
    canvas, scale = letterbox(img)
    return to_tensor(canvas), scale


if __name__ == "__main__":
    if not checks.check_ex3(letterbox, to_tensor):
        raise SystemExit
    frame = read_frame()
    canvas, scale = letterbox(frame)
    x = to_tensor(canvas)
    print(f"frame {frame.shape} -> canvas {canvas.shape}, scale {scale:.4f} -> tensor {x.shape} {x.dtype}")
    cv2.imwrite("ex3_letterbox.jpg", canvas)
    print("Saved ex3_letterbox.jpg: this is exactly what the model sees")
