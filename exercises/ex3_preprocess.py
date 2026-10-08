"""Exercise 3: turn a camera frame into the exact input the model expects.

Read BASICS.md, section 5 first: it explains every step (letterbox, color order, value range,
dimension order, number type).

The robot's own code is the only truth about how the model's input must look, because that's what
the model is used with. Read it in:
    reference/sp_vision_25/tasks/auto_aim/yolos/yolov5.cpp
    - YOLOV5::detect(): how the picture is resized and where it's placed (cv::Rect(0, 0, w, h))
    - the "ppp" block near the top: what happens to the colors and the values

Fill in the two functions, then run:  python3 ex3_preprocess.py
"""
import cv2  # noqa: F401
import numpy as np  # noqa: F401

import checks
import config
from backends import read_frame


def letterbox(img, size=config.INPUT_SIZE):
    """Resize img to fit inside a size x size square WITHOUT stretching it (the "letterbox").

    Return (canvas, scale):
      canvas -- a size x size x 3 picture of whole numbers (uint8) holding the resized img, placed
                where the robot's code places it, with the leftover area filled the way it fills it
      scale  -- how much the picture shrank: resized size / original size (e.g. 640 / 1440 = 0.444)
    """
    # TODO
    raise NotImplementedError


def to_tensor(canvas):
    """640x640x3 picture (uint8, BGR order) -> the model's input tensor.

    Return a float32 numpy array. Work out the shape it needs (exercise 1 printed it), the color
    order and the value range (the "ppp" block in the robot's code).
    """
    # TODO
    raise NotImplementedError


def preprocess(img):
    try:
        canvas, scale = letterbox(img)
        return to_tensor(canvas), scale
    except NotImplementedError:
        raise SystemExit("Finish exercise 3 (letterbox and to_tensor) first.")


if __name__ == "__main__":
    if not checks.check_ex3(letterbox, to_tensor):
        raise SystemExit
    frame = read_frame()
    canvas, scale = letterbox(frame)
    x = to_tensor(canvas)
    print(f"frame {frame.shape} -> canvas {canvas.shape}, scale {scale:.4f} -> tensor {x.shape} {x.dtype}")
    cv2.imwrite("ex3_letterbox.jpg", canvas)
    print("Saved ex3_letterbox.jpg: this is exactly what the model sees")
