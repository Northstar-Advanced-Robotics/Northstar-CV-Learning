"""Exercise 5: decode ALL 25200 rows, keep the confident ones, remove the duplicates.

One plate usually shows up in several rows. NMS ("non-maximum suppression") keeps the best row
of each group and drops the rest (BASICS.md, section 7).

Fill in decode_all() (checked automatically) and draw_detections() (check it by looking
at ex5_result.jpg), then run:  python3 ex5_decode.py
"""
import cv2
import numpy as np

import checks
import config
from backends import get_model, read_frame
from ex3_preprocess import preprocess
from ex4_raw_output import sigmoid


def decode_all(raw, scale, score_thr=config.SCORE_THRESHOLD, iou_thr=config.NMS_IOU_THRESHOLD):
    """raw: the model's (25200, 22) output. Return a list of plates (dicts with the same keys as
    decode_row in exercise 4), highest score first, with duplicates removed.

    Steps:
      1. keep only rows whose score is at least score_thr (remember: column 8 is a logit)
      2. decode those rows
      3. remove duplicates with NMS: cv2.dnn.NMSBoxes(boxes, scores, score_thr, iou_thr) does it.
         It wants one box per plate as [x, y, width, height], but you have 4 corners: use the
         smallest and largest x and y of the corners. It returns the indices of the plates to keep.

    Try doing steps 1 and 2 on all rows at once with numpy (BASICS.md, section 3) instead of a
    Python loop over 25200 rows. Time both: the robot runs this on every frame.
    """
    # TODO
    raise NotImplementedError


COLOR_BGR = {"blue": (255, 128, 0), "red": (0, 0, 255), "gray": (160, 160, 160), "purple": (255, 0, 200)}


def draw_detections(img, dets):
    """Draw each plate on img: its 4-corner outline, a dot at its center, and a label like
    'blue 3 0.92'. Return img. (cv2.polylines, cv2.circle, cv2.putText)"""
    # TODO
    return img


if __name__ == "__main__":
    if not checks.check_ex5(decode_all):
        raise SystemExit
    model = get_model()
    frame = read_frame()
    x, scale = preprocess(frame)
    raw = model(x)["output/sink_port_0"][0]

    above = int((sigmoid(raw[:, 8]) >= config.SCORE_THRESHOLD).sum())
    dets = decode_all(raw, scale)
    print(f"{above} rows above the score threshold -> {len(dets)} plate(s) after NMS")
    for d in dets:
        print(f"  {d['color']} {d['number']}  score {d['score']:.2f}")
    cv2.imwrite("ex5_result.jpg", draw_detections(frame, dets))
    print("Saved ex5_result.jpg")
