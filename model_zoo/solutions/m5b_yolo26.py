"""m5b: family "yolo26_e2e" - our own YOLO26 model (ns_yolo26n_v2).

YOLO26 is the newest Ultralytics generation. Like m5's models it's END-TO-END: it removes its own
duplicates and answers with a short list of finished plates, so there's no NMS to write
(BASICS.md, section 7). Its output even has the same width as m5's, 18 numbers per row. But the 18
numbers MEAN something else, which is the whole point of this exercise:

    m5 (Actor&Thinker), 30 rows:   box(4) | score | class | 4 COLOR scores    | 4 corners x (x, y)
    m5b (YOLO26),      300 rows:   box(4) | score | class | 4 corners x (x, y, visible)

Same shape, different contract: if you decoded this with m5's decoder, nothing would crash, and every
answer would be wrong. Always find out what the numbers mean, never just their shape.

  - input: 640x640, centered on gray 114, RGB, 0..1 (standard Ultralytics)
  - corners have a 3rd value: how sure the model is that the corner is visible (0..1). We don't need it
    to find the plate, but it could tell the aiming code to distrust a half-hidden plate.
  - 8 classes, color and number in one name: red_sentry, red_1, red_2, red_3, blue_sentry, blue_1, ...
    It only learned sentry, 1, 2 and 3: on a 4, 5, outpost or base it will guess something else.
Trained by us: synthetic armor pictures first, then fine-tuned on Actor&Thinker's labeled dataset.

Run:  python3 m5b_yolo26.py
"""
import _path  # noqa: F401  (solutions-only)
import cv2
import numpy as np

import checks
from common import family_demo, input_hw, letterbox, make_detection

SCORE_THRESHOLD = 0.5
LABELS = ["red_sentry", "red_1", "red_2", "red_3", "blue_sentry", "blue_1", "blue_2", "blue_3"]  # from the checkpoint
CORNER_ORDER = [0, 3, 2, 1]  # it learned from Actor&Thinker's labels: TL, BL, BR, TR. We want TL, TR, BR, BL


def preprocess(frame, inp):
    """Returns (tensor, Transform), the standard Ultralytics way."""
    # Centered letterbox on gray 114, BGR -> RGB, divide by 255, NCHW, float32 (like m4 and m5).
    # ANSWER-START
    canvas, tf = letterbox(frame, input_hw(inp), align="center", pad=114)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), tf
    # ANSWER-END


def label_info(class_id):
    """Class id (0..7) -> (color, number), e.g. 5 -> ("blue", "1")."""
    # LABELS[class_id] is a name like "blue_1": split it at the "_".
    # ANSWER-START
    color, number = LABELS[class_id].split("_")
    return color, number
    # ANSWER-END


def decode(outputs, tf, score_thr=SCORE_THRESHOLD, iou_thr=None):
    """Returns a list of detection dicts, best first. iou_thr is ignored: the model did NMS."""
    # For each of the 300 rows with score (column 4) >= score_thr:
    #   class id = int(round(row[5]))  (a whole number stored as a decimal), then label_info()
    #   corners  = row[6:18] -> reshape to (4, 3): x, y, visible. Keep x and y (the first 2 columns),
    #              then CORNER_ORDER -> tf.to_frame()
    # Return them sorted by score, highest first.
    # ANSWER-START
    rows = np.squeeze(outputs[0], 0).astype(np.float32)
    dets = []
    for row in rows[rows[:, 4] >= score_thr]:
        class_id = int(round(float(row[5])))
        color, number = label_info(class_id)
        corners = tf.to_frame(row[6:18].reshape(4, 3)[:, :2][CORNER_ORDER])
        dets.append(make_detection(corners, row[4], color, number, None, LABELS[class_id]))
    return sorted(dets, key=lambda d: -d["score"])
    # ANSWER-END


def investigate(outputs, tf):
    """Part A: print the first rows, return the first row's corners as 4 points."""
    rows = np.squeeze(outputs[0], 0)
    print("    row | x1     y1     x2     y2   | score class | corners: x y visible, x y visible, ...")
    for i, r in enumerate(rows[:5]):
        print(f"    {i:3d} | " + " ".join(f"{v:6.1f}" for v in r[:4]) + f" | {r[4]:.2f}  {r[5]:4.1f} | "
              + ", ".join(" ".join(f"{v:.0f}" if k < 2 else f"{v:.2f}" for k, v in enumerate(r[6 + 3 * c:9 + 3 * c]))
                          for c in range(4)))
    print(f"    ... {len(rows)} rows in total, {int((rows[:, 4] > 0.01).sum())} with a score above 0.01")
    return tf.to_frame(rows[0, 6:18].reshape(4, 3)[:, :2])


if __name__ == "__main__":
    family_demo(__import__(__name__), "yolo26_e2e", checks)
