"""m4b: family "szurp_v8" - RobotPilots' 2026 model (rp_v8n_0726), from their vision group SZURPVision.

It started as an Ultralytics YOLOv8n-pose like m4's models, but RobotPilots redesigned its last layers
(the "head"), so it answers differently. Spot the differences with m4 (words: BASICS.md, sections 5-8):
  - input: 480 high x 640 wide, the camera's 4:3 shape, so a 1440x1080 frame is just resized: no padding.
    (Other shapes get CENTERED with gray 124 padding.) RGB, values 0..1.
  - output: (1, 21, 6300): one COLUMN per guess, like m4. 6300 = 60x80 + 30x40 + 15x20 grid cells.
  - each guess's 21 numbers: 4 color scores, 9 class scores, then the 4 corners. NO box at all, and no
    separate "is this a plate?" score: the score is the best class score (already 0..1)
  - colors: blue, red, white (a switched-off plate, "gray" here), purple. This model never says purple,
    and never says class 5: the team removed them for 2026.
Source: https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models (README section 4.1, and
src/core/algorithm/NNdeployment/src/postprocess/network_postprocess.cpp, V8_21InfantryPostProcessor).

Run:  python3 m4b_szurp_v8.py
"""
import _path  # noqa: F401  (solutions-only)
import cv2
import numpy as np

import checks
from common import family_demo, input_hw, letterbox, make_detection, nms

SCORE_THRESHOLD = 0.5  # RobotPilots' value (config/detect.json)
IOU_THRESHOLD = 0.2    # RobotPilots' NMS value for armor (defaultNmsThreshold): stricter than the others' 0.3
PAD = 124              # their padding gray. Ultralytics uses 114: every team picks its own
COLORS = ["blue", "red", "gray", "purple"]
# Class 0 sentry, 1 hero, 2 engineer, 3-5 infantry, 6 outpost, 7 base (bottom plate), 8 base (top plate).
# Their code marks the hero (1) and the base's bottom plate (7) as big plates, everything else small.
CLASSES = [("sentry", "small"), ("1", "big"), ("2", "small"), ("3", "small"), ("4", "small"), ("5", "small"),
           ("outpost", "small"), ("base", "big"), ("base", "small")]
CORNER_ORDER = [0, 3, 2, 1]  # corners come out TL, BL, BR, TR (their README); we want TL, TR, BR, BL


def preprocess(frame, inp):
    """Returns (tensor, Transform). The input is 480 high x 640 wide: input_hw(inp) gives (480, 640)."""
    # RobotPilots' preprocessing (network_preprocess.cpp): resize to fit, and if the shapes don't match,
    # center it on gray (124) padding. Then their OpenVINO step swaps BGR -> RGB and divides by 255.
    # NCHW. Use the model's dtype (inp.dtype): check what this "fp16" file wants in m1.
    # ANSWER-START
    canvas, tf = letterbox(frame, input_hw(inp), align="center", pad=PAD)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x.astype(inp.dtype)), tf
    # ANSWER-END


def as_rows(output):
    """(1, 21, N) one column per guess -> (N, 21) float32, one row per guess."""
    # ANSWER-START
    return np.squeeze(output, 0).T.astype(np.float32)
    # ANSWER-END


def decode(outputs, tf, score_thr=SCORE_THRESHOLD, iou_thr=IOU_THRESHOLD):
    """Returns a list of detection dicts, best first, duplicates removed."""
    # 1. rows = as_rows(outputs[0]): each row is 4 color scores | 9 class scores | 8 corner values
    # 2. score = the biggest of the 9 class scores, class = which one. Keep rows with score >= score_thr
    # 3. color = argmax of the 4 color scores
    # 4. corners = columns 13-20 -> (4, 2) -> CORNER_ORDER -> tf.to_frame()
    # 5. make_detection(corners, score, color, number, size, label) using CLASSES, then nms(dets, iou_thr)
    # ANSWER-START
    rows = as_rows(outputs[0])
    classes = rows[:, 4:13]
    best, cls = classes.max(1), classes.argmax(1)
    dets = []
    for i in np.flatnonzero(best >= score_thr):
        number, size = CLASSES[cls[i]]
        color_id = int(np.argmax(rows[i, 0:4]))
        corners = tf.to_frame(rows[i, 13:21].reshape(4, 2)[CORNER_ORDER])
        dets.append(make_detection(corners, best[i], COLORS[color_id], number, size, f"color {color_id} class {cls[i]}"))
    return nms(dets, iou_thr)
    # ANSWER-END


def investigate(outputs, tf):
    """Part A: print the 5 best guesses, return the best one's corners as 4 points."""
    rows = as_rows(outputs[0])
    best = rows[:, 4:13].max(1)
    print(f"    {rows.shape[0]} guesses x {rows.shape[1]} values -> 4 color scores + 9 class scores + 8 corner values")
    print("    guess | colors B R W P          | best class (score) | corners x1 y1 ... x4 y4 (input pixels)")
    for r in np.argsort(-best)[:5]:
        print(f"    {r:5d} | " + " ".join(f"{v:.2f}" for v in rows[r, 0:4]) + f" | {rows[r, 4:13].argmax():3d} ({best[r]:.2f})"
              + "         | " + " ".join(f"{v:5.0f}" for v in rows[r, 13:21]))
    return tf.to_frame(rows[int(np.argmax(best)), 13:21].reshape(4, 2))


if __name__ == "__main__":
    family_demo(__import__(__name__), "szurp_v8", checks)
