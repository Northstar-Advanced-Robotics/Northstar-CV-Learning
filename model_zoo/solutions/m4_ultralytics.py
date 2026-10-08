"""m4: family "ultralytics" - YOLOv8 / YOLO11 pose models (reborn_yolov8_int8, tongji_yolo11_int8/_onnx).

These were trained with the Ultralytics library as "pose" models whose 4 keypoints are the plate
corners. Their output is the most different from what you've seen:
  - VALUES-FIRST: (1, 4 + classes + 8, N), one COLUMN per guess. Transpose it to get one row per guess
    (BASICS.md, section 6).
  - no separate "is this a plate?" score: a guess's score is its highest class score (already 0..1)
  - box (cx, cy, w, h) and keypoints are already in input pixels
  - the classes say a lot or very little:
        reborn_yolov8 (416x416):  2 classes, "B" and "R". Color only! On the robot, a second, separate
                                  model reads the number on each plate (not part of this track).
        tongji_yolo11     (640x640): 38 classes such as "Bfour", "Rbasesmall", "Ebalancethree" that
                                  pack color + number + plate size into one label.
  - the labels are stored INSIDE the file (look at the m1 card). An engine loses them, which is why
    decode() picks the table from the number of classes instead.

Run:  python3 m4_ultralytics.py
"""
import _path  # noqa: F401  (solutions-only)
import cv2
import numpy as np

import checks
from common import family_demo, input_hw, letterbox, make_detection, nms

SCORE_THRESHOLD = 0.7  # the robot's value (../reference/sp_vision_25/tasks/auto_aim/yolos/yolo11.hpp)
CORNER_ORDER = [0, 3, 2, 1]  # keypoints come out TL, BL, BR, TR (Reborn's labeling README); we want TL, TR, BR, BL

# reborn_yolov8: labels "B R"
REBORN_CLASSES = [("blue", "?", None), ("red", "?", None)]
REBORN_LABELS = ["B", "R"]

# tongji_yolo11: armor_properties in ../reference/sp_vision_25/tasks/auto_aim/armor.hpp (same order as the labels in the
# file). E = "extinguished" (gray). P = purple (the base while it's invincible). "balance" = big 3/4/5.
TONGJI_CLASSES = [(c, n, "small") for n in ("sentry", "1", "2", "3", "4", "5", "outpost") for c in ("blue", "red", "gray")]
TONGJI_CLASSES += [(c, "base", s) for s in ("big", "small") for c in ("blue", "red", "gray", "purple")]
TONGJI_CLASSES += [(c, n, "big") for n in ("3", "4", "5") for c in ("blue", "red", "gray")]
TONGJI_LABELS = ("Bsentry Rsentry Esentry Bone Rone Eone Btwo Rtwo Etwo Bthree Rthree Ethree Bfour Rfour Efour "
                 "Bfive Rfive Efive Boutpost Routpost Eoutpost Bbase Rbase Ebase Pbase Bbasesmall Rbasesmall "
                 "Ebasesmall Pbasesmall Bbalancethree Rbalancethree Ebalancethree Bbalancefour Rbalancefour "
                 "Ebalancefour Bbalancefive Rbalancefive Ebalancefive").split()

TABLES = {2: (REBORN_CLASSES, REBORN_LABELS), 38: (TONGJI_CLASSES, TONGJI_LABELS)}


def preprocess(frame, inp):
    """Returns (tensor, Transform) the way the models were TRAINED."""
    # Ultralytics trains with the image CENTERED and gray (114) bars: the files even say so in their
    # metadata (model_info: pad_value 114, reverse_input_channels YES, scale_values 255).
    # "reverse_input_channels" and "scale_values" are instructions for YOU: the network wants RGB and
    # values 0..1, so swap BGR -> RGB and divide by 255. NCHW, float32.
    # (The robot's code feeds them top-left with black bars instead. Try it: does the score change?)
    # ANSWER-START
    canvas, tf = letterbox(frame, input_hw(inp), align="center", pad=114)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), tf
    # ANSWER-END


def as_rows(output):
    """(1, C, N) values-first output -> (N, C) float32: one row per guess (C values each, N guesses)."""
    # ANSWER-START
    return np.squeeze(output, 0).T.astype(np.float32)
    # ANSWER-END


def decode(outputs, tf, score_thr=SCORE_THRESHOLD, iou_thr=0.3):
    """Returns a list of detection dicts, best first, duplicates removed."""
    # 1. rows = as_rows(outputs[0]). Each row: cx, cy, w, h, <nc class scores>, 8 keypoint values,
    #    so nc = row length - 4 - 8. Look up (classes, labels) = TABLES[nc].
    # 2. score = the highest class score, class = which one. Keep rows with score >= score_thr.
    # 3. keypoints -> reshape (4, 2) -> CORNER_ORDER -> tf.to_frame()
    # 4. make_detection(corners, score, color, number, size, label) from the class table, then nms()
    # ANSWER-START
    rows = as_rows(outputs[0])
    nc = rows.shape[1] - 4 - 8
    classes, labels = TABLES[nc]
    scores = rows[:, 4:4 + nc]
    best, cls = scores.max(1), scores.argmax(1)
    dets = []
    for i in np.flatnonzero(best >= score_thr):
        color, number, size = classes[cls[i]]
        kpts = rows[i, 4 + nc:4 + nc + 8].reshape(4, 2)[CORNER_ORDER]
        dets.append(make_detection(tf.to_frame(kpts), best[i], color, number, size, labels[cls[i]]))
    return nms(dets, iou_thr)
    # ANSWER-END


def investigate(outputs, tf):
    """Part A: print the 5 best candidates, return the best one's keypoints as 4 points."""
    rows = as_rows(outputs[0])
    nc = rows.shape[1] - 4 - 8
    best = rows[:, 4:4 + nc].max(1)
    print(f"    {rows.shape[0]} candidates x {rows.shape[1]} values -> 4 box + {nc} class scores + 8 keypoint values")
    print("    cand    cx     cy     w      h   | best class (score) | keypoints x1 y1 ... x4 y4 (input pixels)")
    for r in np.argsort(-best)[:5]:
        print(f"    {r:4d} " + " ".join(f"{v:6.1f}" for v in rows[r, :4]) + f" | {rows[r, 4:4 + nc].argmax():3d} ({best[r]:.2f})"
              + "       | " + " ".join(f"{v:5.0f}" for v in rows[r, 4 + nc:]))
    return tf.to_frame(rows[int(np.argmax(best)), 4 + nc:4 + nc + 8].reshape(4, 2))


if __name__ == "__main__":
    family_demo(__import__(__name__), "ultralytics", checks)
