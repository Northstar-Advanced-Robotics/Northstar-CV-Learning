"""m5: family "end2end" - HEBUST Actor&Thinker's models (at_c2psa_640, at_c2psa_576x768).

"End-to-end" means the clean-up is INSIDE the model: it picks its 30 best guesses, and it was trained
so that each plate gets only one guess. So the output is already a short list of finished plates,
(1, 30, 18), and there is NO NMS to write (BASICS.md, section 7).
Each of the 30 rows:
    [0:4]   x1, y1, x2, y2   box, in the model's input pixels (undo the letterbox for frame pixels)
    [4]     confidence       already 0..1; unused rows are ~0
    [5]     class id         a FLOAT holding a whole number 0..11 (see CLASSES)
    [6:10]  color scores     blue, red, gray, purple
    [10:18] 4 corners        x, y pairs, in the model's input pixels
Input: RGB, 0..1, image CENTERED with gray (114) bars. The 576x768 version is not square: it's
height 576 x width 768, the same 4:3 shape as the 1440x1080 camera, so the frame needs no bars.
Source: https://github.com/PraySky1337/AT_NN_Detector (README).

Run:  python3 m5_end2end.py
"""
import _path  # noqa: F401  (solutions-only)
import cv2
import numpy as np

import checks
from common import family_demo, input_hw, letterbox, make_detection

SCORE_THRESHOLD = 0.5  # AT's preview script uses 0.25; real detections on the demo score ~0.9
COLORS = ["blue", "red", "gray", "purple"]
# Class names are s<size>_o<object>: size 0 = small plate, 1 = big plate;
# object 0 sentry, 1-5 robots, 6 outpost, 7 base.
CLASSES = ["s0_o0", "s0_o2", "s0_o3", "s0_o4", "s0_o5", "s0_o6", "s1_o0", "s1_o1", "s1_o3", "s1_o4", "s1_o5", "s1_o7"]
OBJECTS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base"]
# The README says the keypoints go "clockwise". On screen (y pointing DOWN) they go TL, BL, BR, TR,
# which looks counter-clockwise: "clockwise" depends on which way your y axis points.
CORNER_ORDER = [0, 3, 2, 1]


def preprocess(frame, inp):
    """Returns (tensor, Transform). Careful: the input can be 576 high x 768 wide."""
    # AT's README: letterbox with the image CENTERED and gray (114) bars, BGR -> RGB, /255, NCHW,
    # float32. input_hw(inp) gives (height, width) in that order.
    # ANSWER-START
    canvas, tf = letterbox(frame, input_hw(inp), align="center", pad=114)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), tf
    # ANSWER-END


def class_info(class_id):
    """Class id (0..11) -> (number, size), e.g. 7 -> ("1", "big")."""
    # CLASSES[class_id] is a name like "s1_o1". Split it: "s1" -> size, "o1" -> OBJECTS[1].
    # ANSWER-START
    size, obj = CLASSES[class_id].split("_")
    return OBJECTS[int(obj[1:])], "big" if size == "s1" else "small"
    # ANSWER-END


def decode(outputs, tf, score_thr=SCORE_THRESHOLD, iou_thr=None):
    """Returns a list of detection dicts, best first. iou_thr is ignored: the model did NMS."""
    # For each of the 30 rows with confidence >= score_thr:
    #   class id = int(round(row[5]))   (it's a float), then class_info()
    #   color    = argmax of row[6:10]
    #   corners  = row[10:18] -> (4, 2) -> CORNER_ORDER -> tf.to_frame()
    # Return them sorted by score, highest first.
    # ANSWER-START
    rows = np.squeeze(outputs[0], 0).astype(np.float32)
    dets = []
    for row in rows[rows[:, 4] >= score_thr]:
        class_id = int(round(float(row[5])))
        number, size = class_info(class_id)
        corners = tf.to_frame(row[10:18].reshape(4, 2)[CORNER_ORDER])
        dets.append(make_detection(corners, row[4], COLORS[int(np.argmax(row[6:10]))], number, size,
                                   CLASSES[class_id]))
    return sorted(dets, key=lambda d: -d["score"])
    # ANSWER-END


def investigate(outputs, tf):
    """Part A: print the rows, return the first row's keypoints as 4 points."""
    rows = np.squeeze(outputs[0], 0)
    print("    row | x1     y1     x2     y2   | conf  class | colors B R G P  | keypoints (input pixels)")
    for i, r in enumerate(rows[:6]):
        print(f"    {i:3d} | " + " ".join(f"{v:6.1f}" for v in r[:4]) + f" | {r[4]:.2f}  {r[5]:5.1f} | "
              + " ".join(f"{v:.2f}" for v in r[6:10]) + " | " + " ".join(f"{v:4.0f}" for v in r[10:18]))
    print(f"    ... {len(rows)} rows in total, {int((rows[:, 4] > 0.01).sum())} with confidence above 0.01")
    return tf.to_frame(rows[0, 10:18].reshape(4, 2))


if __name__ == "__main__":
    family_demo(__import__(__name__), "end2end", checks)
