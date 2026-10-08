"""m2: family "yolov5" - RobotPilots' 4-corner YOLOv5 (rp_0708, rp_0526, rp_0708_openvino, engines).

You decoded this family in exercises 4-5 (the main track's model IS rp_0708, converted). What's new
is how many FILES carry it, and how they differ at the edges (words: BASICS.md, sections 5 and 6):
  - rp_0526 wants float16 input; everything else wants float32
  - the raw output is called "output" in some files and "output/sink_port_0" in others, and the
    engine built from the NMS-fused ONNX has 4 more outputs next to it
  - RobotPilots' README says the 4 color columns are red, blue, gray, purple. The robot's code reads
    them as blue, red, gray, purple. The demo video only has BLUE plates: which one is right?

Run:  python3 m2_yolov5.py                      # every file of this family that runs here
      python3 m2_yolov5.py --models rp_0526     # just one
"""
import _path  # noqa: F401  (solutions-only)
import cv2
import numpy as np

import checks
from common import family_demo, input_hw, letterbox, make_detection, nms, sigmoid

SCORE_THRESHOLD = 0.7  # the robot's value (../reference/sp_vision_25/tasks/auto_aim/yolos/yolov5.hpp)

# Column 9-12: color. Checked on the demo video (all plates blue -> column 9 is the biggest), so
# the robot's code has the order right and RobotPilots' README doesn't.
COLORS = ["blue", "red", "gray", "purple"]
# Column 13-21: what's on the plate. RobotPilots' README: G 1 2 3 4 5 O Bs Bb, where Bs/Bb are the
# base's small and big plates. (The robot's code calls index 8 "not_armor": we follow the people who trained it.)
NUMBERS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base", "base"]
SIZES = [None] * 7 + ["small", "big"]
# Columns 0-7: 4 corners as TL, BL, BR, TR (counter-clockwise on screen). This picks TL, TR, BR, BL.
CORNER_COLUMNS = [0, 1, 6, 7, 4, 5, 2, 3]


def preprocess(frame, inp):
    """frame: BGR uint8 camera image. inp: the model's input Tensor (name, shape, dtype).
    Returns (tensor, Transform), the tensor shaped and typed exactly as inp says."""
    # The same recipe as exercise 3: the robot's code puts the picture in the TOP-LEFT corner of a black
    # square (common.letterbox does that), swaps BGR -> RGB, reorders (height, width, channels) into
    # (1, channels, height, width), and divides by 255 to get values 0..1.
    # New: the dtype must match the model, and rp_0526 is float16. Do the math in float32 and
    # convert to inp.dtype at the very end.
    # ANSWER-START
    canvas, tf = letterbox(frame, input_hw(inp), align="topleft", pad=0)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x.astype(inp.dtype)), tf
    # ANSWER-END


def raw_output(outputs):
    """Find the (25200, 22) raw output among everything the file returns, whatever it's called."""
    # Don't look the output up by NAME: it's "output" in 0708.onnx and the OpenVINO file, but
    # "output/sink_port_0" in the NMS-fused ONNX and its engine (which also returns num_detections,
    # detection_boxes, ...). Look for the SHAPE instead: the array whose last dimension is 22.
    # Return it as a float32 (N, 22) array.
    # ANSWER-START
    for o in outputs:
        if o.shape[-1] == 22 and o.size >= 22 * 1000:
            return o.reshape(-1, 22).astype(np.float32)
    raise ValueError(f"no (N, 22) output among {[o.shape for o in outputs]}")
    # ANSWER-END


def decode(outputs, tf, score_thr=SCORE_THRESHOLD, iou_thr=0.3):
    """outputs: the list the runner returned. tf: the Transform from preprocess().
    Returns a list of detection dicts (see common.py), best first, duplicates removed."""
    # Exercise 5 again, on raw_output(outputs):
    #   1. column 8 is a logit (a raw score, BASICS.md section 6): keep rows whose sigmoid >= score_thr
    #   2. corners: pick CORNER_COLUMNS, reshape to (-1, 4, 2), then tf.to_frame() maps them back
    #      to frame pixels (it undoes the letterbox scale AND padding)
    #   3. color = argmax of columns 9-12, number = argmax of columns 13-21 (COLORS, NUMBERS, SIZES)
    #   4. build each one with make_detection(...) and run nms(dets, iou_thr)
    # ANSWER-START
    raw = raw_output(outputs)
    cand = raw[sigmoid(raw[:, 8]) >= score_thr]
    dets = []
    for row in cand:
        color_id, number_id = int(np.argmax(row[9:13])), int(np.argmax(row[13:22]))
        dets.append(make_detection(tf.to_frame(row[CORNER_COLUMNS].reshape(4, 2)), sigmoid(row[8]),
                                   COLORS[color_id], NUMBERS[number_id], SIZES[number_id],
                                   label=f"color {color_id} number {number_id}"))
    return nms(dets, iou_thr)
    # ANSWER-END


def investigate(outputs, tf):
    """Part A: print the 5 most confident rows, return the best row's first 8 values as 4 points."""
    raw = raw_output(outputs)
    top = np.argsort(-raw[:, 8])[:5]
    print("    col: " + " ".join(f"{i:>5d}" for i in range(22)))
    for r in top:
        print(f"    {r:5d}" + "".join(f"{v:6.1f}" for v in raw[r]))
    return tf.to_frame(raw[top[0], :8].reshape(4, 2))


if __name__ == "__main__":
    family_demo(__import__(__name__), "yolov5", checks)
