"""m3: family "yolox" - TUP's 2022 armor models (tup_0527, tup_1208 as ONNX and two NCNN files).

TUP trained a YOLOX-style network. Compared with the yolov5 family, almost everything at the
edges is different (grid cells and strides: BASICS.md, section 8):
  - input: 416x416, image CENTERED with black bars, colors left as BGR, values 0..255 (no /255!)
  - output: 3549 rows x 21 values. One row per GRID CELL:
        52x52 cells (stride 8) + 26x26 (stride 16) + 13x13 (stride 32) = 2704 + 676 + 169 = 3549
  - the 8 corner values are NOT pixels yet: they're offsets in cells, relative to the cell
  - column 8 (how sure it is that this is a plate, called "objectness") and the color/class scores
    already went through a sigmoid inside the model: they're already 0..1
  - the NCNN file made with pnnx returns the same numbers TRANSPOSED: (21, 3549)
Source: TUP-InfantryVision-2022, autoaim/detector/inference.cpp (scaledResize, generateYoloxProposals).

Run:  python3 m3_yolox.py
"""
import _path  # noqa: F401  (solutions-only)
import numpy as np

import checks
from common import family_demo, input_hw, letterbox, make_detection, nms

SCORE_THRESHOLD = 0.75  # TUP's BBOX_CONF_THRESH
STRIDES = (8, 16, 32)
COLORS = ["blue", "red", "gray", "purple"]  # TUP autoaim.cpp: keys B, R, N (none), P
NUMBERS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base"]  # 0 sentry, 1 hero, 2 engineer, 3-5 infantry
CORNER_ORDER = [0, 3, 2, 1]  # the model gives TL, BL, BR, TR (look at Part A's picture); we want TL, TR, BR, BL


def preprocess(frame, inp):
    """Returns (tensor, Transform) the way TUP's C++ feeds the model."""
    # TUP's scaledResize(): scale to fit, then copyMakeBorder() adds EQUAL black bars on both sides,
    # so the image sits in the CENTER (letterbox(..., align="center", pad=0)).
    # Then they convert to float32 and copy the B, G and R planes straight in: no cvtColor, no /255.
    # So: BGR, values 0..255, NCHW, float32.
    # ANSWER-START
    canvas, tf = letterbox(frame, input_hw(inp), align="center", pad=0)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(x), tf
    # ANSWER-END


def as_rows(output):
    """One model output -> a (3549, 21) float32 array, whatever orientation the runtime gave it."""
    # ONNX Runtime and the onnx2ncnn file give (1, 3549, 21). The pnnx NCNN file gives (1, 21, 3549).
    # Drop the batch dimension (np.squeeze) and transpose if the 21 is first.
    # ANSWER-START
    a = np.squeeze(output).astype(np.float32)
    return a.T if a.shape[0] == 21 and a.shape[1] != 21 else a
    # ANSWER-END


def grid_strides(in_h, in_w):
    """(3549, 3) array: (cell x, cell y, stride) for every row, in the order the model emits rows."""
    # The model walks the stride-8 grid first, then 16, then 32. Inside each grid it goes row by
    # row (y outer loop, x inner loop). See generate_grids_and_stride() in TUP's inference.cpp.
    # np.mgrid[0:h, 0:w] gives you the y and x of every cell; .ravel() flattens them row by row.
    # ANSWER-START
    rows = []
    for s in STRIDES:
        gy, gx = np.mgrid[0:in_h // s, 0:in_w // s]
        rows.append(np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, s)], axis=1))
    return np.concatenate(rows).astype(np.float32)
    # ANSWER-END


def corners_in_input_pixels(rows, grids):
    """(N, 21) rows + their (N, 3) grid entries -> (N, 4, 2) corners in 416x416 input pixels."""
    # YOLOX decode, from TUP's comment:  pixel = (value + cell) * stride, for x with cell x and for
    # y with cell y. Columns 0-7 are x1, y1, x2, y2, x3, y3, x4, y4.
    # ANSWER-START
    xy = rows[:, :8].reshape(-1, 4, 2)
    return (xy + grids[:, None, :2]) * grids[:, None, 2:3]
    # ANSWER-END


def decode(outputs, tf, score_thr=SCORE_THRESHOLD, iou_thr=0.3):
    """Returns a list of detection dicts, best first, duplicates removed."""
    # 1. rows = as_rows(outputs[0]); grids = grid_strides(tf.in_h, tf.in_w)
    # 2. keep rows whose column 8 >= score_thr (it's ALREADY a probability: no sigmoid)
    # 3. corners_in_input_pixels() -> reorder with CORNER_ORDER -> tf.to_frame()
    #    (to_frame undoes the scale AND the centered padding)
    # 4. color = argmax of columns 9-12, number = argmax of columns 13-20
    # 5. make_detection(...) for each, then nms(dets, iou_thr)
    # ANSWER-START
    rows, grids = as_rows(outputs[0]), grid_strides(tf.in_h, tf.in_w)
    keep = rows[:, 8] >= score_thr
    rows, grids = rows[keep], grids[keep]
    corners = corners_in_input_pixels(rows, grids)[:, CORNER_ORDER]
    dets = []
    for row, c in zip(rows, corners):
        color_id, number_id = int(np.argmax(row[9:13])), int(np.argmax(row[13:21]))
        dets.append(make_detection(tf.to_frame(c), row[8], COLORS[color_id], NUMBERS[number_id],
                                   label=f"color {color_id} class {number_id}"))
    return nms(dets, iou_thr)
    # ANSWER-END


def investigate(outputs, tf):
    """Part A: print the 5 rows with the highest plate score (column 8), return the best row's 4 corners."""
    rows, grids = as_rows(outputs[0]), grid_strides(tf.in_h, tf.in_w)
    top = np.argsort(-rows[:, 8])[:5]
    print("    row   cell(x,y) stride | raw corner values (in cells)              | obj   colors        classes")
    for r in top:
        gx, gy, s = grids[r].astype(int)
        print(f"    {r:4d}  ({gx:2d},{gy:2d})  {s:2d}    | " + " ".join(f"{v:5.1f}" for v in rows[r, :8])
              + f" | {rows[r, 8]:.2f}  " + " ".join(f"{v:.2f}" for v in rows[r, 9:13])
              + "  " + " ".join(f"{v:.2f}" for v in rows[r, 13:21]))
    return tf.to_frame(corners_in_input_pixels(rows[top[:1]], grids[top[:1]])[0])


if __name__ == "__main__":
    family_demo(__import__(__name__), "yolox", checks)
