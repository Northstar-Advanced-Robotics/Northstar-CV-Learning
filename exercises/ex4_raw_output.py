"""Exercise 4: figure out what the 22 numbers in each output row mean, then decode one row.

The model answers with 25200 rows of 22 numbers: one row per guess (BASICS.md, section 6).
Nothing tells you which number means what. That's what you work out here.

Run this file as soon as exercise 3 works (on a laptop, with BACKEND=onnxruntime). Part A
prints the 5 rows the model is most sure about and draws their first 8 numbers as 4 points,
BEFORE checking your code: use it to investigate. Then fill in the functions and run it again.

Where to look: YOLOV5::parse() in reference/sp_vision_25/tasks/auto_aim/yolos/yolov5.cpp, and
what each color and number index means in reference/sp_vision_25/tasks/auto_aim/armor.hpp.

Run:  python3 ex4_raw_output.py
"""
import cv2
import numpy as np

import checks
import config  # noqa: F401  (config.COLORS and config.NUMBERS hold the label names)
from backends import get_model, read_frame
from ex3_preprocess import preprocess


def sigmoid(x):
    """Turn a logit (a raw score like 2.6 or -6) into a probability from 0 to 1.

    The formula is 1 / (1 + e^-x) (BASICS.md, section 6). np.exp(-x) is e^-x, and it works on
    single numbers and on whole numpy arrays, which is what this function must handle too.
    """
    # TODO
    return 1 / (1 + np.exp(-x))


def decode_row(row, scale):
    """Decode one row (a numpy array of 22 numbers) into a dict with these keys:

      "corners"   -- the 4 plate corners as a 4x2 array (4 rows of x, y) in ORIGINAL frame pixels,
                     in this order: top-left, top-right, bottom-right, bottom-left
      "score"     -- how sure the model is, from 0 to 1
      "color_id"  -- the index of the winning color score (an int), and "color": its name from config.COLORS
      "number_id" -- the index of the winning number score (an int), and "number": its name from config.NUMBERS

    scale is the letterbox scale from exercise 3: the corners come out in the 640x640 input's pixels,
    so divide by scale to get back to the original frame (BASICS.md, section 6, "Back to frame pixels").
    """
    # TODO
    decoded = {}

    decoded["corners"] = np.array([[row[0]/scale,row[1]/scale], [row[6]/scale,row[7]/scale], [row[4]/scale,row[5]/scale], [row[2]/scale,row[3]/scale]])
    decoded["score"] = sigmoid(row[8])  
    max = 0
    index = 0
    for i in range(9, 12):
        if(sigmoid(row[i]) > max):
            max = row[i]
            index = i
    decoded["color_id"] = index - 9
    decoded["color"] = config.COLORS[index - 9]
    max = 0
    index = 0
    for i in range (13,21):
        if(sigmoid(row[i]) > max):
            max = row[i]
            index = i
    decoded["number_id"] = index - 13
    decoded["number"] = config.NUMBERS[index - 13]

    return decoded

    

if __name__ == "__main__":
    # Part A: investigate. This runs before the checks so you can use it to work out the layout.
    model = get_model()
    frame = read_frame()
    x, scale = preprocess(frame)
    raw = model(x)["output/sink_port_0"][0]  # (25200, 22)

    # The 5 rows with the largest value in column 8 (why column 8? try other columns too)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    top = np.argsort(-raw[:, 8])[:5]
    print("col: " + " ".join(f"{i:>6d}" for i in range(22)))
    for r in top:
        print(f"{r:5d}" + " ".join(f"{v:6.1f}" for v in raw[r]))

    # Draw the first 8 values of the best row as 4 points, labeled in the RAW order
    debug = frame.copy()
    pts = raw[top[0], :8].reshape(4, 2) / scale
    for i, (px, py) in enumerate(pts.astype(int)):
        cv2.circle(debug, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(debug, str(i), (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite("ex4_raw_points.jpg", debug)
    print("\nSaved ex4_raw_points.jpg: the first 8 values of the top row, drawn as 4 numbered points")

    # Part B: check your decode_row(), then use it on the real output
    if not checks.check_ex4(sigmoid, decode_row):
        raise SystemExit
    d = decode_row(raw[top[0]], scale)
    print(f"\nBest row decoded: {d['color']} {d['number']}  score {d['score']:.2f}")
    print("corners (TL, TR, BR, BL):", np.asarray(d["corners"]).round(1).tolist())
    out = frame.copy()
    cv2.polylines(out, [np.asarray(d["corners"]).astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite("ex4_decoded.jpg", out)
    print("Saved ex4_decoded.jpg")
