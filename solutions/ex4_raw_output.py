"""Exercise 4: figure out what the 22 numbers in each output row mean, then decode one row.

Run:  python3 ex4_raw_output.py
"""
import cv2
import numpy as np

import checks
import config
from backends import get_model, read_frame
from ex3_preprocess import preprocess

# The model's corners come out as TL, BL, BR, TR. These column indices pick them in
# TL, TR, BR, BL order, which is the order the PnP step (exercise 7) expects.
CORNER_COLUMNS = [0, 1, 6, 7, 4, 5, 2, 3]


def sigmoid(x):
    """Logit -> probability. Written with tanh so huge inputs can't overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def decode_row(row, scale):
    """Decode one 22-value row into a dict:
    corners (4x2, TL/TR/BR/BL, original-image pixels), score, color, color_id, number, number_id.
    """
    corners = row[CORNER_COLUMNS].reshape(4, 2) / scale
    color_id = int(np.argmax(row[9:13]))
    number_id = int(np.argmax(row[13:22]))
    return {
        "corners": corners.astype(np.float32),
        "score": float(sigmoid(row[8])),
        "color_id": color_id,
        "color": config.COLORS[color_id],
        "number_id": number_id,
        "number": config.NUMBERS[number_id],
    }


if __name__ == "__main__":
    # Part A: investigate. This runs before the checks so you can use it to work out the layout.
    model = get_model()
    frame = read_frame()
    x, scale = preprocess(frame)
    raw = model(x)["output/sink_port_0"][0]  # (25200, 22)

    # Investigate: the 5 rows with the largest value in column 8
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
    print("corners (TL, TR, BR, BL):", d["corners"].round(1).tolist())
    out = frame.copy()
    cv2.polylines(out, [d["corners"].astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite("ex4_decoded.jpg", out)
    print("Saved ex4_decoded.jpg")
