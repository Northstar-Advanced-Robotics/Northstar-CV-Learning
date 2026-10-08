"""Exercise 5: decode ALL 25200 rows, keep the confident ones, remove duplicates with NMS.

Run:  python3 ex5_decode.py
"""
import math

import cv2
import numpy as np

import checks
import config
from backends import get_model, read_frame
from ex3_preprocess import preprocess
from ex4_raw_output import CORNER_COLUMNS, sigmoid


def decode_all(raw, scale, score_thr=config.SCORE_THRESHOLD, iou_thr=config.NMS_IOU_THRESHOLD):
    """raw: (25200, 22) model output. Returns a list of detection dicts (same keys as
    decode_row in exercise 4), best score first, with duplicates removed."""
    raw = raw.reshape(-1, 22)

    # 1. Threshold on the logit directly: sigmoid is monotonic, so this skips 25000+ sigmoids
    logit_thr = math.log(score_thr / (1 - score_thr))
    cand = raw[raw[:, 8] >= logit_thr]
    if len(cand) == 0:
        return []

    # 2. Decode every surviving row at once (vectorized: no Python loop over rows)
    scores = sigmoid(cand[:, 8])
    corners = cand[:, CORNER_COLUMNS].reshape(-1, 4, 2) / scale
    color_ids = cand[:, 9:13].argmax(1)
    number_ids = cand[:, 13:22].argmax(1)

    # 3. NMS works on axis-aligned boxes, so use the rectangle around the four corners
    mins, maxs = corners.min(1), corners.max(1)
    boxes = np.hstack([mins, maxs - mins])  # x, y, w, h
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_thr, iou_thr)
    keep = np.array(keep, dtype=int).reshape(-1)

    dets = [{
        "corners": corners[i].astype(np.float32),
        "score": float(scores[i]),
        "color_id": int(color_ids[i]),
        "color": config.COLORS[color_ids[i]],
        "number_id": int(number_ids[i]),
        "number": config.NUMBERS[number_ids[i]],
    } for i in keep]
    return sorted(dets, key=lambda d: -d["score"])


COLOR_BGR = {"blue": (255, 128, 0), "red": (0, 0, 255), "gray": (160, 160, 160), "purple": (255, 0, 200)}


def draw_detections(img, dets):
    """Draw each plate's outline, a dot at its center, and a 'color number score' label."""
    for d in dets:
        pts = d["corners"].astype(np.int32)
        color = COLOR_BGR[d["color"]]
        cv2.polylines(img, [pts], True, color, 2)
        cx, cy = d["corners"].mean(0).astype(int)
        cv2.circle(img, (cx, cy), 4, color, -1)
        label = f"{d['color']} {d['number']} {d['score']:.2f}"
        x, y = pts[:, 0].min(), pts[:, 1].min() - 8
        cv2.putText(img, label, (int(x), int(max(y, 15))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
