"""Helpers every family module shares: frames, letterbox, NMS, corner order, drawing, running.

A detection is a dict, the same as in exercise 5, so exercise 7's PnP code works on any model:
    corners  4x2 float32 array: top-left, top-right, bottom-right, bottom-left, in FRAME pixels
    score    0..1
    color    "blue" | "red" | "gray" | "purple"
    number   "sentry" | "1" ... "5" | "outpost" | "base" | "?" when the model doesn't say
    size     "small" | "big" | None when the model doesn't say
    label    the model's own class name, untouched (useful when a mapping looks wrong)
"""
import argparse
import importlib
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

import zoo

FAMILY_MODULES = {"yolov5": "m2_yolov5", "yolox": "m3_yolox", "ultralytics": "m4_ultralytics",
                  "szurp_v8": "m4b_szurp_v8", "end2end": "m5_end2end", "yolo26_e2e": "m5b_yolo26"}
OUT_DIR = zoo.HERE / "out"  # every script saves its pictures here


# ----------------------------------------------------------------------------- frames
def read_frames(source=None, start=0, count=None, step=1, **settings):
    """Yield (index, frame) from anything cameras.py can open: a video (default: the demo video),
    a USB / HikRobot / MindVision / CSI camera, a stream or a folder of images."""
    from cameras import open_camera

    try:
        cam = open_camera(source, **settings)
    except (OSError, RuntimeError) as e:
        hint = " It should be videos/demo.avi in this repo, or set VIDEO=/path/to/video." if source is None else ""
        raise SystemExit(f"Can't open {source or zoo.VIDEO_PATH}: {e}.{hint}") from None
    with cam:
        yield from cam.frames(start, count, step)


def read_frame(index=60, source=None, **settings):
    """Frame number `index` (on a live camera: skip that many frames first, which also lets
    its exposure settle)."""
    for _, frame in read_frames(source, start=index, count=1, **settings):
        return frame
    raise SystemExit(f"{source or zoo.VIDEO_PATH} has fewer than {index + 1} frames")


# ----------------------------------------------------------------------------- letterbox
class Transform(namedtuple("Transform", "scale pad_x pad_y in_h in_w")):
    """How a frame was squeezed into the model input. to_frame() undoes it for points."""

    def to_frame(self, points):
        pts = np.asarray(points, np.float32)
        return ((pts - np.array([self.pad_x, self.pad_y], np.float32)) / self.scale).astype(np.float32)


def letterbox(img, size_hw, align="topleft", pad=0):
    """Resize img to fit inside size_hw (height, width) without stretching it.

    align="topleft": image in the top-left corner, padding right/bottom (sp_vision's C++)
    align="center":  image centered, padding split on both sides (TUP, Ultralytics, AT)
    pad:             the value the padding is filled with (0 = black, 114 = Ultralytics gray)
    Returns (canvas uint8 HxWx3, Transform).
    """
    h, w = img.shape[:2]
    out_h, out_w = size_hw
    scale = min(out_h / h, out_w / w)
    new_w, new_h = min(out_w, round(w * scale)), min(out_h, round(h * scale))  # round: 311.9999 is 312
    pad_x, pad_y = ((out_w - new_w) // 2, (out_h - new_h) // 2) if align == "center" else (0, 0)
    canvas = np.full((out_h, out_w, 3), pad, np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(img, (new_w, new_h))
    return canvas, Transform(scale, pad_x, pad_y, out_h, out_w)


def input_hw(inp):
    """(height, width) from an NCHW input Tensor."""
    return int(inp.shape[2]), int(inp.shape[3])


# ----------------------------------------------------------------------------- decoding helpers
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))  # tanh form can't overflow


def sort_corners(pts):
    """Order 4 points as TL, TR, BR, BL by position (what sp_vision does with yolo11's keypoints).
    Works when the plate isn't rolled much; a model with a FIXED keypoint order doesn't need it."""
    pts = np.asarray(pts, np.float32)
    by_y = pts[np.argsort(pts[:, 1])]
    top, bottom = by_y[:2][np.argsort(by_y[:2, 0])], by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], np.float32)


def make_detection(corners, score, color, number="?", size=None, label=""):
    return {"corners": np.asarray(corners, np.float32).reshape(4, 2), "score": float(score), "color": color,
            "number": number, "size": size, "label": str(label)}


def nms(dets, iou_thr=0.3):
    """Non-maximum suppression on the rectangle around each plate's corners. Best score first."""
    if len(dets) <= 1:
        return list(dets)
    boxes = []
    for d in dets:
        (x0, y0), (x1, y1) = d["corners"].min(0), d["corners"].max(0)
        boxes.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    keep = cv2.dnn.NMSBoxes(boxes, [d["score"] for d in dets], 0.0, iou_thr)
    return sorted((dets[i] for i in np.array(keep, int).reshape(-1)), key=lambda d: -d["score"])


def match(dets_a, dets_b, max_px=20.0):
    """Pair up plates from two lists whose centers are within max_px. Greedy, best score first.
    Returns a list of (index in dets_a, index in dets_b)."""
    pairs, used = [], set()
    for i in sorted(range(len(dets_a)), key=lambda k: -dets_a[k]["score"]):
        center = dets_a[i]["corners"].mean(0)
        best, best_dist = None, max_px
        for j, b in enumerate(dets_b):
            dist = float(np.linalg.norm(b["corners"].mean(0) - center))
            if j not in used and dist <= best_dist:
                best, best_dist = j, dist
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def is_big(det):
    """Big or small plate, for PnP. Use the model's answer if it gives one. Otherwise guess from
    the number: hero (1) and base use big plates. Balance infantry (big 3/4/5) gets this wrong."""
    if det.get("size"):
        return det["size"] == "big"
    return det.get("number") in ("1", "base")


# ----------------------------------------------------------------------------- running
def family_module(family):
    return importlib.import_module(FAMILY_MODULES[family])


def detect(module, runner, frame, **kw):
    """preprocess -> run -> decode, timing each step. Returns (detections, {step: seconds})."""
    t0 = time.perf_counter()
    x, tf = module.preprocess(frame, runner.inputs[0])
    t1 = time.perf_counter()
    outputs = runner(x)
    t2 = time.perf_counter()
    dets = module.decode(outputs, tf, **kw)
    t3 = time.perf_counter()
    return dets, {"pre": t1 - t0, "infer": t2 - t1, "post": t3 - t2}


def load_runners(models):
    """Yield (model, runner, None) for every model that loads here, or (model, None, why_not)."""
    import runtimes

    for m in models:
        try:
            yield m, runtimes.load(m), None
        except ImportError as e:
            yield m, None, f"{m.format} runtime not installed ({e.name})"
        except Exception as e:
            yield m, None, str(e).splitlines()[0][:160]


# ----------------------------------------------------------------------------- drawing
COLOR_BGR = {"blue": (255, 128, 0), "red": (0, 0, 255), "gray": (170, 170, 170), "purple": (255, 0, 200)}


def draw(img, dets, title=None):
    """Outline each plate, number its corners 0-3 (should go TL, TR, BR, BL) and label it."""
    for d in dets:
        pts = d["corners"].astype(np.int32)
        color = COLOR_BGR.get(d["color"], (0, 255, 0))
        cv2.polylines(img, [pts], True, color, 2)
        for i, (x, y) in enumerate(pts):
            cv2.putText(img, str(i), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        size = f" {d['size']}" if d.get("size") else ""
        text = f"{d['color']} {d['number']}{size} {d['score']:.2f}"
        x, y = pts[:, 0].min(), pts[:, 1].min() - 10
        cv2.putText(img, text, (int(x), int(max(y, 20))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if title:
        cv2.putText(img, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def draw_points(img, pts, title=None):
    """Draw 4 raw points numbered in the order the model gave them (Part A: what order is it?)."""
    for i, (x, y) in enumerate(np.asarray(pts).astype(int)):
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i), (int(x) + 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if title:
        cv2.putText(img, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


# ----------------------------------------------------------------------------- family scripts (m2-m5b)
def family_demo(module, family, checks):
    """What every family script does when you run it:
    1. check the functions Part A needs (preprocess and friends) on synthetic input
    2. Part A: run every model of this family that loads here and let module.investigate() show
       the RAW output, so you can work out the layout before decoding it
    3. check decode() on a synthetic output
    4. Part B: decode the real output and save a picture per model in out/
    """
    from cameras import add_source_args, source_from_args

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated names from zoo.py (default: the whole family)")
    ap.add_argument("--frame", type=int, default=60, help="which frame to use (on a camera: skip this many first)")
    add_source_args(ap)
    args = ap.parse_args()
    source, settings = source_from_args(args)
    OUT_DIR.mkdir(exist_ok=True)

    if not checks.check_part_a(family, module):
        raise SystemExit
    frame = read_frame(args.frame, source, **settings)
    loaded = []
    print(f"\n=== Part A: raw outputs on frame {args.frame} ===")
    for m, runner, why in load_runners(zoo.find(args.models, family=family)):
        if runner is None:
            print(f"\n--- {m.name}: skipped ({why})")
            continue
        x, tf = module.preprocess(frame, runner.inputs[0])
        outputs = runner(x)
        print(f"\n--- {m.name}  [{runner.backend}]  input {x.dtype} {x.shape}  ->  outputs "
              + ", ".join(f"{t.name or '(no name)'} {o.shape}" for t, o in zip(runner.outputs, outputs)))
        pts = module.investigate(outputs, tf)
        if pts is not None:
            path = OUT_DIR / f"{m.name}_raw_points.jpg"
            cv2.imwrite(str(path), draw_points(frame.copy(), pts, f"{m.name}: raw point order"))
            print(f"    saved {path.relative_to(zoo.HERE)}: the best row's 4 points, numbered as they come out")
        loaded.append((m, runner, outputs, tf))

    if not checks.check_part_b(family, module):
        raise SystemExit
    print(f"=== Part B: decoded detections on frame {args.frame} ===")
    for m, runner, outputs, tf in loaded:
        dets = module.decode(outputs, tf)
        print(f"\n--- {m.name}: {len(dets)} plate(s)")
        for d in dets:
            size = f" ({d['size']})" if d["size"] else ""
            print(f"    {d['color']:6s} {d['number']:7s}{size}  score {d['score']:.2f}  model label {d['label']!r}")
        path = OUT_DIR / f"{m.name}_decoded.jpg"
        cv2.imwrite(str(path), draw(frame.copy(), dets, m.name))
        print(f"    saved {path.relative_to(zoo.HERE)}")
