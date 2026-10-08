"""Exercise 6: live detection on a camera or video, inside a region of interest (ROI).

Run:
    python3 ex6_live.py                                   # demo video, whole frame
    python3 ex6_live.py --roi 420 50 600 600              # demo video, ROI only
    python3 ex6_live.py --source usb:0 --width 1280 --height 720 --fps 60   # USB camera
    python3 ex6_live.py --source hik --cam exposure_ms=2   # the robot's HikRobot camera
    Any --source from shared/cameras.py works: a video, usb:N, hik, mv, csi:0, config:<camera .yaml>, ...

Keys in the window:  q quit   space pause   n toggle the engine's own NMS boxes (TensorRT only)
"""
import argparse
import time

import cv2
import numpy as np

import config
from backends import get_model
from cameras import add_source_args, open_camera, source_from_args
from ex3_preprocess import preprocess
from ex5_decode import decode_all, draw_detections


def detect_in_roi(model, frame, roi):
    """Run the model on frame[roi] only and return detections in FULL-frame pixels.
    Also returns the raw outputs and scale so the caller can inspect them."""
    x0, y0, w, h = roi
    crop = frame[y0:y0 + h, x0:x0 + w]
    tensor, scale = preprocess(crop)
    out = model(tensor)
    dets = decode_all(out["output/sink_port_0"][0], scale)
    for d in dets:
        d["corners"] += np.array([x0, y0], np.float32)  # crop pixels -> frame pixels
    return dets, out, scale


def draw_engine_nms(img, out, scale, roi):
    """Bonus: draw the boxes from the engine's fused EfficientNMS outputs.
    They are (min_y, min_x, max_y, max_x) in 640x640 input pixels -- see exercise 1."""
    if "num_detections" not in out:
        return
    n = int(out["num_detections"].reshape(-1)[0])
    for y1, x1, y2, x2 in out["detection_boxes"][0, :n]:
        p1 = (int(x1 / scale + roi[0]), int(y1 / scale + roi[1]))
        p2 = (int(x2 / scale + roi[0]), int(y2 / scale + roi[1]))
        cv2.rectangle(img, p1, p2, (255, 0, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    add_source_args(ap)  # --source (a video, usb:0, hik, csi:0, ...) and --cam KEY=VALUE: see shared/cameras.py
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--width", type=int, help="same as --cam width=...")
    ap.add_argument("--height", type=int, help="same as --cam height=...")
    ap.add_argument("--fps", type=int, help="same as --cam fps=...")
    ap.add_argument("--out", help="also save the annotated video here")
    ap.add_argument("--headless", action="store_true", help="no window (e.g. over SSH); use with --out")
    args = ap.parse_args()
    source, settings = source_from_args(args)
    settings.update({k: v for k, v in (("width", args.width), ("height", args.height), ("fps", args.fps)) if v})

    model = get_model()
    try:
        cam = open_camera(source or config.VIDEO_PATH, **settings)
    except (OSError, RuntimeError) as e:
        raise SystemExit(f"Could not open {source or config.VIDEO_PATH}: {e}")
    print(cam.describe())
    frame = cam.read()
    if frame is None:
        raise SystemExit("Could not read a frame")
    fh, fw = frame.shape[:2]
    rx, ry, rw, rh = args.roi or (0, 0, fw, fh)
    rx, ry = max(0, rx), max(0, ry)
    roi = (rx, ry, min(rw, fw - rx), min(rh, fh - ry))
    print(f"Frame {fw}x{fh}, ROI {roi}")

    writer = None
    if args.out:
        src_fps = cam.info.get("fps") or 0
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps if src_fps > 0 else 30, (fw, fh))

    show_nms, paused, fps, frames, t_start = False, False, 0.0, 0, time.perf_counter()
    while frame is not None:
        if not paused:
            t0 = time.perf_counter()
            dets, out, scale = detect_in_roi(model, frame, roi)
            dt = time.perf_counter() - t0
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt  # smoothed

            view = frame.copy()
            draw_detections(view, dets)
            if show_nms:
                draw_engine_nms(view, out, scale, roi)
            cv2.rectangle(view, roi[:2], (roi[0] + roi[2], roi[1] + roi[3]), (0, 200, 255), 2)
            cv2.putText(view, f"{fps:.0f} FPS (detect)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if writer:
                writer.write(view)
            frames += 1

        if not args.headless:
            cv2.imshow("armor detection", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty("armor detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("n"):
                show_nms = not show_nms
        if not paused:
            frame = cam.read()

    print(f"{frames} frames in {time.perf_counter() - t_start:.1f} s")
    cam.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
