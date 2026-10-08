"""Exercise 7: use the four corners to measure how far away each plate is (solvePnP).

We know how big a plate really is. How big and how tilted its 4 corners look in the picture tells
you where it is in 3D. OpenCV's solvePnP does that math (BASICS.md, section 9).
Where to look: reference/sp_vision_25/tasks/auto_aim/solver.cpp (plate sizes, 3D corners, solvePnP call).

Run:
    python3 ex7_pnp.py                     # demo video (its camera calibration is in config.py)
    python3 ex7_pnp.py --source usb:0 --calib ../configs/my_camera.yaml   # your camera: calibrate it first (README)
"""
import argparse

import cv2
import numpy as np

import checks
import config
from backends import get_model
from ex5_decode import draw_detections
from cameras import add_source_args, calibration_from_config, open_camera, source_from_args
from ex6_live import detect_in_roi


def armor_object_points(big):
    """Return the plate's 4 corners in 3D, in meters, as a 4x3 float32 array, in the SAME
    order as the image corners (TL, TR, BR, BL).

    Measure from the plate's center (like the robot's solver.cpp does): x points out of the
    plate, y to the plate's left, z up. The corners are the ends of the two light bars, so the plate
    is config.SMALL_ARMOR_WIDTH or config.BIG_ARMOR_WIDTH wide and config.LIGHTBAR_LENGTH tall.
    Each corner is [x, y, z]. The top-left one is [0, +width / 2, +height / 2].
    """
    # TODO
    raise NotImplementedError


def solve_pnp(corners, big, camera_matrix, dist_coeffs):
    """corners: the 4 corners in the picture (TL, TR, BR, BL). Return (rvec, tvec): which way the
    plate faces, and where it is (in meters, from the camera). Use cv2.solvePnP. It has several
    methods ("flags"): which one suits 4 points on a flat plate? (See what solver.cpp uses, then
    look it up in the OpenCV docs.)
    """
    # TODO
    raise NotImplementedError


def is_big(det):
    # Simplification: the base uses big plates. The real code also handles big 3/4/5 plates.
    return det["number"] == "base"


if __name__ == "__main__":
    if not checks.check_ex7(armor_object_points, solve_pnp):
        raise SystemExit
    ap = argparse.ArgumentParser()
    add_source_args(ap)  # --source (a video, usb:0, hik, csi:0, ...) and --cam KEY=VALUE: see shared/cameras.py
    ap.add_argument("--calib", help="a camera config with YOUR camera's camera_matrix and distort_coeffs (like configs/demo_camera.yaml)")
    args = ap.parse_args()
    source, settings = source_from_args(args)

    if args.calib:
        K, dist = calibration_from_config(args.calib)
    else:  # the camera that recorded the demo video
        K = np.array(config.CAMERA_MATRIX, np.float64)
        dist = np.array(config.DIST_COEFFS, np.float64)
    model = get_model()
    cam = open_camera(source or config.VIDEO_PATH, **settings)
    while True:
        frame = cam.read()
        if frame is None:
            break
        try:
            dets, _, _ = detect_in_roi(model, frame, (0, 0, frame.shape[1], frame.shape[0]))
        except NotImplementedError:
            raise SystemExit("Finish exercise 6 (detect_in_roi) first.")
        draw_detections(frame, dets)
        for d in dets:
            rvec, tvec = solve_pnp(d["corners"], is_big(d), K, dist)
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.05)
            dist_m = float(np.linalg.norm(tvec))
            x, y = d["corners"][2].astype(int)
            cv2.putText(frame, f"{dist_m:.2f} m", (int(x) + 5, int(y) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("armor distance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()
