"""m8: which model should we use? Measure speed and quality on real footage.

Speed (milliseconds per frame, median): preprocess + inference + decode. FPS = 1000 / total.

Quality WITHOUT labels (works on any video, no work needed):
  hit %        frames where the model found at least one plate
  plates/frame average number of plates found
  jitter px    how much a plate's SHAPE wobbles from one frame to the next. A plate can move, but
               its 4 corners should move together: jitter is what's left after removing the
               shared motion. Steadier corners -> steadier PnP -> steadier aim.
  dist cm      median frame-to-frame change of the PnP distance, in cm (demo camera calibration)
  These compare models on the SAME footage; they are not absolute scores. A model that finds
  nothing has perfect jitter.

Quality WITH labels (make them with label_frames.py, then pass --labels):
  precision    of the plates it reported, how many were real
  recall       of the real plates, how many it found
  corner px    median distance between its corners and yours
  color %, number %   how often the labels match yours (number only counts models that give one)
  This is the real test. Label 50+ frames of footage that looks like a match.

Run:  python3 m8_benchmark.py                                # every model, 300 frames of the demo
      python3 m8_benchmark.py --labels labels.json           # also score against your labels
      python3 m8_benchmark.py --models rp_0526_trt,tongji_yolo11_trt --frames 600 --csv jetson.csv
      python3 m8_benchmark.py --score 0.5                    # same score threshold for every model
      python3 m8_benchmark.py --reference                    # use the decoders in solutions/
      python3 m8_benchmark.py --source ../videos/ours1.avi --calib ../configs/my_camera.yaml
On the Jetson, run `sudo nvpmodel -m 0 && sudo jetson_clocks` first or the timings wander.

Our own camera: record first (python3 ../shared/cameras.py --source hik --record clip.avi), then benchmark the
recording, so every model sees the same frames. --calib reads that camera's camera_matrix and
distort_coeffs from a camera config file (like ../configs/demo_camera.yaml); without it, distances use the
demo camera's calibration.
"""
import sys
from pathlib import Path

if "--reference" in sys.argv:  # must happen before the family modules get imported
    sys.path.insert(0, str(Path(__file__).resolve().parent / "solutions"))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import zoo  # noqa: E402
from cameras import add_source_args, calibration_from_config, open_camera, source_from_args  # noqa: E402
from common import detect, family_module, is_big, load_runners, match, read_frames  # noqa: E402

K = np.array(zoo.CAMERA_MATRIX, np.float64)  # the demo camera's, until --calib says otherwise
DIST = np.array(zoo.DIST_COEFFS, np.float64)


def plate_points(big):
    """3D corners of a plate (TL, TR, BR, BL), the same as exercise 7."""
    w, h = (zoo.BIG_ARMOR_WIDTH if big else zoo.SMALL_ARMOR_WIDTH), zoo.LIGHTBAR_LENGTH
    return np.array([[0, w / 2, h / 2], [0, -w / 2, h / 2], [0, -w / 2, -h / 2], [0, w / 2, -h / 2]], np.float32)


def distance(det):
    ok, _, tvec = cv2.solvePnP(plate_points(is_big(det)), det["corners"].astype(np.float32), K, DIST,
                               flags=cv2.SOLVEPNP_IPPE)
    return float(np.linalg.norm(tvec)) if ok else None


def shape_jitter(prev, cur):
    """Corner movement left after removing the plate's overall movement, averaged over 4 corners."""
    d = cur["corners"] - prev["corners"]
    return float(np.linalg.norm(d - d.mean(0), axis=1).mean())


def median(values):
    return statistics.median(values) if values else float("nan")


def run_video(runner, module, args, kw):
    ms, plates, jitter, dist_change, prev = [], [], [], [], []
    for _, frame in read_frames(args.src, count=args.frames, **args.settings):
        dets, t = detect(module, runner, frame, **kw)
        ms.append([v * 1000 for v in t.values()])
        plates.append(len(dets))
        for i, j in match(prev, dets, args.match_px):
            jitter.append(shape_jitter(prev[i], dets[j]))
            d0, d1 = distance(prev[i]), distance(dets[j])
            if d0 and d1:
                dist_change.append(abs(d1 - d0) * 100)
        prev = dets
    pre, infer, post = (median([m[k] for m in ms]) for k in range(3))
    return {"pre ms": pre, "infer ms": infer, "post ms": post, "FPS": 1000 / (pre + infer + post),
            "hit %": 100 * float(np.mean([p > 0 for p in plates])), "plates/frame": float(np.mean(plates)),
            "jitter px": median(jitter), "dist cm": median(dist_change)}


def run_labels(runner, module, labels, args, kw):
    wanted = {int(k): v for k, v in labels["frames"].items()}
    tp = fp = fn = color_ok = number_ok = number_seen = 0
    corner_err = []
    for idx, frame in read_frames(labels.get("video") or args.src, count=max(wanted) + 1):
        if idx not in wanted:
            continue
        dets, _ = detect(module, runner, frame, **kw)
        truth = [{**p, "corners": np.array(p["corners"], np.float32), "score": 1.0} for p in wanted[idx]]
        pairs = match(truth, dets, args.match_px)
        tp, fn, fp = tp + len(pairs), fn + len(truth) - len(pairs), fp + len(dets) - len(pairs)
        for i, j in pairs:
            t, d = truth[i], dets[j]
            corner_err.append(float(np.linalg.norm(t["corners"] - d["corners"], axis=1).mean()))
            color_ok += t["color"] == d["color"]
            if d["number"] != "?":
                number_seen += 1
                number_ok += t["number"] == d["number"]
    return {"precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "corner px": median(corner_err),
            "color %": 100 * color_ok / tp if tp else float("nan"),
            "number %": 100 * number_ok / number_seen if number_seen else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated names from zoo.py (default: every model that loads)")
    ap.add_argument("--one-per-network", action="store_true", help="only the first file of each network")
    add_source_args(ap)
    ap.add_argument("--calib", help="camera config file with this camera's camera_matrix + distort_coeffs (like ../configs/demo_camera.yaml)")
    ap.add_argument("--frames", type=int, default=300, help="how many frames to run for speed + stability")
    ap.add_argument("--labels", help="labels.json from label_frames.py")
    ap.add_argument("--score", type=float, help="use this score threshold for every model")
    ap.add_argument("--match-px", type=float, default=30, help="max center distance to call two plates the same")
    ap.add_argument("--csv", help="also save the results here")
    ap.add_argument("--reference", action="store_true", help="use the solutions/ decoders")
    args = ap.parse_args()
    args.src, args.settings = source_from_args(args)
    labels = json.loads(Path(args.labels).read_text()) if args.labels else None
    kw = {"score_thr": args.score} if args.score is not None else {}
    if args.calib:
        global K, DIST
        K, DIST = calibration_from_config(args.calib)
    with open_camera(args.src, **args.settings) as cam:
        if cam.live:
            print(f"NOTE: {cam.describe()} is live, so every model sees DIFFERENT frames: only the speed columns "
                  "compare fairly. Record a clip first (python3 ../shared/cameras.py --record) and benchmark that.")

    results, seen = [], set()
    for m, runner, why in load_runners(zoo.find(args.models)):
        if runner is None:
            print(f"skip {m.name}: {why}")
            continue
        if args.one_per_network and m.network in seen:
            continue
        seen.add(m.network)
        module = family_module(m.family)
        print(f"running {m.name} ({runner.backend}) ...", flush=True)
        try:
            row = {"model": m.name, "runtime": runner.backend, **run_video(runner, module, args, kw)}
            if labels:
                row.update(run_labels(runner, module, labels, args, kw))
        except NotImplementedError:
            raise SystemExit(f"{module.__name__}.py isn't finished yet. Finish it, or run with --reference.")
        results.append(row)

    if not results:
        raise SystemExit("No model loaded.")
    cols = [c for c in results[0] if c not in ("model", "runtime")]
    print(f"\n{'model':24s} {'runtime':12s}" + "".join(f"{c:>13s}" for c in cols))
    for r in sorted(results, key=lambda r: -r["FPS"]):
        print(f"{r['model']:24s} {r['runtime']:12s}" + "".join(f"{r[c]:13.2f}" for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved {args.csv}")
    print("\nFPS here is THIS machine. The robot's camera runs at its own FPS: a model is fast enough on the "
          "Jetson when its FPS there beats the camera's, with room left for tracking and aiming.")


if __name__ == "__main__":
    main()
