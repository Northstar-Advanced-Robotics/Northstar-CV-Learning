"""m7: every model, side by side, on the same frames, or live on a camera.

Run:  python3 m7_compare.py                          # frames 60 and 400 of the demo video
      python3 m7_compare.py --frames 60,200,400,600
      python3 m7_compare.py --models rp_0526,tongji_yolo11,at_c2psa_576x768
      python3 m7_compare.py --one-per-network        # skip duplicate files of the same weights
      python3 m7_compare.py --reference              # use the decoders in solutions/

Live, on any camera (see cameras.py for every --source):
      python3 m7_compare.py --live --source usb:0 --models rp_0526,at_c2psa_576x768
      python3 m7_compare.py --live --source hik --cam exposure_ms=2 --one-per-network
      python3 m7_compare.py --live --source csi:0 --headless --save live.avi --seconds 30    # over SSH
      keys: q quit, space pause, s snapshot

Region of interest (only look at part of the frame, like the robot's use_roi):
      python3 m7_compare.py --roi 320,240,800,600            # x,y,width,height in frame pixels
      python3 m7_compare.py --roi 0,200,-1,600               # -1 = don't crop that dimension
      python3 m7_compare.py --live --source hik --roi-config ../configs/my_robot.yaml   # its use_roi + roi

Saves out/compare_frame<N>.jpg (one tile per model) and prints a table. Needs m2-m5 finished,
or --reference.
"""
import sys
import time
from pathlib import Path

if "--reference" in sys.argv:  # must happen before the family modules get imported
    sys.path.insert(0, str(Path(__file__).resolve().parent / "solutions"))

import argparse  # noqa: E402
import re  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import zoo  # noqa: E402
from cameras import add_source_args, open_camera, source_from_args  # noqa: E402
from common import OUT_DIR, detect, draw, family_module, load_runners, read_frames  # noqa: E402

TILE_W = 480


def zoom_box(frame, all_dets, aspect=0.75):
    """The part of the frame around every plate any model found (with a margin), so small plates
    stay readable in the tiles. Keeps the tile aspect ratio; the whole frame if nothing was found."""
    corners = [d["corners"] for dets in all_dets for d in dets]
    h, w = frame.shape[:2]
    if not corners:
        return 0, 0, w, h
    pts = np.concatenate(corners)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    bw = max((x1 - x0) * 2.5, (y1 - y0) * 2.5 / aspect, 360)
    bw = min(bw, w, h / aspect)
    bh = bw * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(np.clip(cx - bw / 2, 0, w - bw)), int(np.clip(cy - bh / 2, 0, h - bh))
    return left, top, int(bw), int(bh)


def parse_roi(args):
    """(x, y, width, height) from --roi or --roi-config, or None. Width/height -1 = no crop that way."""
    if args.roi:
        try:
            roi = [int(v) for v in args.roi.split(",")]
        except ValueError:
            roi = []
        if len(roi) != 4:
            raise SystemExit(f"--roi wants x,y,width,height (4 integers), got {args.roi!r}")
        return tuple(roi)
    if args.roi_config:
        use, r = roi_from_config(args.roi_config)
        if not use:
            print(f"{args.roi_config}: use_roi is not true, so no ROI")
            return None
        return tuple(r.get(k, d) for k, d in (("x", 0), ("y", 0), ("width", -1), ("height", -1)))
    return None


def roi_from_config(path):
    """(use_roi, {x, y, width, height}) from a robot config. The robot writes it as
        use_roi: true
        roi:
          x: 0
          ...
    a nested block, which cameras.read_config skips, so it's read here line by line."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise SystemExit(f"config {path} doesn't exist")
    use, roi, in_roi = False, {}, False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#")[0].rstrip()
        if not text.strip():
            continue
        top = re.match(r"^(\w+):\s*(.*)$", text)
        if top:
            in_roi = top.group(1) == "roi"
            if top.group(1) == "use_roi":
                use = top.group(2).strip().lower() == "true"
            continue
        sub = re.match(r"^\s+(x|y|width|height):\s*(-?\d+)", text)
        if in_roi and sub:
            roi[sub.group(1)] = int(sub.group(2))
    return use, roi


def roi_rect(roi, frame):
    """The ROI as (x, y, w, h), clamped to this frame. Same rule as sp_vision: -1 = the full size."""
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    rw = w - x if rw == -1 else min(rw, w - x)
    rh = h - y if rh == -1 else min(rh, h - y)
    return x, y, rw, rh


def detect_roi(module, runner, frame, roi):
    """detect() on the ROI only, corners moved back to full-frame pixels (sp_vision's offset_)."""
    if roi is None:
        return detect(module, runner, frame)
    x, y, w, h = roi_rect(roi, frame)
    dets, t = detect(module, runner, np.ascontiguousarray(frame[y:y + h, x:x + w]))
    for d in dets:
        d["corners"] = d["corners"] + np.array([x, y], np.float32)
    return dets, t


def draw_roi(img, roi):
    if roi is not None:
        x, y, w, h = roi_rect(roi, img)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2)
    return img


def tile(frame, dets, title, subtitle, box, tile_w=TILE_W, roi=None):
    x, y, bw, bh = box
    img = draw(draw_roi(frame.copy(), roi), dets)[y:y + bh, x:x + bw]
    img = cv2.resize(img, (tile_w, int(tile_w * bh / bw)))
    cv2.rectangle(img, (0, 0), (tile_w, 52), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, subtitle, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def grid(tiles, cols=4):
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def load(args, first_frame):
    """Every model that loads here, warmed up on first_frame: [(model, runner, family module)]."""
    runners, seen = [], set()
    for m, runner, why in load_runners(zoo.find(args.models)):
        if runner is None:
            print(f"skip {m.name}: {why}")
            continue
        if args.one_per_network and m.network in seen:
            continue
        seen.add(m.network)
        module = family_module(m.family)
        try:
            detect_roi(module, runner, first_frame, args.roi)  # warm-up: the first run is always slow
        except NotImplementedError:
            raise SystemExit(f"{module.__name__}.py isn't finished yet. Finish it, or run with --reference.")
        runners.append((m, runner, module))
    if not runners:
        raise SystemExit("No model loaded.")
    return runners


def compare_frames(args, source, settings):
    wanted = sorted(int(f) for f in args.frames.split(","))
    frames = {i: f for i, f in read_frames(source, count=max(wanted) + 1, **settings) if i in wanted}
    runners = load(args, frames[wanted[0]])
    for idx in wanted:
        frame, results = frames[idx], []
        print(f"\n=== frame {idx} ===")
        print(f"{'model':22s} {'runtime':12s} {'input':12s} {'pre':>6s} {'infer':>7s} {'post':>6s}  plates")
        for m, runner, module in runners:
            dets, t = detect_roi(module, runner, frame, args.roi)
            ms = {k: v * 1000 for k, v in t.items()}
            hw = f"{runner.inputs[0].shape[2]}x{runner.inputs[0].shape[3]}"
            labels = ", ".join(f"{d['color']} {d['number']} {d['score']:.2f}" for d in dets) or "-"
            print(f"{m.name:22s} {runner.backend:12s} {hw:12s} {ms['pre']:6.1f} {ms['infer']:7.1f} {ms['post']:6.1f}  {labels}")
            results.append((m, dets, f"{runner.backend} {hw}  {sum(ms.values()):.0f} ms  {len(dets)} plate(s)"))
        h, w = frame.shape[:2]
        box = (0, 0, w, h) if args.no_zoom else zoom_box(frame, [dets for _, dets, _ in results])
        tiles = [tile(frame, dets, m.name, sub, box, roi=args.roi) for m, dets, sub in results]
        path = OUT_DIR / f"compare_frame{idx}.jpg"
        cv2.imwrite(str(path), grid(tiles))
        print(f"saved {path.relative_to(zoo.HERE)}")
    print("\n(times are milliseconds on THIS machine. Run it on the Jetson for numbers that matter.)")


def live(args, source, settings):
    """Run every chosen model on each new camera frame and show them side by side. The models run one
    after another, so the window's frame rate is the SUM of their times: compare the ms per tile."""
    with open_camera(source, **settings) as cam:
        print(cam.describe())
        frame = cam.read()
        if frame is None:
            raise SystemExit("the camera gave no frame")
        runners = load(args, frame)
        cols = min(len(runners), 3)
        tile_w = max(360, 1440 // cols)
        writer, paused, shown, t0 = None, False, 0, time.perf_counter()
        while args.seconds is None or time.perf_counter() - t0 < args.seconds:
            if not paused:
                frame = cam.read()
                if frame is None:
                    break
                tiles = []
                for m, runner, module in runners:
                    dets, t = detect_roi(module, runner, frame, args.roi)
                    ms = sum(t.values()) * 1000
                    labels = " ".join(f"{d['color'][0]}{d['number']}" for d in dets)
                    h, w = frame.shape[:2]
                    tiles.append(tile(frame, dets, m.name, f"{runner.backend}  {ms:.0f} ms  {labels}", (0, 0, w, h), tile_w, args.roi))
                view = grid(tiles, cols)
                shown += 1
                if args.save:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                                 (view.shape[1], view.shape[0]))
                    writer.write(view)
            if args.headless:
                if shown % 20 == 0:
                    print(f"{shown} frames, {shown / (time.perf_counter() - t0):.1f} FPS for all models together", flush=True)
                continue
            cv2.imshow("models, live", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s"):
                path = OUT_DIR / f"live_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), view)
                print(f"saved {path.relative_to(zoo.HERE)}")
        if writer:
            writer.release()
            print(f"saved {args.save}")
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="60,400", help="comma-separated frame numbers")
    ap.add_argument("--models", help="comma-separated names from zoo.py (default: every model that loads)")
    ap.add_argument("--one-per-network", action="store_true", help="only the first file of each network")
    add_source_args(ap)
    ap.add_argument("--no-zoom", action="store_true", help="show the whole frame in every tile")
    ap.add_argument("--live", action="store_true", help="run on every new frame and show a window")
    ap.add_argument("--save", help="--live: also record the side-by-side view to this .avi")
    ap.add_argument("--headless", action="store_true", help="--live without a window (over SSH): use with --save")
    ap.add_argument("--seconds", type=float, help="--live: stop after this long")
    ap.add_argument("--roi", help="only detect inside x,y,width,height (frame pixels; -1 = full width/height)")
    ap.add_argument("--roi-config", help="take the ROI from a robot config file (its use_roi and roi: x, y, width, height)")
    ap.add_argument("--reference", action="store_true", help="use the solutions/ decoders")
    args = ap.parse_args()
    args.roi = parse_roi(args)
    OUT_DIR.mkdir(exist_ok=True)
    source, settings = source_from_args(args)
    if args.live:
        live(args, source, settings)
    else:
        compare_frames(args, source, settings)


if __name__ == "__main__":
    main()