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

Saves out/compare_frame<N>.jpg (one tile per model) and prints a table. Needs m2-m5 finished,
or --reference.
"""
import sys
import time
from pathlib import Path

if "--reference" in sys.argv:  # must happen before the family modules get imported
    sys.path.insert(0, str(Path(__file__).resolve().parent / "solutions"))

import argparse  # noqa: E402

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


def tile(frame, dets, title, subtitle, box, tile_w=TILE_W):
    x, y, bw, bh = box
    img = draw(frame.copy(), dets)[y:y + bh, x:x + bw]
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
            detect(module, runner, first_frame)  # warm-up: the first run is always slow
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
            dets, t = detect(module, runner, frame)
            ms = {k: v * 1000 for k, v in t.items()}
            hw = f"{runner.inputs[0].shape[2]}x{runner.inputs[0].shape[3]}"
            labels = ", ".join(f"{d['color']} {d['number']} {d['score']:.2f}" for d in dets) or "-"
            print(f"{m.name:22s} {runner.backend:12s} {hw:12s} {ms['pre']:6.1f} {ms['infer']:7.1f} {ms['post']:6.1f}  {labels}")
            results.append((m, dets, f"{runner.backend} {hw}  {sum(ms.values()):.0f} ms  {len(dets)} plate(s)"))
        h, w = frame.shape[:2]
        box = (0, 0, w, h) if args.no_zoom else zoom_box(frame, [dets for _, dets, _ in results])
        tiles = [tile(frame, dets, m.name, sub, box) for m, dets, sub in results]
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
                    dets, t = detect(module, runner, frame)
                    ms = sum(t.values()) * 1000
                    labels = " ".join(f"{d['color'][0]}{d['number']}" for d in dets)
                    h, w = frame.shape[:2]
                    tiles.append(tile(frame, dets, m.name, f"{runner.backend}  {ms:.0f} ms  {labels}", (0, 0, w, h), tile_w))
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
    ap.add_argument("--reference", action="store_true", help="use the solutions/ decoders")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    source, settings = source_from_args(args)
    if args.live:
        live(args, source, settings)
    else:
        compare_frames(args, source, settings)


if __name__ == "__main__":
    main()
