"""m6: one network, several files. Do they really give the same answers?

Look at the `network` column in zoo.py: some entries hold the SAME trained weights in different
packaging. Converting can quietly change a model: precision drops (FP16, INT8), a converter
mishandles an operation, a runtime rounds differently. The only way to know is to run both files
on the same frames and compare what comes out. This script does that for every network that has
two or more files that load on this machine.

Run:  python3 m6_same_network.py                 # 30 frames of the demo video
      python3 m6_same_network.py --frames 200 --every 2
      python3 m6_same_network.py --source usb:0     # or any camera (see cameras.py)
"""
import argparse
from collections import defaultdict

import numpy as np

import checks
import zoo
from cameras import add_source_args, source_from_args
from common import detect, family_module, load_runners, match, read_frames


def compare(dets_a, dets_b, max_px=20.0):
    """Compare two files' detections on ONE frame. Returns
        {"matched": how many plates both found,
         "only_a": plates only A found, "only_b": plates only B found,
         "corner_px": the largest distance between matching corners of a matched pair (None if none),
         "score_diff": the largest |score A - score B| over matched pairs (None if none),
         "labels_agree": True if every matched pair has the same color AND number}
    """
    # common.match(dets_a, dets_b, max_px) pairs up plates whose centers are close and returns
    # (i, j) index pairs. Corners are 4x2 arrays in the same order (TL, TR, BR, BL), so the
    # corner distance is np.linalg.norm(a - b, axis=1) and you want its max.
    # TODO
    raise NotImplementedError


def verdict(s):
    if s["frames_with_plates"] == 0:
        return "neither found a plate: try more frames"
    if s["only_a"] + s["only_b"] == 0 and s["label_mismatches"] == 0:
        if s["corner_px"] < 0.5:
            return "same answers (differences are rounding noise)"
        return f"same plates and labels, corners move up to {s['corner_px']:.1f} px (lost precision)"
    return "DIFFERENT: the conversion changed what the model finds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--every", type=int, default=5, help="use every Nth frame")
    add_source_args(ap)
    args = ap.parse_args()
    source, settings = source_from_args(args)

    if not checks.check_m6(compare):
        raise SystemExit
    groups = defaultdict(list)
    for m, runner, why in load_runners(zoo.find()):
        if runner is None:
            print(f"skip {m.name}: {why}")
        else:
            groups[m.network].append((m, runner))
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        raise SystemExit("No network has two files that load here.")

    # Grab the frames ONCE, so every file sees exactly the same pictures (matters for live cameras)
    frames = [f for _, f in read_frames(source, step=args.every, count=args.frames, **settings)]
    print(f"\nComparing on {len(frames)} frames (every {args.every}th).")
    rows = []
    for network, members in groups.items():
        (ref, ref_runner), others = members[0], members[1:]
        module = family_module(ref.family)
        ref_dets = [detect(module, ref_runner, f)[0] for f in frames]
        for m, runner in others:
            s = {"frames": len(frames), "frames_with_plates": 0, "only_a": 0, "only_b": 0, "corner_px": 0.0,
                 "score_diff": 0.0, "label_mismatches": 0}
            for frame, a in zip(frames, ref_dets):
                b = detect(module, runner, frame)[0]
                c = compare(a, b)
                s["frames_with_plates"] += bool(a or b)
                s["only_a"] += c["only_a"]
                s["only_b"] += c["only_b"]
                s["label_mismatches"] += not c["labels_agree"]
                if c["matched"]:
                    s["corner_px"] = max(s["corner_px"], c["corner_px"])
                    s["score_diff"] = max(s["score_diff"], c["score_diff"])
            rows.append((network, ref, m, s))

    print(f"\n{'network':17s} {'A (reference)':22s} {'B':22s} {'only A':>6s} {'only B':>6s} {'corner px':>9s} "
          f"{'score diff':>10s} {'label diff':>10s}  verdict")
    for network, a, b, s in rows:
        print(f"{network:17s} {a.name:22s} {b.name:22s} {s['only_a']:6d} {s['only_b']:6d} {s['corner_px']:9.2f} "
              f"{s['score_diff']:10.3f} {s['label_mismatches']:10d}  {verdict(s)}")
    print("\n(corner px and score diff are the WORST over all matched plates; only A / only B are totals)")


if __name__ == "__main__":
    main()
