"""Label the plates in video frames, so m8_benchmark.py can score every model against YOU.

    python3 label_frames.py                               # demo video, every 30th frame
    python3 label_frames.py --source match.avi --every 15 --out match_labels.json

Labels need frames that come back the same every time, so label a RECORDING, not a live camera.
Record one first:  python3 ../shared/cameras.py --source hik --record ../videos/match.avi

For each plate, click its 4 corners in this order: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
(the outer ends of the two light bars, the same points the models find). After the 4th click the
plate is stored with the current color / number / size shown at the top of the window.

Keys:
    b r g p          color: blue, red, gray, purple
    s 1 2 3 4 5 o h  number: sentry, 1-5, outpost, h = base ("home")
    z                switch small / big plate
    u                undo the last click, or the last plate
    space or n       save this frame and go to the next one. A frame with NO plates is useful too:
                     any plate a model reports there is a false alarm.
    q or Esc         save and quit. Run the same command again to carry on where you stopped.

Label every plate whose 4 corners you can see, including far, blurry and gray ones.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import zoo
from cameras import open_camera
from common import COLOR_BGR, read_frames

COLOR_KEYS = {"b": "blue", "r": "red", "g": "gray", "p": "purple"}
NUMBER_KEYS = {"s": "sentry", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "o": "outpost", "h": "base"}
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


class Labeler:
    def __init__(self, display_scale):
        self.scale = display_scale
        self.color, self.number, self.size = "blue", "3", "small"
        self.plates, self.clicks = [], []

    def click(self, x, y):
        self.clicks.append([x / self.scale, y / self.scale])
        if len(self.clicks) == 4:
            self.plates.append({"corners": self.clicks, "color": self.color, "number": self.number, "size": self.size})
            self.clicks = []

    def undo(self):
        if self.clicks:
            self.clicks.pop()
        elif self.plates:
            self.plates.pop()

    def key(self, ch):
        if ch in COLOR_KEYS:
            self.color = COLOR_KEYS[ch]
        elif ch in NUMBER_KEYS:
            self.number = NUMBER_KEYS[ch]
        elif ch == "z":
            self.size = "big" if self.size == "small" else "small"
        elif ch == "u":
            self.undo()

    def render(self, frame, index, done):
        img = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        for p in self.plates:
            pts = (np.array(p["corners"]) * self.scale).astype(np.int32)
            cv2.polylines(img, [pts], True, COLOR_BGR[p["color"]], 2)
            cv2.putText(img, f"{p['color']} {p['number']} {p['size']}", (int(pts[:, 0].min()), int(pts[:, 1].min()) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BGR[p["color"]], 1)
        for x, y in self.clicks:
            cv2.circle(img, (int(x * self.scale), int(y * self.scale)), 4, (0, 255, 255), -1)
        status = (f"frame {index} ({done} labeled) | next plate: {self.color} {self.number} {self.size} | "
                  f"click the {CORNER_NAMES[len(self.clicks)]} corner")
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, status, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img


def load(path, video):
    if Path(path).exists():
        data = json.loads(Path(path).read_text())
        if data.get("video") and Path(data["video"]).resolve() != Path(video).resolve():
            raise SystemExit(f"{path} has labels for {data['video']}, not {video}. Use another --out.")
        return data
    return {"video": str(video), "frames": {}}


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "--video", dest="video", default=str(zoo.VIDEO_PATH), help="a recorded video")
    ap.add_argument("--every", type=int, default=30, help="label every Nth frame")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="labels.json")
    ap.add_argument("--display-scale", type=float, default=0, help="window size factor (default: fit 1280 px)")
    args = ap.parse_args()

    with open_camera(args.video) as cam:
        if cam.live:
            raise SystemExit(f"{args.video} is a live camera. Record it first: python3 ../shared/cameras.py --source "
                             f"{args.video} --record clip.avi, then label clip.avi")
    data = load(args.out, args.video)
    window = "label plates (see the top of label_frames.py for keys)"
    cv2.namedWindow(window)
    labeler = None
    for index, frame in read_frames(args.video, start=args.start, step=args.every):
        if str(index) in data["frames"]:
            continue
        scale = args.display_scale or min(1.0, 1280 / frame.shape[1])
        state = labeler or Labeler(scale)
        state.plates, state.clicks = [], []
        cv2.setMouseCallback(window, lambda e, x, y, *_: e == cv2.EVENT_LBUTTONDOWN and state.click(x, y))
        while True:
            cv2.imshow(window, state.render(frame, index, len(data["frames"])))
            k = cv2.waitKey(20) & 0xFF
            if k in (ord(" "), ord("n")):
                data["frames"][str(index)] = state.plates
                save(args.out, data)
                break
            if k in (ord("q"), 27):
                if state.plates:
                    data["frames"][str(index)] = state.plates
                save(args.out, data)
                print(f"Saved {len(data['frames'])} labeled frames to {args.out}")
                return
            if k != 255:
                state.key(chr(k))
        labeler = state  # keep the color/number/size for the next frame
    save(args.out, data)
    print(f"End of video. Saved {len(data['frames'])} labeled frames to {args.out}")


if __name__ == "__main__":
    main()
