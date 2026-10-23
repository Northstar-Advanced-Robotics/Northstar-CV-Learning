python3 m7_compare.py --live --reference --source usb:0 --cam width=1920 --cam height=1200 --models tup_0527_trt,rp_0526_trt --roi 0,0,640,640


# m7_compare.py: run models side by side

`m7_compare.py` runs several detection models on the same picture and shows their results next to each other, one
tile per model. Use it to *see* how models differ: which plates each one finds, where it puts the corners, what it
calls them, and how long it takes. For numbers over hundreds of frames (hit rate, jitter, precision/recall), use
`m8_benchmark.py` instead.

It has two modes:

| mode | what it does | output |
|---|---|---|
| **frames** (default) | runs every model on a few chosen frames of a video | a printed table + `out/compare_frame<N>.jpg` per frame |
| **live** (`--live`) | runs every model on each new frame from a camera (or video) and shows a window | the window, optionally a recording |
python3 m7_compare.py --live --reference --source usb:0 --cam width=1920 --cam height=1200 --models tup_0527_trt,rp_0526_trt --roi 0,0,640,640
All commands below are run from the `model_zoo/` folder.

---

## Quick start

```bash
cd model_zoo

# Two frames of the demo video, every model that loads on this machine
python3 m7_compare.py --reference

# Just the models you care about
python3 m7_compare.py --reference --models tup_0527_trt,rp_0526_trt,at_c2psa_576x768_trt

# Live on the robot's HikRobot camera
python3 m7_compare.py --reference --live --source hik --models at_c2psa_576x768_trt

# Live over SSH (no screen): record 30 s of the side-by-side view
python3 m7_compare.py --reference --live --source hik --headless --save live.avi --seconds 30 --models rp_0526_trt
```

Drop `--reference` once you've finished m2-m5 yourself: it swaps in the finished decoders from `solutions/`.
Without it, m7 uses *your* decoders and stops with "isn't finished yet" if one isn't.

---

## Every option

### Picking models

| option | what it does |
|---|---|
| `--models A,B,C` | Comma-separated names from `zoo.py` (no spaces). Default: every model whose file exists **and** loads here. Names are the file name without its extension, plus `_trt` for engines you built (`rp_0526_trt`). A name that isn't in the zoo stops the script and lists the valid ones. |
| `--one-per-network` | Keep only the first file of each trained network. The zoo holds the same weights in several formats (ONNX, OpenVINO, NCNN, TensorRT); this shows each network once. Which file "wins" depends on the order in `zoo.py`, so use `--models` when you care which format runs. |
| `--reference` | Use the decoders in `solutions/` instead of your m2-m5. |

Models that can't load (wrong TensorRT version, missing runtime, TensorRT-only nodes) are printed as `skip <name>: <why>`
and left out. The rest still run.

### Picking frames (frames mode)

| option | what it does |
|---|---|
| `--frames 60,400` | Which frame numbers to compare, comma-separated. Default `60,400`. Each one becomes `out/compare_frame<N>.jpg`. |
| `--no-zoom` | Show the whole frame in every tile. By default each tile zooms to the area around every plate any model found (with a margin), so small plates stay readable. If no model found anything, it shows the whole frame anyway. |

On a live camera, "frame 60" means "skip 60 frames first", which also lets auto exposure settle. For repeatable
comparisons, record a clip and use that.

### Picking the source (both modes)

| option | what it does |
|---|---|
| `--source S` | Where frames come from. Default: `videos/demo.avi` (or `$VIDEO`). |
| `--cam KEY=VALUE` | A camera setting. Repeat it for more than one: `--cam exposure_ms=2 --cam gain=12`. |

Sources (the same strings work in every script, see `shared/cameras.py`):

| `--source` | what it opens |
|---|---|
| `clip.avi`, `match.mp4` | a video file |
| `shots/`, `"shots/*.png"` | a folder or pattern of images |
| `usb:0`, `0`, `/dev/video2` | a USB webcam |
| `hik`, `hik:1` | HikRobot industrial camera (the robot's camera, needs the MVS SDK) |
| `mv`, `mv:0` | MindVision industrial camera |
| `csi:0` | Jetson CSI ribbon camera (needs OpenCV with GStreamer) |
| `gst:<pipeline>` | any GStreamer pipeline ending in `appsink` |
| `rtsp://...`, `http://...` | a network stream |
| `config:<file.yaml>` | the camera a robot config file describes, with its settings |

Camera settings per source:

| source | `--cam` keys |
|---|---|
| `usb` | `width`, `height`, `fps`, `exposure`, `gain`, `gamma`, `fourcc` (the driver's own units) |
| `hik` | `exposure_ms`, `gain`, `fps` (defaults: 2 ms, gain 16, like the robot) |
| `mv` | `exposure_ms`, `gamma` |
| `csi` | `width`, `height`, `fps` |

A camera can ignore a setting. The first line m7 prints in live mode is what the camera actually does.
To see which cameras this machine can find: `python3 ../shared/cameras.py --list`.

### Region of interest

| option | what it does |
|---|---|
| `--roi X,Y,W,H` | Only detect inside this rectangle, in full-frame pixels: left edge, top edge, width, height. `-1` for W or H means "to the edge of the frame" (no crop in that direction). |
| `--roi-config FILE` | Take the ROI from a robot config file (`use_roi` and the `roi:` block). If `use_roi` isn't `true`, no ROI is used. |

Details in [Region of interest](#region-of-interest-roi) below.

### Live mode

| option | what it does |
|---|---|
| `--live` | Run on every new frame and show a window. Without it, m7 runs in frames mode. |
| `--save FILE.avi` | Also record the side-by-side view (MJPG, written at 10 FPS). |
| `--headless` | No window. Prints the overall FPS every 20 frames. Use with `--save` over SSH. |
| `--seconds N` | Stop after N seconds. Without it, live mode runs until you press `q` (or the video ends). |

Keys in the live window:

| key | action |
|---|---|
| `q` or `Esc` | quit |
| `space` | pause / resume (freezes the view on the current frame) |
| `s` | save a snapshot to `out/live_<date>_<time>.jpg` |

`--frames` and `--no-zoom` do nothing in live mode: live tiles always show the whole frame.

---

## Reading the output

### Frames mode table

Example (numbers from a Jetson):

```
model                  runtime      input           pre   infer   post  plates
at_c2psa_576x768_trt   tensorrt     576x768         6.5    17.0    0.4  blue sentry 0.90
rp_0708_trt            tensorrt     640x640         6.5    12.9    1.7  blue sentry 0.93
```

| column | meaning |
|---|---|
| `runtime` | which engine ran the file: onnxruntime, openvino, ncnn or tensorrt |
| `input` | the model's input size, height x width |
| `pre` | ms to letterbox the frame and turn it into the model's input tensor (CPU, numpy) |
| `infer` | ms in the runtime, including the copies to and from the GPU for TensorRT |
| `post` | ms to decode the raw output into plates (Python + NMS) |
| `plates` | what it found: color, number, score. `-` = nothing |

Each tile in the saved picture shows the same: model name, runtime, input size, total ms and number of plates. Corners
are numbered 0-3 and should go top-left, top-right, bottom-right, bottom-left. If they don't, the decoder's corner order
is wrong.

### Live mode tiles

Each tile shows the runtime, the total ms for that model on that frame, and a short label per plate (`b1` = blue 1,
`rsentry` = red sentry).

**The models run one after another on each frame,** so the window's frame rate is the *sum* of all their times. Three
models at 15 ms each show ~20 FPS, even though each alone would run at ~65. Compare the ms in each tile, not how
smooth the window feels. To judge one model's real speed, run it alone (`--models` with one name).

### Timing tips

- On the Jetson, lock the clocks first, or TensorRT times can be 3x too slow:
  `sudo nvpmodel -m 0 && sudo jetson_clocks`
- The first run of every model is slow, so m7 does one warm-up run before timing anything.
- These are Python timings: preprocessing and decoding on the CPU. The robot's C++ does both on the GPU and is much
  faster. Use m7 to compare models with each other, not to predict the robot's FPS.

---

## Region of interest (ROI)

An ROI makes the models look at only part of the frame, the same way the robot's `use_roi` does. Typical uses: ignore
the floor and ceiling, ignore the robot's own barrel at the edge of the picture, or test how a model does on a "zoomed
in" view.

```bash
# Middle of a 1440x1080 frame
python3 m7_compare.py --reference --roi 320,240,800,600

# Full width, only rows 200 to 800 (cut off top and bottom)
python3 m7_compare.py --reference --roi 0,200,-1,600

# Whatever the robot's config file says
python3 m7_compare.py --reference --live --source hik --roi-config ../configs/my_robot.yaml
```

The config file uses the robot's format:

```yaml
use_roi: true
roi:
  x: 320
  y: 240
  width: 800     # -1 = to the right edge
  height: 600    # -1 = to the bottom edge
```

What happens:

1. The frame is cropped to the ROI. A rectangle that runs off the frame is clipped to it.
2. Only the crop goes through the model (letterboxed to the model's input like any frame).
3. The detected corners are shifted back by (x, y), so they're in full-frame pixels again (the robot's `offset_`).
   PnP and anything else downstream works unchanged.
4. The tiles show the full frame with the ROI drawn as a yellow rectangle. Plates outside it are simply not detected.

### An ROI changes the plate size the model sees

The model input has a fixed size (640x640, 576x768, ...). A smaller crop gets scaled **up** to fill it, so every plate
looks bigger to the model than it does in a full frame. Example on frame 60 of the demo video (blue sentry):

| model | full frame | `--roi 300,100,1000,800` | `--roi 400,200,800,600` |
|---|---|---|---|
| rp_0708 (640x640) | score 0.93 | 0.75 | 0.74 |
| at_c2psa_576x768 | 0.90 | 0.86 | **not found** |

Same plate, same place (the corners land within a few pixels), but the more the crop is magnified, the less the
models trust it, until one stops seeing it. They were trained on full frames, where plates are smaller.

So:

- **Don't make the ROI much smaller than the model's input.** An ROI about the size of the input (e.g. 640x640 for the
  yolov5 family, 768x576 for `at_c2psa_576x768`) runs at scale 1: no resizing at all, so far plates keep every pixel.
- **Smaller ROIs help far plates and hurt near ones.** A big close plate magnified past the edge of the crop can't be
  detected at all.
- **Check an ROI on recorded match footage** with `m8_benchmark.py` before using it on the robot. m8 has no `--roi`;
  crop the clip instead, or compare with m7 frame by frame.

---

## Examples

```bash
# What does every network think of frame 600, once each?
python3 m7_compare.py --reference --one-per-network --frames 600

# Compare formats of the SAME weights: are the conversions faithful?
python3 m7_compare.py --reference --models tup_1208,tup_1208_pnnx,tup_1208_onnx2ncnn,tup_1208_trt --frames 60,200,400

# INT8 vs FP32 of Tongji's YOLO11: how far do the corners move?
python3 m7_compare.py --reference --models tongji_yolo11,tongji_yolo11_int8 --frames 60,400 --no-zoom

# Live on a USB webcam at 1280x720
python3 m7_compare.py --reference --live --source usb:0 --cam width=1280 --cam height=720 --models rp_0526_trt

# Live on the robot camera with the robot's exposure and ROI, recorded for later
python3 m7_compare.py --reference --live --source config:../configs/demo_camera.yaml \
    --roi-config ../configs/my_robot.yaml --models at_c2psa_576x768_trt --save roi_test.avi --seconds 60

# Run on a recorded clip as if it were live (handy to watch a whole match)
python3 m7_compare.py --reference --live --source ../videos/match1.avi --models tup_0527_trt,at_c2psa_576x768_trt
```

---

## Troubleshooting

| message | cause | fix |
|---|---|---|
| `m4_ultralytics.py isn't finished yet` | your decoder for that family still raises `NotImplementedError` | finish it, or add `--reference` |
| `Unknown model(s) [...]` | typo, or an engine you haven't built | check the list it prints; build engines with `python3 jetson/build_engines.py` |
| `skip X: TensorRT 10.3.0 can't load ...` | engine built on another device or TensorRT version | rebuild it on this machine with `jetson/build_engines.py` |
| `skip X: ... EfficientNMS_TRT ...` | TensorRT-only file run on ONNX Runtime | use its `_trt` engine |
| `No model loaded.` | nothing in `--models` exists or loads here | run without `--models` to see what does |
| `the camera gave no frame` | camera not found, busy, or SDK missing | `python3 ../shared/cameras.py --list`, close other programs using it |
| `cannot open display` / Qt errors over SSH | no screen | add `--headless --save out.avi` |
| `--roi wants x,y,width,height` | not exactly 4 integers | `--roi 320,240,800,600`, no spaces |
| window is slow | models run one after another | fewer `--models`; read the ms per tile |