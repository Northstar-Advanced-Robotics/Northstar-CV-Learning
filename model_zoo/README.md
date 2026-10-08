# Model zoo: every armor detector we could find, and which one we should use

The main track (exercises 1-7) took one model apart. This track gives you **8 armor-detecting models from 5
RoboMaster teams, saved in 4 different file formats: 16 files in all**. They all do the same job: find each armor
plate in a picture and give back its 4 corners. But they disagree about almost everything else:

- **what they want as input:** 640×640 or 416×416 or 576×768 pixels, RGB or BGR color order, values 0..1 or 0..255,
  float32 or float16 numbers, the picture placed top-left or centered, padded with black or gray
- **what they give back:** 25200 guesses, or 3549, or 30. Corners in pixels, or in "grid cells". Scores that are
  logits or probabilities. One row per guess or one per finished plate. With or without duplicates removed
- **how they're stored:** ONNX, OpenVINO, NCNN or TensorRT files, and what each format does and doesn't record

You'll learn to find those differences in the files and in each team's code, get every model running, check whether
converted copies really match their originals, and then **measure all of them and recommend one for our robot**.

**Before you start:** finish exercises 3-5 of the main track, and keep [BASICS.md](../BASICS.md) open: every term
here is explained there. [MODELS.md](MODELS.md) is the reference: where every file came from, exactly what each one
expects and returns, and what we found while checking. Read it when the hints run out.

```
m1  read what each file says about itself (no running)
 │
m2  m3  m4  m5   one decoder per "family" of models: get every model in it running
 │
m6  the same model in different files: do they really give the same answers?
 │
m7  every model side by side, on pictures or live on a camera
 │
m8  measure them all, on the Jetson and against your own labels, and pick the best
```

## Setup

**1. Everything is in this repo:** the models in `../models/<team>/` and the demo video in `../videos/demo.avi`.

**2. Python packages.**

On a **laptop** (Windows, Mac or Linux), everything except TensorRT:
```
pip install numpy opencv-python onnx onnxruntime openvino ncnn
```
Those are the 4 programs that run model files ([BASICS 10](../BASICS.md#10-model-files-and-the-programs-that-run-them)),
plus numpy and OpenCV. On Windows, type `python` where this page says `python3`.

On the **Jetson Orin Nano**:
- TensorRT and PyTorch as in the main track's setup.
- `pip install onnx onnxruntime openvino ncnn`. On the Jetson these run on its processor (CPU); only TensorRT uses
  its GPU.
- If PyTorch complains about "NumPy 2", run `pip install "numpy<2"`.

**3. Work in this folder:** `cd model_zoo`, then check that everything loads: `python3 m1_io_cards.py`. It stops at
its own TODOs: that's expected.

## What's in this folder

| file | what it is |
|---|---|
| `zoo.py` | the list of every model file: where it is, which family it belongs to, which team made it. **Start here.** |
| `runtimes.py` | opens any model file, whatever its format, and runs it the same way |
| `common.py` | shared helpers: letterbox, NMS, drawing, timing |
| `m1_io_cards.py` ... `m6_same_network.py` | **yours**: functions marked TODO, with comments saying what to write and why |
| `m7_compare.py`, `m8_benchmark.py` | finished tools: side-by-side pictures, and the scoreboard |
| `label_frames.py` | click plate corners on video frames to make the right answers for m8 |
| `jetson/build_engines.py` | turns every ONNX model into a TensorRT engine, on the Jetson |
| `solutions/` | working versions of m1-m6. Each runs on its own too: `python3 solutions/m2_yolov5.py` |
| `tools/` | for CV leads: `make_exercises.py` rebuilds m1-m6 from the solutions; `export_reborn_yolov8.py` |
| `out/` | where every script saves its pictures |
| `../shared/cameras.py` | reads frames from any camera or video (shared with the main track) |

## How the exercises work

Like the main track: each TODO says what to write and why, and running the file checks your functions automatically.

A **family** is a group of models that answer in the same layout, so one decoder works for all of them. The family
scripts (m2-m5) run in two parts:

- **Part A** checks your `preprocess()` (and any helpers), runs every model of the family that works on your computer,
  and prints its **raw** answer. It also saves `out/<model>_raw_points.jpg`, with the best guess's 4 points numbered in
  the order the model gave them. Use it to work out the layout before you decode anything.
- **Part B** checks your `decode()`, decodes the real answer, and saves `out/<model>_decoded.jpg`. Corners are drawn
  numbered 0-3: they must go top-left, top-right, bottom-right, bottom-left.

Files that can't run on your computer (TensorRT engines on a laptop, the file with a TensorRT-only NMS step) are
skipped, with the reason printed. Reading those reasons is part of the lesson.

Write your answers to the questions somewhere and talk them over with a CV lead.

## Using a camera instead of the demo video

Every script takes `--source`, which tells it where to get pictures from. The default is `../videos/demo.avi`.

| `--source` | what it opens |
|---|---|
| `../videos/match1.avi` | a video file |
| `shots/` or `"shots/*.png"` | a folder of pictures |
| `usb:0` (or `0`, or `/dev/video2` on Linux) | a USB webcam: 0 is the first one, 1 the second... |
| `hik` | the robot's camera, a HikRobot industrial camera (needs HikRobot's software, see below) |
| `mv` | a MindVision industrial camera (needs MindVision's software) |
| `csi:0` | a Jetson ribbon-cable camera |
| `config:../configs/usb_camera.yaml` | the camera described in a config file, with its settings |
| `rtsp://...` | a network video stream |

Camera settings go in `--cam NAME=VALUE`, repeated as needed: `width`, `height`, `fps`, `exposure`, `gain` for a USB
camera; `exposure_ms` (how long each picture collects light, in milliseconds) and `gain` (brightness boost) for HikRobot.

```
python3 ../shared/cameras.py --list                      # which cameras does this computer see?
python3 ../shared/cameras.py --source usb:0              # live preview. Keys: r record, s snapshot, q quit
python3 ../shared/cameras.py --source hik --cam exposure_ms=2 --record ../videos/clip.avi --seconds 30
python3 m2_yolov5.py --source usb:0 --frame 30           # any script: skip 30 frames, use the next one
python3 m7_compare.py --live --source usb:0 --one-per-network   # every model, live, side by side
```

- **Record first, then compare.** A live camera never shows two models exactly the same picture, so for fair
  comparisons (m6, m8) and for labels, record a clip and use the clip.
- **Distances need the right camera's calibration** ([BASICS 9](../BASICS.md#9-from-4-corners-to-a-distance)). Without
  `--calib`, m8 uses the demo camera's. `../configs/demo_camera.yaml` shows the file format.
- **HikRobot / MindVision cameras** need their maker's software (an "SDK") installed, which includes the Python part
  `cameras.py` uses. If it says it can't find it, the message tells you which setting to point at it.
- **A different kind of camera?** See the top of `../shared/cameras.py`: write one small class and every script can use it.

---

## m1: What does each file say about itself? (`m1_io_cards.py`)

**What this is about.** Before you can use a model you need to know what it expects and what it returns: the name,
shape and number type of each input and output. Most model files record that, and some record more: the names of
their classes, the picture size they were trained at, which file they were converted from. Nothing runs here: you
only read the files.

**What to do.** Fill in `onnx_card()` (using the `onnx` package) and `openvino_card()` (using `openvino`). The NCNN
and TensorRT ones are written for you. Add `--markdown io_cards.md` to also save the summary table.

**Questions**
1. Which model wants **float16** input numbers instead of float32? (Later, in m2, feed it float32 and read the error.)
2. `rp_0708_openvino.xml` remembers which file it was converted from. Under what name? Which file in `../models/` is that now?
3. Which file has an output with **no name**? How can code use an output that has no name?
4. What does an NCNN `.param` file NOT tell you that ONNX and OpenVINO files do? Open one in a text editor.
5. The Ultralytics files (yolov8, yolo11) say `reverse_input_channels: YES` and `scale_values: 255`. Does that mean the
   model swaps the colors and divides by 255 for you, or that YOU must? (Look at the `ppp` block in
   `../reference/sp_vision_25/tasks/auto_aim/yolos/yolo11.cpp` before answering.)
6. Which TensorRT version built each `.engine` file? Which ones can load on our Jetson (JetPack 6 = TensorRT 10.3)?
7. The `rp_0708_openvino2onnx_nms` card shows 5 outputs and an `EfficientNMS_TRT` operation. What can run it, and what can't?

<details><summary>Hint 1</summary>

`onnx.load(path).graph.input[0].type.tensor_type` has `.elem_type` (the number type) and `.shape.dim` (the sizes). Print them.
</details>
<details><summary>Hint 2</summary>

In OpenVINO, `port.get_names()` is a set of names and can be empty. For the extra notes, loop over
`runtimes.OpenVinoRunner.RT_INFO` and use `model.has_rt_info(list(key))` and `model.get_rt_info(list(key)).astype(str)`.
</details>

**Done when:** the checks pass and you can answer the questions from the cards.

---

## m2: The yolov5 family (`m2_yolov5.py`)

**What this is about.** RobotPilots' model: the same kind you decoded in exercises 4 and 5. But it comes in 6 files,
and they differ in small ways that break code: one wants float16 numbers, the answer's name changes from file to file,
and two files return 4 extra outputs.

**What to do.** Fill in `preprocess()`, `raw_output()` and `decode()`.

**Questions**
1. RobotPilots' own README says the color columns 9-12 are red, blue, gray, purple. The robot's code reads them as
   blue, red, gray, purple. The demo video only has blue plates. Who's right? Show the Part A numbers that prove it.
2. Why find the answer by its shape instead of its name? What names does it have across the files?
3. Compare `rp_0708` (2024) and `rp_0526` (2025) on frames 60, 400 and 600 (`--frame 400`): scores, numbers, corners.
4. On a laptop processor, float16 can be slower than float32; on the Jetson's GPU it's faster. Why might that be?
5. Number column 8: RobotPilots says it means "big base plate", the robot's code says "not armor". How could you find out?

**Done when:** the checks pass and every file that runs outlines the plate in `out/`.

---

## m3: TUP's YOLOX models (`m3_yolox.py`)

**What this is about.** TUP's models answer with one row per **grid cell** ([BASICS 8](../BASICS.md#8-grids-cells-and-strides)),
and their corners aren't in pixels yet: they're measured in cells, from the cell's own position. They also want their
input differently: centered, in BGR order, with values 0..255.

**Where to look.** TUP's code (a link, nothing to download): `generateYoloxProposals()` and `scaledResize()` in
[TUP-InfantryVision-2022, `autoaim/detector/inference.cpp`](https://github.com/tup-robomaster/TUP-InfantryVision-2022/blob/main/autoaim/detector/inference.cpp).
The comments in `m3_yolox.py` say what to look for.

**What to do.** Fill in `preprocess()`, `as_rows()`, `grid_strides()`, `corners_in_input_pixels()` and `decode()`.

**Questions**
1. Why exactly 3549 rows for a 416×416 input? And why 25200 for the yolov5 family at 640×640?
2. What does a corner "in cells" mean? Why would a model that thinks in grid cells answer that way?
3. Give TUP's model the yolov5 family's input (RGB, divided by 255). What happens to its confidence? Why?
4. The two NCNN files give the same numbers, arranged differently. What's different?
5. On frame 60, what does `tup_0527` (from 2022) say the plate is? What do the other families say? Why might a
   2022 model get it wrong?
6. `tup_1208` finds nothing on frame 60 with TUP's threshold of 0.75. What's its best confidence there? Should we
   lower the threshold? What would that cost?

<details><summary>Hint</summary>

`np.mgrid[0:h, 0:w]` gives you the row number and column number of every cell; `.ravel()` flattens them row by row,
which is the order the model uses. The stride-8 cells come first.
</details>

**Done when:** the checks pass and the plates are outlined in `out/`.

---

## m4: Ultralytics "pose" models (`m4_ultralytics.py`)

**What this is about.** Reborn's and Tongji's models were made with a popular library called Ultralytics. They answer
**values-first** (one column per guess instead of one row: transpose it), have no separate "is this a plate?" score,
and some of their class names pack several facts into one word, like `Bfour` (blue, number 4).

**Where to look.** `parse()` in `../reference/sp_vision_25/tasks/auto_aim/yolos/yolo11.cpp`, and the list
`armor_properties` in `../reference/sp_vision_25/tasks/auto_aim/armor.hpp`.

**What to do.** Fill in `preprocess()`, `as_rows()` and `decode()`.

**Questions**
1. The output's shape is (1, 50, 8400). Which of those numbers counts the guesses? Why transpose?
2. Where does each file store its class names (look at m1's cards)? Why does `decode()` pick the names by counting the
   classes instead of reading them from the file?
3. `reborn_yolov8` only knows blue or red, not the number. How does the robot's code get the number?
4. Change `preprocess()` to the robot's way (top-left, black padding). Do the scores change? Which way matches how
   these models were trained?
5. On frame 400, how many plates does `tongji_yolo11` find? How many do the other families find?

**Done when:** the checks pass and the plates are outlined in `out/`.

---

## m5: End-to-end models (`m5_end2end.py`)

**What this is about.** Actor&Thinker's models remove the duplicates themselves ("end-to-end",
[BASICS 7](../BASICS.md#7-one-plate-many-guesses-nms)), so they answer with a short list of finished plates. One of them
also takes a non-square 576×768 picture.

**Where to look.** [AT_NN_Detector's README](https://github.com/PraySky1337/AT_NN_Detector) (a link) lists the
input and output; `m5_end2end.py` repeats what you need.

**What to do.** Fill in `preprocess()`, `class_info()` and `decode()`.

**Questions**
1. Why is there no NMS step to write here? Look at the rows whose confidence is about 0: what are they?
2. Why would a team train at 576×768? How much padding does a 1440×1080 camera picture get at 640×640, and at 576×768?
3. The class number comes out as a decimal like 7.0. Why might a model output a whole number as a decimal?
4. Their README says the corners go "clockwise". Look at Part A's picture: do they?

**Done when:** the checks pass and the plates are outlined in `out/`.

---

## m6: One model, many files (`m6_same_network.py`)

**What this is about.** Several files in the zoo are the same trained model converted to another format or precision
([BASICS 10](../BASICS.md#10-model-files-and-the-programs-that-run-them)). A conversion can quietly change the answers.
The only way to know is to run both files on the same pictures and compare.

**What to do.** Needs m2-m5 done. Fill in `compare()`, then run it. It compares every model that has two or more files
on your computer (`--frames 100` and `--every 2` look at more pictures).

**Questions**
1. Which conversions changed nothing, and which changed the answers? Make a table: original, converted file, tool, verdict.
2. What happened to `tup_1208_onnx2ncnn`? Would you have noticed without comparing?
3. `tongji_yolo11_int8` (INT8) vs `tongji_yolo11` (FP32): how far apart are the corners? Use exercise 7's `solve_pnp`
   to work out how much distance error that means for a small plate 5 m away.
4. On the Jetson, add your engines (`*_trt`). Does FP16 change the answers?

**Done when:** you can say which files we can trust, and why.

---

## m7: Side by side (`m7_compare.py`, nothing to write)

```
python3 m7_compare.py --frames 60,400,600
python3 m7_compare.py --one-per-network --frames 100,300,500
```
Each `out/compare_frame<N>.jpg` shows one tile per model, zoomed in on the plates. Add `--reference` if your m2-m5
aren't finished yet (it uses the solutions).

Live, on a camera (keys: q quit, space pause, s snapshot):
```
python3 m7_compare.py --live --source usb:0 --models rp_0526,tongji_yolo11,at_c2psa_576x768
python3 m7_compare.py --live --source hik --one-per-network --headless --save live.avi --seconds 30   # over SSH
```
Hold a plate in front of the camera: move it, tilt it, walk away with it until each model loses it.

**Questions**
1. Find a frame where the models disagree. Who's right?
2. Which models find far, small or tilted plates that others miss?
3. Live: how far away does each model still see a small plate? Does the camera's exposure change that (`--cam exposure_ms=...`)?

---

## On the Jetson: build the engines

```
python3 jetson/build_engines.py                 # an FP16 engine for every ONNX model: a few minutes each
python3 jetson/build_engines.py --fp32          # FP32 engines too, to compare
```
Engines go to `engines/` and show up in every script as `<model>_trt` (and `<model>_trt_fp32`). An engine only works on
the device that built it (m1, question 6), which is why you build your own.

**Questions**
1. Which model is fastest on the GPU? Is it the same one that's fastest on the laptop?
2. FP16 vs FP32: how much faster, and does m6 show any change in the answers?
3. Did any ONNX file fail to build? Read its log in `engines/`.

---

## m8: The mission: pick our model (`m8_benchmark.py`)

Everything so far was about understanding. Now decide.

1. **Speed, on the Jetson.** Run `sudo nvpmodel -m 0 && sudo jetson_clocks` first (full power, fixed clock speeds: otherwise
   timings jump around), then `python3 m8_benchmark.py --csv jetson_speed.csv`. Compare the engines (`*_trt`) with the
   same models run by OpenVINO and ONNX Runtime.
2. **The right answers.** Label frames by hand ([BASICS 11](../BASICS.md#11-measuring-speed-and-quality)): first
   `python3 label_frames.py --every 30` on the demo video, then on **our own footage**, recorded with our camera:
   `python3 ../shared/cameras.py --source hik --record ../videos/ours1.avi --seconds 30`. Use match lighting, red AND blue
   plates, near and far, moving and spinning. Then `python3 label_frames.py --source ../videos/ours1.avi --out ../videos/ours1_labels.json`.
   50+ frames per video. Split the work.
3. **Quality.** `python3 m8_benchmark.py --labels labels.json --csv quality.csv`, and for our footage
   `python3 m8_benchmark.py --source ../videos/ours1.avi --labels ../videos/ours1_labels.json --calib <our camera's config>.yaml`.
4. **Write it up:** one table and a recommendation (which model, which file format, which score threshold, and why).
   Present it to the CV leads.

Things to weigh:
- **recall** (a plate we never see can't be hit) against **precision** (false alarms make the gimbal jump)
- **corner error and jitter**: they turn straight into distance and aiming error
- **FPS on the Jetson** compared with the camera's FPS, with time left over for tracking and aiming
- **what the model tells us**: number, plate size, gray (dead) plates, outpost, base. A model that only knows the color
  needs a second model to read the number
- red plates: the demo video only has blue ones
- licenses: see [MODELS.md](MODELS.md#licenses) before we put a model on the robot

**Questions**
1. Does the score threshold change which model wins? Try `--score 0.5` for every model.
2. Which model is best on the demo video but worst on our footage, or the other way round? Why might that be?
3. What would it take to retrain the winner on our own pictures? (Reborn's and AT's repos include how they trained.)

A first look, from a Windows laptop's processor on 200 demo frames (NOT the Jetson, NOT against labels). "hit %" is how
many frames had at least one plate found; "jitter" is how much the corners wobbled between frames:

| model | runs with | FPS | hit % | jitter (pixels) |
|---|---|---|---|---|
| reborn_yolov8_int8 | OpenVINO, INT8 | 215 | 89 | 2.5 |
| tup_1208 | ONNX Runtime | 171 | 36 | 3.5 |
| tongji_yolo11_int8 | OpenVINO, INT8 | 99 | 81 | 3.0 |
| rp_0708_openvino | OpenVINO | 87 | 93 | 2.7 |
| rp_0526 | ONNX Runtime | 71 | 91 | 2.3 |
| at_c2psa_640 | ONNX Runtime | 57 | 95 | 1.8 |
| at_c2psa_576x768 | ONNX Runtime | 55 | 95 | 1.9 |

Your numbers on the Jetson are the ones that count.

---

## Going further

- Turn Reborn's model into an ONNX file so it can become a TensorRT engine: [MODELS.md, section 7](MODELS.md#7-exporting-yolov8onnx-yourself).
- Add more models: AT's `praysky_coord_noe2e_*`, the OpenVINO versions in TUP's repos, RobotPilots' next release.
  [MODELS.md, section 8](MODELS.md#8-adding-a-new-model) says how.
- Plug the winner into the main track: its plates have the same format as exercise 5's, so exercises 6 and 7 work with it.
