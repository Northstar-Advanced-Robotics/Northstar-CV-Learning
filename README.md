# Northstar CV learning track: armor detection from scratch

Our robot has to spot the other robots' armor plates in its camera picture and work out how far away they are.
This track takes you through every step of that, in Python: from "here is a model file" to "this plate is 2.8 m
away". You use the same detection model the robot uses. The robot's code already does all of this; your job is to
work out how, and rebuild it yourself.

**You need basic Python** (variables, lists, loops, functions). Nothing else. **Start with
[BASICS.md](BASICS.md)**: it explains every computer-vision idea and every numpy trick the exercises use, in plain
words. When a word below confuses you, its glossary has it.

```
camera picture (1440 x 1080)
   │
   ▼   prepare it for the model: resize, color order, number range      <- exercise 3
model input (1, 3, 640, 640)
   │
   ▼   the model                                                        <- exercises 1 and 2
25200 guesses, 22 numbers each
   │
   ▼   read the guesses, keep the good ones, remove duplicates          <- exercises 4, 5 and 6
armor plates: 4 corners, color and number each
   │
   ▼   corners + real plate size -> position in 3D (solvePnP)           <- exercise 7
distance and direction -> the robot aims
```

Each exercise is a Python file with functions marked `TODO`. The comments say what to write. When you run the file,
it checks your functions automatically and tells you what's wrong. Once everything passes, it runs your code on
real match footage.

```
exercises/   you work here                          solutions/   working versions + ANSWERS.md
BASICS.md    the ideas you need, in plain words     reference/   the robot's own code, for looking things up
models/      the model files (models/README.md)     videos/      the demo footage, and your own recordings
shared/      cameras.py: read frames from any camera or video, for both tracks
model_zoo/   the second track: every armor model we found, and which one we should use
```

Everything you need is in this repo. Nothing has to be downloaded or cloned apart from the Python packages below.

## How to work through it

- Do the exercises in order. Later ones use your code from earlier ones.
- **Investigate before you code.** The answers are in three places: the model file itself, the robot's code in
  [reference/](reference/README.md), and the numbers the model prints. Learning to dig them out is the real skill.
  You don't need to know C++ to read the robot's code: each exercise tells you which lines to look at.
- Each exercise has hints that start vague and get more specific. Open the next one only when you're stuck.
- Don't read `checks.py` until you've finished an exercise: it gives answers away.
- Write your answers to the **Questions** somewhere and talk them over with a CV lead.

## Setup

**On a laptop** (Windows, Mac or Linux): exercises 1 and 3-7.

```
pip install onnx onnxruntime opencv-python numpy
```

Then tell the exercises to run the model on your laptop's processor:

- Linux / Mac: `export BACKEND=onnxruntime`
- Windows PowerShell: `$env:BACKEND = "onnxruntime"` (and type `python` where this page says `python3`)

That uses ONNX Runtime, a program that runs model files on any computer. It's slower than the robot but gives the
same answers.

**On the Jetson** (the robot's computer, see [BASICS 12](BASICS.md#12-the-jetson)): every exercise, including 2.

- TensorRT comes with JetPack. Check it: `python3 -c "import tensorrt; print(tensorrt.__version__)"`.
- Install PyTorch for Jetson with the JetPack 6 steps in the
  [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson). Check it:
  `python3 -c "import torch; print(torch.cuda.is_available())"` must print `True`.
- `pip install onnx numpy`, and use the OpenCV that came with the Jetson. If `cv2.imshow` fails,
  run `pip uninstall opencv-python-headless`.

**Then:** `cd exercises` and start with exercise 1.

---

## Exercise 1: What's in the model? (`ex1_inspect_onnx.py`)

**What this is about.** A model file is a recipe of operations plus millions of learned numbers
([BASICS 4](BASICS.md#4-what-a-model-is)). Before you run it, you need to know what it expects to receive and what it
will give back: its **inputs** and **outputs**, each with a name and a shape. ONNX files store all of that, and
the `onnx` Python package can read it without running anything.

**What to do.** Fill in `describe_io`, `count_ops` and `find_nms_node`. Then open
`models/robotpilots/rp_0708_openvino2onnx_nms.onnx` in [Netron](https://netron.app) (a website that draws a model
as a diagram: drag the file onto it) and scroll through it.

**Questions**
1. What are the input's name and shape? What does each of the 4 numbers mean? ([BASICS 5](BASICS.md#5-preparing-the-models-input))
2. There are 5 outputs. Which one looks like the model's real answer, and which ones look added later?
3. The real output has 25200 rows of 22 numbers. Why exactly 25200? ([BASICS 8](BASICS.md#8-grids-cells-and-strides))
4. One operation is called `EfficientNMS_TRT`. What does NMS do ([BASICS 7](BASICS.md#7-one-plate-many-guesses-nms)), and
   which program can run this particular operation? (The `_TRT` is a clue.)
5. The script prints which values feed the NMS step's box input. In what order are the box coordinates?
   Remember this for exercise 6.
6. The file says it was made by `openvino2onnx`. Read `reference/sp_vision_25/JETSON_ORIN.md` (search for
   "openvino2onnx"): where did this file come from, and why couldn't it just be exported from the original training code?

<details><summary>Hint 1</summary>

`model.graph.input` and `model.graph.output` are lists. Print one element and read what's inside.
</details>
<details><summary>Hint 2</summary>

A shape lives at `value.type.tensor_type.shape.dim`, and each `dim` has a `dim_value`.
</details>
<details><summary>Hint 3</summary>

`node.op_type` is the operation's name. `node.attribute` is a list of settings, and
`onnx.helper.get_attribute_value(attr)` gives you a setting's value.
</details>

**Done when:** all checks pass and you can answer the questions.

---

## Exercise 2: Build and load a TensorRT engine (`ex2_load_engine.py`, Jetson only)

**What this is about.** On the Jetson, the fastest way to run a model is NVIDIA's **TensorRT**. It takes an ONNX file
and compiles it into an **engine**: a version optimized for that exact GPU ([BASICS 10](BASICS.md#10-model-files-and-the-programs-that-run-them)).
Then you load the engine from Python, hand it a picture and read its answer.

**Part A: build the engine.** TensorRT comes with a command-line tool called `trtexec` (in `/usr/src/tensorrt/bin/`).
Use it to build `exercises/yolov5.engine` from `../models/robotpilots/rp_0708_openvino2onnx_nms.onnx`, with FP16
(half precision) turned on. Run `trtexec --help` and find the options yourself.

Things that go wrong:
- **Type the command on one line.** A stray `\` or space from copy-pasting gives a misleading
  "Model missing or format not recognized" error.
- **The build takes about 4 minutes** and needs memory. Close other programs that use the GPU. Lines with
  `NvMapMemAlloc ... error 12` mean the Jetson ran low on memory.
- When it finishes, `trtexec` measures the engine's speed. Write down the "GPU Compute Time".

**Part B: load it in Python.** Fill in `TRTModel`. The comments in the file list the steps. PyTorch is only used to
hold the numbers in GPU memory.

**Questions**
1. Build a second engine without FP16. How do the speeds compare? Why might FP16 be good enough for this model?
2. Why can't you copy an engine from one Jetson to a laptop, or to a Jetson with a different JetPack? (Read the
   `.engine` part of `reference/sp_vision_25/JETSON_ORIN.md`.)
3. The NMS step from exercise 1 is a TensorRT "plugin": an add-on that has to be registered before loading. What
   happens if you skip registering the plugins? Try it and read the error.
4. The script times 100 runs. Is that more or less than trtexec's number? Where does the extra time go?

<details><summary>Hint 1</summary>

The options you need are `--onnx=`, `--saveEngine=` and `--fp16`.
</details>
<details><summary>Hint 2</summary>

TensorRT 10 names every input and output. Loop over `range(engine.num_io_tensors)`, use
`engine.get_tensor_name(i)`, then `get_tensor_shape`, `get_tensor_dtype` and `get_tensor_mode`.
</details>
<details><summary>Hint 3</summary>

`context.set_tensor_address(name, buffer.data_ptr())` tells TensorRT where each input and output lives in memory.
To run, call `context.execute_async_v3(torch.cuda.current_stream().cuda_stream)` and then
`torch.cuda.synchronize()` (wait until the GPU is done) before reading the outputs.
</details>

**Done when:** all checks pass. On a laptop the checks are skipped, so pair up with someone on the Jetson for this one.

---

## Exercise 3: Preparing the picture (`ex3_preprocess.py`)

**What this is about.** The camera gives a 1440×1080 picture of whole numbers 0-255 in BGR color order. The model
wants something else entirely: the (1, 3, 640, 640) tensor it was trained on. Read
[BASICS 5](BASICS.md#5-preparing-the-models-input) first: it walks through all 5 steps. This exercise is about doing
them **exactly the way the robot does**.

**Where to look.** `reference/sp_vision_25/tasks/auto_aim/yolos/yolov5.cpp`:
- the function `YOLOV5::detect()`: how the picture is resized and where it's placed (look for `cv::Rect(0, 0, w, h)`)
- the block with `ppp` near the top of the file (the OpenVINO preprocessing): what happens to the colors and values

**What to do.** Fill in `letterbox` and `to_tensor`.

**Questions**
1. Why not just stretch the frame to 640×640?
2. Where does the robot put the resized picture on the 640×640 canvas, and what fills the rest? Many examples online
   center it instead. Why does it matter which one you match?
3. What has to happen to the color order and to the pixel values?
4. What is the `scale` for, and where will you need it again?

<details><summary>Hint 1</summary>

`scale = min(640 / height, 640 / width)`. Resize with `cv2.resize`, then copy the result into
a black 640×640 array (`np.zeros((640, 640, 3), np.uint8)`). `cv::Rect(0, 0, w, h)` means "starting at the top-left".
</details>
<details><summary>Hint 2</summary>

The robot converts BGR→RGB, whole numbers→decimals, divides by 255, and reorders the dimensions from
(height, width, channels) to (channels, height, width). In Python that's `cv2.cvtColor`,
`.astype(np.float32) / 255`, `.transpose(2, 0, 1)` and adding the batch dimension with `[None]`.
</details>

**Done when:** all checks pass and `ex3_letterbox.jpg` looks like what you expect.

---

## Exercise 4: Make sense of the raw output (`ex4_raw_output.py`)

**What this is about.** The model answers with 25200 rows of 22 numbers each: one row per guess
([BASICS 6](BASICS.md#6-reading-the-models-output)). Nothing says which number means what. This is the detective
exercise: work it out, then turn one row into a plate.

**What to do.** **Run the file first**, before writing any code. It prints the 5 rows with the largest value in
column 8 and saves `ex4_raw_points.jpg`, which draws the first 8 numbers of the best row as 4 numbered points. Then
read `YOLOV5::parse()` in `reference/sp_vision_25/tasks/auto_aim/yolos/yolov5.cpp`. Then fill in `sigmoid` and
`decode_row`.

**Questions**
1. Which columns are the plate's corners? In what order do the 4 corners come out? Look at the numbered points in
   `ex4_raw_points.jpg`.
2. Column 8 is how sure the model is, but its values are like 2.6 or -6, not 0 to 1. Why? What turns it into a
   probability? ([BASICS 6](BASICS.md#6-reading-the-models-output): logits)
3. Columns 9–12 and 13–21 are two groups of scores. What does each group tell you, and how do you pick the answer
   from a group? (`reference/sp_vision_25/tasks/auto_aim/armor.hpp` lists what each index means.)
4. Why are 4 corners more useful for aiming than a plain rectangle around the plate?

<details><summary>Hint 1</summary>

The corners are in 640×640 *input* pixels. Divide by the letterbox scale to get back to the original frame.
There's nothing to subtract first, because the picture sits at the top-left.
</details>
<details><summary>Hint 2</summary>

In `parse()`, the robot builds `armor_key_points` from `row[0..1]`, `row[6..7]`, `row[4..5]`, `row[2..3]`,
in that order. Compare that with the order you saw in the picture.
</details>
<details><summary>Hint 3</summary>

`np.argmax(row[9:13])` gives the index of the biggest color score. Use it to pick from `config.COLORS`.
Same for the numbers with `config.NUMBERS`.
</details>

**Done when:** all checks pass and `ex4_decoded.jpg` outlines the plate.

---

## Exercise 5: Decode every row and remove duplicates (`ex5_decode.py`)

**What this is about.** Exercise 4 decoded one row. Now do all 25200 at once with numpy
([BASICS 3](BASICS.md#3-numpy-in-15-minutes)), keep only the confident ones, and remove the duplicates with NMS
([BASICS 7](BASICS.md#7-one-plate-many-guesses-nms)). The result: a clean list of plates.

**What to do.** Fill in `decode_all` and `draw_detections`.

**Questions**
1. Run your exercise 4 code on the top 5 rows. Why does one plate show up in several rows?
2. What does NMS do, and what does the IoU threshold (0.3) control?
3. The robot compares column 8 with `log(t / (1 - t))` instead of taking the sigmoid of every row. Why does that give
   the same result, and why is it faster?
4. How long does your `decode_all` take per frame? Compare a Python `for` loop over all 25200 rows with numpy
   working on the whole array at once.

<details><summary>Hint 1</summary>

`raw[raw[:, 8] >= threshold]` keeps only the rows whose column 8 is big enough, all at once.
</details>
<details><summary>Hint 2</summary>

`cv2.dnn.NMSBoxes(boxes, scores, score_thr, iou_thr)` wants one box per plate as `[x, y, width, height]`. Make it
from the smallest and largest x and y of the plate's 4 corners. It returns the indices of the plates to keep.
</details>

**Done when:** all checks pass and `ex5_result.jpg` shows labeled plates.

---

## Exercise 6: Live video, and looking at only part of the picture (`ex6_live.py`)

**What this is about.** Run your detector on every frame of a video or a live camera. Then only look inside a
**region of interest (ROI)**: a smaller rectangle of the picture ([BASICS 11](BASICS.md#11-measuring-speed-and-quality)).

**What to do.** Fill in `detect_in_roi`, then run:
```
python3 ex6_live.py                                    # demo footage
python3 ex6_live.py --roi 420 50 600 600               # same, but only a 600x600 square starting at x=420, y=50
python3 ex6_live.py --source usb:0 --width 1280 --height 720 --fps 60   # a USB camera
python3 ex6_live.py --source hik --cam exposure_ms=2                    # the robot's HikRobot camera
```
`--source` takes anything [shared/cameras.py](shared/cameras.py) can open: a video file, a USB camera (`usb:0`), the
robot's HikRobot or MindVision camera (`hik`, `mv`), a Jetson ribbon camera (`csi:0`), or a camera config file
(`config:../configs/usb_camera.yaml`). In the window: `q` quits, `space` pauses, `n` toggles the bonus.

Over SSH there's no screen ([BASICS 12](BASICS.md#12-the-jetson)): add `--headless --out result.mp4` and watch the
file afterwards. To see which sizes and frame rates a USB camera supports on Linux, run
`v4l2-ctl -d /dev/video0 --list-formats-ext`.

**Questions**
1. Why might the robot only look inside an ROI? What does it gain, and what does it risk?
2. A 600×600 ROI gets scaled UP to 640×640, while the whole 1440×1080 frame gets scaled DOWN. Does a small ROI help
   with far-away plates? Is there a limit?
3. Compare the detection FPS with the camera's FPS. Which one limits the system?

**Bonus:** fill in `draw_engine_nms` so that pressing `n` draws the boxes from the engine's own NMS step (TensorRT
only). Draw them the obvious way first. Something will be wrong. Explain what, using what you found in exercise 1,
question 5, then fix it.

**Done when:** outlines follow the plates in the window, and stay correct with an ROI set.

---

## Exercise 7: How far away is it? (`ex7_pnp.py`)

**What this is about.** You know the 4 corners in the picture and the real size of a plate. OpenCV's `solvePnP` turns
that into where the plate is in 3D, which gives the distance. Read [BASICS 9](BASICS.md#9-from-4-corners-to-a-distance)
first.

**Where to look.** `reference/sp_vision_25/tasks/auto_aim/solver.cpp`: the plate sizes (top of the file), the 3D
corner points (`BIG_ARMOR_POINTS`, `SMALL_ARMOR_POINTS`) and the `solvePnP` call the robot uses.

**What to do.** Fill in `armor_object_points` and `solve_pnp`.

**Questions**
1. What does `solvePnP` need besides the 4 corners in the picture? Why does each one matter?
2. Why must the 3D points be in the same order as the corners in the picture?
3. What are the camera matrix and the distortion coefficients? Why can't you use the demo camera's values with your
   own camera?
4. Small and big plates have the same light-bar height but different widths. What happens to the distance if you use
   the wrong size?

<details><summary>Hint 1</summary>

The corners are the ends of the two light bars. The robot measures from the plate's center, with x pointing out
of the plate, y to its left and z up. So the top-left corner is `(0, +width/2, +lightbar_length/2)`.
</details>
<details><summary>Hint 2</summary>

`cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=...)` returns
`(ok, rvec, tvec)`. `np.linalg.norm(tvec)` is the distance in meters.
</details>

**Done when:** all checks pass and the demo video shows sensible distances with 3D axes on each plate.

**Your own camera:** it needs its own calibration first ([BASICS 9](BASICS.md#9-from-4-corners-to-a-distance),
[OpenCV's tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)). Put the numbers in a config
file like `configs/demo_camera.yaml` and pass it: `python3 ex7_pnp.py --source usb:0 --calib ../configs/my_camera.yaml`.
The robot's own cameras are already calibrated: ask a CV lead for their config file.

---

## Going further

- **Model zoo:** [`model_zoo/`](model_zoo/README.md) has 10 armor detectors from 5 teams plus our own, in 4 file formats (the
  model you just used is one of them). Learn how they differ, get all of them running, and measure which one we
  should put on the robot.
- **INT8:** build an INT8 engine ([BASICS 10](BASICS.md#10-model-files-and-the-programs-that-run-them)). What does
  TensorRT need to "calibrate" it, and how much faster and less accurate is it?
- **Find the slow part:** time preparing the picture, running the model, decoding and drawing separately.
  The robot moved the preparation onto the GPU. Was that worth it?
- **Compare with the robot:** build the robot's C++ program
  ([Northstar-sp_vision_25](https://github.com/Northstar-Advanced-Robotics/Northstar-sp_vision_25)), run it on the
  same frame and check that your corners match its corners to the pixel.
- **Retrain:** where would new training pictures come from, and how would a retrained model get back into this
  pipeline? (`reference/sp_vision_25/JETSON_ORIN.md`)

The `solutions/` folder has a working version of every exercise, and `solutions/ANSWERS.md` answers the questions.
