# Basics: the ideas you need before the exercises

You know basic Python: variables, lists, loops, functions. That's enough. This page explains every other
idea the exercises use, in plain words, with small examples. Read it once from top to bottom (about 45
minutes). Later, when a word in an exercise confuses you, look it up in the [glossary](#glossary) at the end.

1. [What we're building](#1-what-were-building)
2. [A picture is a grid of numbers](#2-a-picture-is-a-grid-of-numbers)
3. [numpy in 15 minutes](#3-numpy-in-15-minutes)
4. [What a "model" is](#4-what-a-model-is)
5. [Preparing the model's input](#5-preparing-the-models-input)
6. [Reading the model's output](#6-reading-the-models-output)
7. [One plate, many guesses: NMS](#7-one-plate-many-guesses-nms)
8. [Grids, cells and strides](#8-grids-cells-and-strides)
9. [From 4 corners to a distance](#9-from-4-corners-to-a-distance)
10. [Model files, and the programs that run them](#10-model-files-and-the-programs-that-run-them)
11. [Measuring speed and quality](#11-measuring-speed-and-quality)
12. [The Jetson](#12-the-jetson)
13. [Glossary](#glossary)

---

## 1. What we're building

Every RoboMaster robot wears **armor plates**. Hit a plate with a projectile and the robot loses health. Each
plate has a sticker in the middle (a number or an icon) and a glowing **light bar** on each side, blue or red
depending on the team:

```
     0 (top-left)                         1 (top-right)
        ┃ ┌───────────────────────────┐ ┃
        ┃ │                           │ ┃
        ┃ │            3              │ ┃      ┃ = light bar (glows blue or red)
        ┃ │                           │ ┃      the number in the middle says which robot it is
        ┃ └───────────────────────────┘ ┃
     3 (bottom-left)                      2 (bottom-right)
```

Our robot's camera looks at the other robots. The vision program has to:

1. take a picture from the camera
2. find every armor plate in it, and the **4 corners** of each one (the ends of the light bars)
3. work out what each plate is (color, number)
4. use the 4 corners to work out **how far away** the plate is and in which direction
5. hand that to the code that aims the gimbal (the turret)

Steps 2 and 3 are done by a **neural network model** (section 4). The exercises are about everything around
it: getting the picture ready for the model, understanding what it answers, and turning that into a distance.

---

## 2. A picture is a grid of numbers

A picture is a grid of tiny colored squares called **pixels**. The demo camera takes pictures 1440 pixels wide
and 1080 pixels tall. Each pixel's color is stored as **3 numbers** from 0 to 255: how much blue, green and red
it has (0 = none, 255 = as much as possible).

```python
import cv2                          # OpenCV: the standard library for working with pictures
img = cv2.imread("plate.jpg")       # read a picture file
print(img.shape)                    # (1080, 1440, 3): 1080 rows, 1440 columns, 3 numbers per pixel
print(img[100, 200])                # [ 12  40 200]: the pixel in row 100, column 200
```

Three things trip everyone up:

- **The order is height, width** (rows, then columns), not width, height. `img.shape` is `(1080, 1440, 3)`.
- **(0, 0) is the top-left corner. x goes right, y goes DOWN.** A point at y = 500 is lower in the picture than y = 100.
- **OpenCV stores the 3 numbers as Blue, Green, Red (BGR)**, not RGB. `[12, 40, 200]` above is mostly red.
  Most models were trained on RGB pictures, so you'll often have to swap the order.

A **video** is just many pictures (**frames**) in a row: the demo video has 687 frames, 30 per second.

---

## 3. numpy in 15 minutes

A picture is too many numbers for Python lists (1440 × 1080 × 3 = 4.6 million), so everyone uses **numpy**.
A numpy **array** is like a list that can have several dimensions and does math on every element at once.
Try these in a Python shell (`python3`, then `import numpy as np`):

```python
import numpy as np

a = np.array([3, 8, 1, 5])
a.shape          # (4,)              4 elements, 1 dimension
a[1]             # 8                 indexing starts at 0, like lists
a[1:3]           # array([8, 1])     from index 1 up to (NOT including) 3
a.max()          # 8                 the biggest value
a.argmax()       # 1                 WHERE the biggest value is (its index)  <- you'll use this a lot
a / 2            # array([1.5, 4. , 0.5, 2.5])   math happens on every element at once
a > 4            # array([False,  True, False,  True])
a[a > 4]         # array([8, 5])     keep only the elements where the test is True
```

Arrays with 2 dimensions are tables: rows and columns.

```python
t = np.array([[1, 2, 3],
              [4, 5, 6]])
t.shape          # (2, 3)            2 rows, 3 columns
t[0]             # array([1, 2, 3])  row 0
t[:, 1]          # array([2, 5])     column 1 of every row  (":" means "all of them")
t[:, 0:2]        # array([[1, 2],
                 #        [4, 5]])   columns 0 and 1 of every row
t.T              # array([[1, 4],
                 #        [2, 5],
                 #        [3, 6]])   "transpose": rows become columns. Shape (3, 2)
t.reshape(3, 2)  # array([[1, 2],
                 #        [3, 4],
                 #        [5, 6]])   same 6 numbers, read in order, poured into a new shape
t.reshape(-1)    # array([1, 2, 3, 4, 5, 6])    -1 means "work this size out yourself"
t[None].shape    # (1, 2, 3)         adds a dimension of size 1 at the front
t.astype(np.float32) / 255           # change the number type, then divide
```

A picture is a 3-dimensional array, shape `(height, width, 3)`. `img.transpose(2, 0, 1)` reorders the
dimensions to `(3, height, width)`: "put dimension 2 first, then 0, then 1".

**Why not a for loop?** A model answers with 25200 rows. A Python loop over them is many times slower than one
numpy line like `rows[rows[:, 8] > 0.7]` ("keep the rows whose column 8 is above 0.7"), and the robot does this for
every frame. Exercise 5 has you time both.

---

## 4. What a "model" is

A **neural network model** is a function: numbers go in, numbers come out. Ours take a picture and return
guesses about where the armor plates are.

Nobody wrote its rules by hand. A team showed it tens of thousands of pictures where people had marked every plate,
and it adjusted millions of internal numbers until its guesses matched. That's **training**, and those internal
numbers are its **weights**. A **model file** stores the network's structure plus its weights.

We never train anything here. We take models other teams trained and run them, which is called **inference**.
The whole difficulty is at the edges:

- feed it a picture prepared **exactly** the way it was trained (size, color order, number range...)
- read its answer **exactly** the way it was meant (which number means what)

Get either one wrong and nothing crashes: the model happily runs and gives you garbage. That's why the exercises
are so careful about details. Names you'll see: **YOLO** ("You Only Look Once") is a family of detection models;
YOLOv5, YOLOv8, YOLO11 and YOLOX are versions of it.

---

## 5. Preparing the model's input

A model accepts one exact shape of input. Ours want something like **(1, 3, 640, 640)**, read as:

| number | meaning |
|---|---|
| 1 | how many pictures at once (the **batch**). We always send 1 |
| 3 | the 3 color **channels** |
| 640 | height |
| 640 | width |

A multi-dimensional array like this is called a **tensor**. "NCHW" is a short way to write that order: N pictures,
C channels, H height, W width. A picture from OpenCV is "HWC" (height, width, channels), so it has to be rearranged.

Turning a camera frame into that tensor takes 5 steps:

**1. Resize without stretching (the "letterbox").** The camera gives 1440×1080, the model wants 640×640. Squeezing
it would make plates look thinner than they are, and the model was trained on undistorted pictures. So you shrink
it until it fits, keeping its shape, and fill the leftover space with a plain color:

```
  camera frame 1440 x 1080            model input 640 x 640
 ┌───────────────────────┐          ┌──────────────┐           ┌──────────────┐
 │                       │          │   picture    │           │▒▒▒ padding ▒▒│
 │                       │   --->   │  640 x 480   │    or     │   picture    │
 │                       │          ├──────────────┤           │  640 x 480   │
 └───────────────────────┘          │▒▒ padding ▒▒▒│           │▒▒▒ padding ▒▒│
                                    └──────────────┘           └──────────────┘
                                     "top-left"                  "centered"
```

The **scale** is how much it shrank: 640 / 1440 = 0.444. You'll need it again to turn the model's answers back
into frame pixels. Where the picture goes (top-left or centered) and the padding color (black = 0, or gray = 114)
must match how the model was trained. Different teams did it differently.

**2. Color order.** OpenCV gives BGR. Most models want RGB: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

**3. Value range.** Pixels are 0..255. Most models want 0..1: divide by 255. (Some want 0..255. Read their code.)

**4. Layout.** `(640, 640, 3)` → `(3, 640, 640)` with `.transpose(2, 0, 1)`, then `[None]` adds the batch: `(1, 3, 640, 640)`.

**5. Number type.** Pixels are stored as `uint8` (whole numbers 0..255). Models want **float32** (normal decimal
numbers), sometimes **float16** ("half precision": decimals with fewer digits, which are faster on a GPU).
`.astype(np.float32)` converts.

---

## 6. Reading the model's output

The model doesn't say "there's a plate at x = 800". It returns a big table of **candidates** (guesses), for example
25200 rows of 22 numbers. Each row is one guess, and its numbers mean things like this (the course model):

| columns | meaning |
|---|---|
| 0-7 | the 4 corners: x, y, x, y, x, y, x, y (in the 640×640 input's pixels) |
| 8 | how sure it is that this is a plate (the **score**, or **confidence**) |
| 9-12 | one score per color: blue, red, gray, purple |
| 13-21 | one score per number: sentry, 1, 2, 3, 4, 5, outpost, base, base (big plate) |

Almost all rows are junk with a tiny score. You keep the rows whose score is above a **threshold**, like 0.7.

**Logits and the sigmoid.** Some models give scores like 2.6 or -6 instead of 0..1. That's a **logit**: a raw score
where bigger means "more sure". The **sigmoid** function turns it into a 0..1 probability:

```
sigmoid(x) = 1 / (1 + e^-x)          sigmoid(-6) = 0.002     sigmoid(0) = 0.5     sigmoid(2.6) = 0.93
```

**Picking a class with argmax.** Columns 9-12 might be `[8.2, -7.4, -9.3, -10.8]`. The biggest is at index 0, so the
color is `COLORS[0]`, "blue". That's `np.argmax(row[9:13])`. A **class** is one of the things a model can name
(blue, red, sentry, 3, ...).

**Back to frame pixels.** The corners are in the 640×640 input. Undo the letterbox: subtract the padding (if the
picture was centered), then divide by the scale. With scale 0.444, x = 400 in the input is x = 900 in the frame.

**Corner order.** Each model gives the 4 corners in its own order (often top-left, bottom-left, bottom-right,
top-right). We always rearrange them to **top-left, top-right, bottom-right, bottom-left**, the order the distance
step needs.

**Rows-first or values-first.** Some models return `(25200, 22)`: one row per candidate. Others return `(22, 25200)`,
"values-first". `.T` (transpose) turns one into the other.

---

## 7. One plate, many guesses: NMS

Neighboring rows usually find the same plate, so one plate shows up 5 or 10 times with slightly different corners.
**Non-maximum suppression (NMS)** keeps only the best guess of each group:

1. sort the guesses by score, best first
2. keep the best one
3. throw away every other guess that overlaps it too much
4. repeat with the best guess that's left

"Overlaps too much" is measured with **IoU** (intersection over union): the area two boxes share, divided by the
area they cover together. 0 = they don't touch, 1 = identical. With an IoU threshold of 0.3, any guess sharing more
than 30% with a better one is dropped.

```
 ┌──────────┐                    shared area      ▓▓
 │    ┌─────┼────┐               IoU = ─────────── = ─────────────────
 │    │▓▓▓▓▓│    │                     total area    everything covered
 └────┼─────┘    │
      └──────────┘
```

NMS works on plain rectangles (**boxes**), so for each plate you use the smallest rectangle around its 4 corners.
`cv2.dnn.NMSBoxes` does the whole thing for you. Some models do NMS **inside** themselves and return only the final
plates ("end-to-end" models, or files with an "NMS node").

---

## 8. Grids, cells and strides

Where do 25200 rows come from? Most detection models split the picture into a grid of **cells**, and every cell
makes guesses about whatever is near it. They use 3 grids at once, so both small and big things get found:

| grid | cells (for a 640×640 input) | one cell covers | good for |
|---|---|---|---|
| fine | 80 × 80 = 6400 | 8 × 8 pixels | small, far plates |
| medium | 40 × 40 = 1600 | 16 × 16 pixels | medium plates |
| coarse | 20 × 20 = 400 | 32 × 32 pixels | big, close plates |

That's 8400 cells. YOLOv5 makes 3 guesses per cell (one per **anchor**, a starting box shape), so 3 × 8400 = 25200
rows. A 416×416 model has 52×52 + 26×26 + 13×13 = 3549 cells.

How many pixels one cell covers (8, 16 or 32) is the **stride**. Some models give positions **in cells, relative to
their own cell** rather than in pixels. Then: `pixel = (value + cell number) × stride`. If the cell in column 5 of
the stride-16 grid says x = -1.0, that's (−1 + 5) × 16 = 64 pixels.

**Keypoints** are specific points a model finds, here the 4 plate corners. "Pose" models are models that find
keypoints.

---

## 9. From 4 corners to a distance

A plate that's far away looks small, a plate that's turned looks narrow. Since we know a plate's **real** size
(a small plate's light bars are 13.5 cm apart and 5.6 cm tall), the 4 corners in the picture tell us where it is in
3D. For a rough feel:

```
size in pixels = focal length (pixels) × real size / distance
   so: distance = focal length × real size / size in pixels
   e.g.  1818 × 0.135 m / 82 px = 3.0 m
```

OpenCV's **solvePnP** does the full version (it also handles the plate being turned and off-center). It needs:

- the 4 corners in the picture (2D, pixels)
- the same 4 corners on the real plate (3D, meters), **in the same order**
- the camera's **calibration**

It returns where the plate is relative to the camera: `tvec` (a position, in meters: its length is the distance)
and `rvec` (which way the plate faces).

**Camera calibration** is measured once per camera and lens, usually by photographing a printed checkerboard from
many angles ([OpenCV's tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)). It gives:

- the **camera matrix**: the focal length in pixels (`fx`, `fy`: how much the lens "zooms") and the pixel the lens
  center points at (`cx`, `cy`)
- the **distortion coefficients**: how much the lens bends straight lines near the edges

Use another camera's calibration and every distance is wrong. `configs/demo_camera.yaml` has the demo camera's.

---

## 10. Model files, and the programs that run them

A model file holds the network's structure and its weights. The same model can be saved in different
**formats**, and each needs its own **runtime** (the library that runs it):

| format | files | runtime | runs on |
|---|---|---|---|
| **ONNX** | `.onnx` | ONNX Runtime (`onnxruntime`) | any computer's CPU. The common format everyone converts to and from |
| **OpenVINO** | `.xml` (structure) + `.bin` (weights) | OpenVINO (`openvino`) | CPUs (also Intel graphics, never NVIDIA) |
| **NCNN** | `.param` (structure) + `.bin` (weights) | NCNN (`ncnn`) | CPUs. Made for phones and small boards |
| **TensorRT engine** | `.engine` | TensorRT | NVIDIA GPUs. The fastest option on the Jetson |

A TensorRT **engine** is special: it's compiled for one exact GPU and one exact TensorRT version. You build it
yourself from an ONNX file, on the machine that will run it, with NVIDIA's `trtexec` tool.

**Precision** is how many digits the weights keep:

| | what | trade-off |
|---|---|---|
| **FP32** (float32) | normal decimals | the reference |
| **FP16** (float16) | half the digits | about twice as fast on a GPU, answers change very slightly |
| **INT8** | whole numbers -128..127 | much smaller and faster on a CPU, answers change more ("**quantized**") |

Converting a model to another format or precision can change its answers, sometimes a lot. The model zoo's m6
checks that.

---

## 11. Measuring speed and quality

**Speed.** **Latency** is how long one frame takes, in milliseconds (ms). **FPS** (frames per second) = 1000 / ms.
If the camera gives 60 frames a second and detection takes 25 ms (40 FPS), the robot falls behind.

**ROI** (region of interest): only look at part of the picture, for example a 640×640 square in the middle. It's
faster, and far plates come out bigger because the model's input isn't shrunk as much. But you miss anything outside it.

**Quality.** You mark the right answers by hand on some frames (**labels**, or **ground truth**), then compare:

- **precision**: of the plates the model reported, how many were real? (Low precision = false alarms.)
- **recall**: of the real plates, how many did the model find? (Low recall = missed plates.)
- **corner error**: how many pixels its corners are from yours.

A higher score threshold usually raises precision and lowers recall. **Jitter** is how much the corners wobble from
one frame to the next when the plate barely moved: wobbly corners mean a wobbly distance and wobbly aim.

---

## 12. The Jetson

The robot's computer is an **NVIDIA Jetson Orin Nano**: a small Linux computer with a GPU (graphics chip) that runs
models fast. **JetPack** is its software bundle: Linux, CUDA (NVIDIA's GPU programming toolkit) and TensorRT.
You usually control it from your laptop over the network with **SSH**. Over SSH there's no screen, so programs
run **headless** (no windows) and save their pictures or videos to files instead.

The robot's camera is an industrial camera made by **HikRobot** (another common brand is MindVision). They need
their maker's software (an **SDK**) to be used from Python. A **USB webcam** works without one.

---

## Glossary

| term | meaning | see |
|---|---|---|
| anchor | a starting box shape a YOLOv5 cell guesses around (3 per cell) | [8](#8-grids-cells-and-strides) |
| argmax | the index of the biggest value in an array | [3](#3-numpy-in-15-minutes), [6](#6-reading-the-models-output) |
| armor plate | the target on every robot: a sticker between two light bars | [1](#1-what-were-building) |
| batch | how many pictures go into the model at once (always 1 here) | [5](#5-preparing-the-models-input) |
| BGR / RGB | the order of a pixel's 3 color numbers. OpenCV uses BGR, most models RGB | [2](#2-a-picture-is-a-grid-of-numbers) |
| box / bounding box | the smallest upright rectangle around something | [7](#7-one-plate-many-guesses-nms) |
| calibration | a camera's measured focal length, center and lens distortion | [9](#9-from-4-corners-to-a-distance) |
| candidate | one guess (one row) in a model's output | [6](#6-reading-the-models-output) |
| cell, grid | the picture split into squares; each square makes guesses | [8](#8-grids-cells-and-strides) |
| channel | one of a pixel's 3 color numbers | [5](#5-preparing-the-models-input) |
| class | one of the things a model can name (blue, red, sentry, 3...) | [6](#6-reading-the-models-output) |
| confidence, score | how sure the model is, 0..1 | [6](#6-reading-the-models-output) |
| end-to-end | a model that does its own NMS and returns finished answers | [7](#7-one-plate-many-guesses-nms) |
| engine | a TensorRT model file, compiled for one GPU + TensorRT version | [10](#10-model-files-and-the-programs-that-run-them) |
| FP32, FP16, float32, float16 | normal and half-precision decimal numbers | [5](#5-preparing-the-models-input), [10](#10-model-files-and-the-programs-that-run-them) |
| FPS, latency | frames per second; milliseconds per frame | [11](#11-measuring-speed-and-quality) |
| frame | one picture of a video | [2](#2-a-picture-is-a-grid-of-numbers) |
| gimbal | the turret that aims the robot's launcher | [1](#1-what-were-building) |
| ground truth, labels | the right answers, marked by hand | [11](#11-measuring-speed-and-quality) |
| headless | running without a window (e.g. over SSH) | [12](#12-the-jetson) |
| inference | running a trained model | [4](#4-what-a-model-is) |
| INT8, quantized | a model whose numbers were rounded to small whole numbers | [10](#10-model-files-and-the-programs-that-run-them) |
| IoU | how much two boxes overlap, 0..1 | [7](#7-one-plate-many-guesses-nms) |
| keypoint | a specific point a model finds (here: a plate corner) | [8](#8-grids-cells-and-strides) |
| letterbox | resize without stretching, fill the rest with padding | [5](#5-preparing-the-models-input) |
| light bar | the glowing strip on each side of a plate | [1](#1-what-were-building) |
| logit | a raw score (any number); sigmoid turns it into 0..1 | [6](#6-reading-the-models-output) |
| metadata | extra notes stored inside a model file (labels, sizes...) | [10](#10-model-files-and-the-programs-that-run-them) |
| model, network, weights | the function that detects plates; its learned numbers | [4](#4-what-a-model-is) |
| NCHW, HWC | orders of dimensions: (count, channels, height, width) vs (height, width, channels) | [5](#5-preparing-the-models-input) |
| NMS | keep the best of overlapping guesses, drop the rest | [7](#7-one-plate-many-guesses-nms) |
| ONNX, OpenVINO, NCNN, TensorRT | model file formats and the programs that run them | [10](#10-model-files-and-the-programs-that-run-them) |
| padding | the plain-colored space added around a resized picture | [5](#5-preparing-the-models-input) |
| pixel | one colored square of a picture | [2](#2-a-picture-is-a-grid-of-numbers) |
| PnP, solvePnP | working out where a known object is in 3D from its corners in a picture | [9](#9-from-4-corners-to-a-distance) |
| pose model | a model that finds keypoints | [8](#8-grids-cells-and-strides) |
| precision, recall | how many reported plates were real; how many real plates were found | [11](#11-measuring-speed-and-quality) |
| ROI | region of interest: only the part of the picture you look at | [11](#11-measuring-speed-and-quality) |
| runtime | the library that runs a model file | [10](#10-model-files-and-the-programs-that-run-them) |
| scale | how much the letterbox shrank the picture | [5](#5-preparing-the-models-input) |
| SDK | a camera maker's software for using their camera from code | [12](#12-the-jetson) |
| shape | the size of each dimension of an array, e.g. (1080, 1440, 3) | [3](#3-numpy-in-15-minutes) |
| sigmoid | turns a logit into 0..1 | [6](#6-reading-the-models-output) |
| SSH | controlling another computer over the network from a terminal | [12](#12-the-jetson) |
| stride | how many pixels one grid cell covers | [8](#8-grids-cells-and-strides) |
| tensor | a multi-dimensional array, like the model's (1, 3, 640, 640) input | [5](#5-preparing-the-models-input) |
| threshold | the cut-off: keep only what scores above it | [6](#6-reading-the-models-output) |
| transpose | swap rows and columns (or reorder dimensions) | [3](#3-numpy-in-15-minutes) |
| tvec, rvec | solvePnP's answer: position (meters) and rotation | [9](#9-from-4-corners-to-a-distance) |
| YOLO | "You Only Look Once", a family of detection models | [4](#4-what-a-model-is) |
