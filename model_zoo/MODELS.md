# The armor model zoo: where every model came from, and how they differ

This is the reference for `model_zoo/`: the detailed facts behind each model. The walkthrough (what to do, in
what order) is in [README.md](README.md), and every term used here is explained in [BASICS.md](../BASICS.md).

Every fact below was checked: each file was compared byte-for-byte with the original in its team's repo, inputs and
outputs were read from the files themselves, and the meaning of every number was confirmed by running the models on
the demo video.

"IO" below means a model's **i**nputs and **o**utputs: their names, shapes and number types.

## Contents
1. [Where each file came from](#1-where-each-file-came-from)
2. [The six output families at a glance](#2-the-six-output-families-at-a-glance)
3. [Family cards](#3-family-cards)
4. [The formats: ONNX, OpenVINO, NCNN, TensorRT](#4-the-formats)
5. [Things we found out the hard way](#5-things-we-found-out-the-hard-way)
6. [Not in the zoo, and why](#6-not-in-the-zoo-and-why)
7. [Exporting yolov8.onnx yourself](#7-exporting-yolov8onnx-yourself)
8. [Adding a new model](#8-adding-a-new-model)
9. [Licenses](#licenses)

---

## 1. Where each file came from

Every RoboMaster team below published their model openly. The files live in `models/<team>/`, renamed so the
name says what's inside. "Original name" is what the file was called where it came from, so you can still find it
there. "Same bytes" means the file is byte-for-byte identical (same git hash) to the one in the linked repo.

| zoo name = file in `models/` | team | original name and place | same bytes? |
|---|---|---|---|
| `robotpilots/rp_0708.onnx` | Shenzhen Univ. **RobotPilots** | `Model/0708.onnx` in [broalantaps/RobotDetectionModel](https://github.com/broalantaps/RobotDetectionModel) | yes |
| `robotpilots/rp_0526.onnx` | RobotPilots | `Model/0526.onnx`, same repo (their 2025 release) | yes |
| `robotpilots/rp_0708_openvino.xml/.bin` | Tongji **SuperPower** (converted RobotPilots' 0708) | `assets/yolov5.xml/.bin` in [TongjiSuperPower/sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25) | yes |
| `robotpilots/rp_0708_openvino2onnx.onnx` | Northstar (our fork) | `scripts/onnx/yolov5_prefusion_backup.onnx` in [Northstar-sp_vision_25](https://github.com/Northstar-Advanced-Robotics/Northstar-sp_vision_25), see its `JETSON_ORIN.md` | yes |
| `robotpilots/rp_0708_openvino2onnx_nms.onnx` | Northstar | `assets/yolov5.onnx`, same repo: **the main track's model** | yes |
| `robotpilots/rp_0708_openvino2onnx_nms_trt10.3.engine` | Northstar | `yolov5.engine`, built from the one above with TensorRT 10.3 | - |
| `robotpilots/rp_v8n_0726.onnx` | RobotPilots' vision group **SZURPVision** | `所有模型/onnx/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx` in [SZURPVision/26_NNDeployment_Lib_and_Detection_Models](https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models) (their 2026 model) | yes |
| `tup/tup_0527.onnx` | Shenyang Aerospace Univ. **TUP** | `model/opt-0527-001.onnx` in [tup-robomaster/TUP-InfantryVision-2022](https://github.com/tup-robomaster/TUP-InfantryVision-2022) | yes |
| `tup/tup_1208.onnx` | TUP | `opt-1208-001.onnx` in a copy of TUP's 2023 repo, [RebornVision/TUP-Vision-2023-Based](https://github.com/RebornVision/TUP-Vision-2023-Based) | yes |
| `tup/tup_1208_onnx2ncnn.param/.bin` | unknown, converted with `onnx2ncnn` | `opt-1208-001.param/.bin` | - |
| `tup/tup_1208_pnnx.ncnn.param/.bin` | unknown, converted with `pnnx` | `opt_1208_001.ncnn.param/.bin` | - |
| `tup/tup_1208_trt10.6.engine` | unknown | `opt-1208-001.engine`, built with TensorRT 10.6.0.26: **won't load on JetPack 6 (10.3)**. Kept as the example | - |
| `reborn/reborn_yolov8_int8.xml/.bin` | USTB **Reborn** | `model/mobilenetv3_last_int_all_new/last.xml` in [RebornVision/Reborn-Vision-2024-armor-Inference](https://github.com/RebornVision/Reborn-Vision-2024-armor-Inference); sp_vision calls it `assets/yolov8.xml` | yes |
| `reborn/reborn_yolov8.onnx` (you make it) | Reborn | training run + `.pt` in [gaoxinstudent/RM-Armor-Detect](https://github.com/gaoxinstudent/RM-Armor-Detect), see [section 7](#7-exporting-yolov8onnx-yourself) | - |
| `tongji/tongji_yolo11_int8.xml/.bin` | Tongji SuperPower | `assets/yolo11.xml/.bin` in sp_vision_25, also `resource/yolo11.xml` in [TongjiSuperPower/LabelRoboMaster](https://github.com/TongjiSuperPower/LabelRoboMaster) | yes |
| `tongji/tongji_yolo11.onnx` | Tongji SuperPower | `resource/best.onnx` in LabelRoboMaster (FP32, before INT8) | yes |
| `actor_thinker/at_c2psa_640.onnx` | HEBUST **Actor&Thinker** | `common/praysky_c2psa_e2e_0228_640x640.onnx` in [PraySky1337/AT_NN_Detector](https://github.com/PraySky1337/AT_NN_Detector) | yes |
| `actor_thinker/at_c2psa_576x768.onnx` | Actor&Thinker | `common/praysky_c2psa_e2e_0228_576x768.onnx`, same repo | yes |
| `northstar/ns_yolo26n_v2.pt` | **our team** | `v2_final.pt`: YOLO26n-pose, trained on synthetic plates (run `armor_pose_ci_synth`), then fine-tuned on Actor&Thinker's dataset (run `armor_pose_ci_ft_actor_thinker`), Ultralytics 8.4.187 | - |
| `northstar/ns_yolo26n_v2.onnx` | our team | exported from the `.pt` with `tools/export_ultralytics.py` (end-to-end head) | - |

The zoo name is the file name without its extension (`rp_0708_openvino`, `tup_1208_pnnx`, ...). The demo video is
`videos/demo.avi`, the same bytes as `assets/demo/demo.avi` in our sp_vision fork.

How the RobotPilots network turned into five files:

```
rp_0708.onnx --OpenVINO model optimizer--> rp_0708_openvino.xml --openvino2onnx--> rp_0708_openvino2onnx.onnx --+ EfficientNMS--> rp_0708_openvino2onnx_nms.onnx --trtexec--> ..._trt10.3.engine
(RobotPilots, PyTorch 1.13)                (sp_vision; rt_info still              (Northstar; the main track's            (Northstar; the main track's model,          (Northstar)
                                            says input_model = 0708.onnx)          laptop model)                           only TensorRT can run it)
```

`sp_vision_25`'s readme credits its 4-point detectors to RobotPilots and Reborn. `yolo11` looks like Tongji's own: it isn't
credited to anyone, and its FP32 version ships with their labeling tool LabelRoboMaster.

---

## 2. The six output families at a glance

Same job (find armor plates, return 4 corners), six completely different ways of being asked and of answering.
Each column is one family: models that answer in the same layout, so one decoder (m2-m5b) reads all of them.

| | **yolov5** (m2) | **yolox** (m3) | **ultralytics** (m4) | **szurp_v8** (m4b) | **end2end** (m5) | **yolo26_e2e** (m5b) |
|---|---|---|---|---|---|---|
| models | rp_0708, rp_0526, rp_0708_openvino* | tup_0527, tup_1208 (+NCNN) | reborn_yolov8, tongji_yolo11 | rp_v8n_0726 | at_c2psa_640, at_c2psa_576x768 | ns_yolo26n_v2 (ours) |
| input size | 640x640 | 416x416 | 416x416 / 640x640 | **480x640** | 640x640 / **576x768** | 640x640 |
| input dtype | float32 (**rp_0526: float16**) | float32 | float32 | float32 (despite "fp16" in its name) | float32 | float32 |
| color order | RGB | **BGR** | RGB | RGB | RGB | RGB |
| values | 0..1 | **0..255** | 0..1 | 0..1 | 0..1 | 0..1 |
| letterbox | **top-left**, black | centered, black | centered, gray 114 | centered, **gray 124** (none for 4:3 frames) | centered, gray 114 | centered, gray 114 |
| output | (1, 25200, 22) | (1, 3549, 21) | **(1, 4+C+8, N)** values-first | **(1, 21, 6300)** values-first | (1, 30, 18) | **(1, 300, 18)** |
| one row is | one anchor box | one grid cell | one grid cell (after transposing) | one grid cell (after transposing) | one finished detection | one finished detection |
| corners | pixels | **cell offsets** | pixels | pixels | pixels | pixels, **plus a "visible" score each** |
| confidence | column 8, **a logit** | column 8, probability | max class score, probability | max class score, probability | column 4, probability | column 4, probability |
| labels | 4 colors + 9 numbers | 4 colors + 8 numbers | 2 colors, or 38 combined classes | 4 colors + 9 classes (**no box**) | 12 size+number classes + 4 colors | 8 color+number classes |
| NMS | you | you | you | you (overlap limit **0.2**) | **inside the model** | **inside the model** |
| corner order | TL, BL, BR, TR | TL, BL, BR, TR | TL, BL, BR, TR | TL, BL, BR, TR | TL, BL, BR, TR | TL, BL, BR, TR |

Everything in **bold** is a trap: get it wrong and the model runs fine but gives garbage, or nothing.

---

## 3. Family cards

### yolov5: RobotPilots' 4-point YOLOv5 (m2)
- **In plain words:** one row per guess, corners already in pixels, the "is it a plate?" score needs a sigmoid, you remove
  the duplicates yourself. The model the main track uses.
- **Where the facts come from:** RobotPilots' `OpenvinoInfer.cpp` and README; the robot's
  `../reference/sp_vision_25/tasks/auto_aim/yolos/yolov5.cpp`.
- **Input:** `images`, (1, 3, 640, 640). The robot's code puts the picture **top-left** on black, converts BGR to RGB,
  divides by 255.
  `rp_0526` takes **float16**: feed it float32 and ONNX Runtime refuses.
- **Output:** 25200 rows = 3 anchors x (80x80 + 40x40 + 20x20) cells. Each row:

  | columns | meaning |
  |---|---|
  | 0-7 | 4 corners x, y in input pixels: TL, BL, BR, TR |
  | 8 | objectness **logit**: sigmoid it, or compare it with log(t / (1 - t)) |
  | 9-12 | color: **blue, red, gray, purple** |
  | 13-21 | number: sentry, 1, 2, 3, 4, 5, outpost, base (small), base (big) |
- **Names differ by file:** `output` (rp_0708, rp_0526, rp_0708_openvino) vs `output/sink_port_0` (the two rp_0708_openvino2onnx files;
  the `_nms` one and its engine also have `num_detections`, `detection_boxes`, `detection_scores`, `detection_classes`). Find the raw output by shape.
- **Threshold:** 0.7 (the robot's value). NMS IoU 0.3.

### yolox: TUP's 2022 YOLOX (m3)
- **In plain words:** one row per grid cell, corners measured in cells (multiply out to get pixels), raw BGR 0..255 input,
  you remove the duplicates yourself.
- **Where the facts come from:** TUP-InfantryVision-2022 `autoaim/detector/inference.cpp` (`scaledResize`, `generateYoloxProposals`)
  and `autoaim/autoaim.cpp` (label meanings).
- **Input:** `images` (`in0` in the pnnx NCNN file), (1, 3, 416, 416). Letterbox **centered** with black bars,
  **BGR, raw 0..255 floats**: no color swap, no division.
- **Output:** 3549 rows = 52x52 + 26x26 + 13x13 grid cells (strides 8, 16, 32), stride-8 grid first, row by row. Each row:

  | columns | meaning |
  |---|---|
  | 0-7 | 4 corners as **offsets in cells**: pixel = (value + cell) x stride. TL, BL, BR, TR |
  | 8 | objectness, already a probability |
  | 9-12 | color: blue, red, gray, purple |
  | 13-20 | class: sentry, 1 (hero), 2 (engineer), 3, 4, 5, outpost, base |
- The pnnx NCNN file returns the same numbers **transposed**: (1, 21, 3549).
- **Threshold:** 0.75 (TUP's `BBOX_CONF_THRESH`). TUP also averages the corners of overlapping boxes after NMS; we don't.
- TUP's 2023 models use a different layout (8 colors, 25 columns): `opt-0527-001` and `opt-1208-001` are both the 21-column kind.

### ultralytics: YOLOv8 / YOLO11 pose (m4)
- **In plain words:** values-first (transpose it), no separate "is it a plate?" score (the best class score is the score),
  corners already in pixels, class names that pack color + number + size into one word.
- **Where the facts come from:** the robot's `../reference/sp_vision_25/tasks/auto_aim/yolos/yolov8.cpp`, `yolo11.cpp`
  and `armor.hpp` (`armor_properties`); the
  `model_info` and Ultralytics metadata inside the files.
- **Input:** (1, 3, H, W): `images` 416x416 for yolov8, **`x`** 640x640 for tongji_yolo11_int8.xml (`images` in tongji_yolo11.onnx). Trained with a
  **centered** letterbox and **gray 114** bars, RGB, 0..1. (The robot's code feeds them top-left on black. That runs too: compare the scores.)
- **Output:** **values-first** (1, 4 + C + 8, N). Transpose to (N, 4 + C + 8): `cx, cy, w, h`, C class scores (already 0..1),
  then 4 keypoints x, y in input pixels (TL, BL, BR, TR). No objectness: score = best class score.
  - `reborn_yolov8`: C = 2, labels `B R`. **Color only**: the robot's code reads the number with a second, separate model.
  - `tongji_yolo11`: C = 38, labels like `Bfour`, `Rbasesmall`, `Ebalancethree` (B blue, R red, E extinguished = gray, P purple;
    `balance` = big 3/4/5 plates). The table is `armor_properties` in the robot's `armor.hpp`.
- tongji_yolo11_int8.xml's output has **no name**: use it by position.
- The files carry their own labels (ONNX `metadata_props["names"]`, OpenVINO rt_info `model_info/labels`). Engines don't, so the
  decoder picks the label table from C.
- **Threshold:** 0.7 (the robot's value).

### szurp_v8: RobotPilots' 2026 model (m4b)
- **In plain words:** an Ultralytics YOLOv8n-pose with a redesigned head. One column per guess like m4, but no box:
  4 color scores, 9 class scores, 4 corners. The score is the best class score.
- **Where the facts come from:** SZURPVision's README (section 4.1) and their `network_preprocess.cpp`,
  `network_deployment_openvino.cpp` and `network_postprocess.cpp` (`V8_21InfantryPostProcessor`).
- **Input:** `images`, (1, 3, **480, 640**): 480 high, 640 wide, the camera's 4:3 shape, so a 1440x1080 frame is
  only resized. Other shapes are centered on **gray 124**. RGB, 0..1, float32 (the "fp16" in the file name is about
  how its weights are stored, not its inputs).
- **Output:** (1, 21, 6300), one column per guess. 6300 = 60x80 + 30x40 + 15x20 grid cells. Transposed, each row:

  | columns | meaning |
  |---|---|
  | 0-3 | color scores: blue, red, **white** (switched off), purple. The 2026 model never says purple |
  | 4-12 | class scores: sentry, hero (1), engineer (2), 3, 4, 5, outpost, base bottom, base top. Never says 5 |
  | 13-20 | 4 corners x, y in input pixels: TL, BL, BR, TR |
- Hero (1) and the base's bottom plate are **big** plates, everything else small (their code says so).
- The class names stored in the file are placeholders (`class0` ... `class8`): use the README's table.
- **Threshold:** 0.5. NMS overlap limit **0.2** (`defaultNmsThreshold` for armor), stricter than the others' 0.3.

### end2end: Actor&Thinker's end-to-end YOLO pose (m5)
- **In plain words:** the model removes the duplicates itself and returns at most 30 finished plates; just keep the ones
  with a high enough score.
- **Where the facts come from:** AT_NN_Detector's README.
- **Input:** `images`, (1, 3, 640, 640) or **(1, 3, 576, 768)**: height 576, width 768. Centered letterbox, gray 114, RGB, 0..1.
  576x768 has the camera's 4:3 shape, so a 1440x1080 frame is just resized: no bars, no wasted pixels.
- **Output:** (1, 30, 18): the 30 best candidates, already one per plate (a TopK step is inside the model). Each row:

  | columns | meaning |
  |---|---|
  | 0-3 | box x1, y1, x2, y2 (input pixels) |
  | 4 | confidence, 0..1 (unused rows are ~0) |
  | 5 | class id as a **float**: 0..11 = `s0_o0 s0_o2 s0_o3 s0_o4 s0_o5 s0_o6 s1_o0 s1_o1 s1_o3 s1_o4 s1_o5 s1_o7` (s0 small / s1 big plate, o0 sentry, o1-o5 robots, o6 outpost, o7 base) |
  | 6-9 | color scores: blue, red, gray, purple |
  | 10-17 | 4 keypoints x, y: TL, BL, BR, TR on screen |
- **No NMS to write.** Threshold: AT's preview uses 0.25; we use 0.5.
- The same repo also has `praysky_coord_noe2e_*` models (CoordAtt attention, NMS export, same 1x30x18 output): worth adding.

---

### yolo26_e2e: our own YOLO26 model (m5b)
- **In plain words:** end-to-end like m5's models and the same row width (18), but the 18 numbers mean something else:
  no color scores, and 3 numbers per corner.
- **Where the facts come from:** the checkpoint (`python3 tools/inspect_pt.py ../models/northstar/ns_yolo26n_v2.pt` reads its
  settings without loading it) and the exported ONNX's card in m1.
- **Input:** `images`, (1, 3, 640, 640), centered on gray 114, RGB, 0..1, float32.
- **Output:** (1, **300**, 18), the 300 best guesses, one plate each:

  | columns | meaning |
  |---|---|
  | 0-3 | box x1, y1, x2, y2 (input pixels) |
  | 4 | score, 0..1 |
  | 5 | class id as a float: 0 red_sentry, 1 red_1, 2 red_2, 3 red_3, 4 blue_sentry, 5 blue_1, 6 blue_2, 7 blue_3 |
  | 6-17 | 4 corners as **x, y, visible** (visible = 0..1): TL, BL, BR, TR |
- It only learned sentry, 1, 2 and 3. On the demo's blue 4s it says "blue 3" with a score under 0.3.
- **The export decides the output.** With `nms=False`, Ultralytics exports this end-to-end head. Without it,
  Ultralytics 8.4 exports the raw head instead: (1, 24, 8400), one column per guess, 4 box + 8 classes + 12 corner
  values, and you'd have to do NMS yourself. Same `.pt`, two different contracts.
- **Threshold:** 0.5 (Ultralytics' own default is 0.25).

## 4. The formats

| | ONNX `.onnx` | OpenVINO `.xml` + `.bin` | NCNN `.param` + `.bin` | TensorRT `.engine` |
|---|---|---|---|---|
| what it is | portable graph + weights | Intel's graph (XML text) + weights | Tencent's graph (plain text) + weights | a compiled program for ONE GPU |
| run it with | onnxruntime | openvino | ncnn | tensorrt |
| runs on | anything (CPU; GPU with onnxruntime-gpu) | CPU anywhere incl. the Jetson; **Intel** GPUs only | CPU (Vulkan GPU optional) | the NVIDIA GPU + TensorRT version that built it |
| stores IO names/shapes/dtypes | yes | yes | names only: **no input size, no dtype** | yes |
| stores labels/metadata | `metadata_props` | `rt_info` | no | no |
| can become an engine | **yes** (trtexec) | no official way back to ONNX | no | - |

- An `.xml` without its `.bin` (same name) is useless: the weights are in the `.bin`.
- `tup_1208_onnx2ncnn.bin` (originally `opt-1208-001.bin`) is NCNN weights: it starts with NCNN's 4-byte flag. TUP's repo has
  an OpenVINO `opt-1208-001.bin` with the same name and different contents. Same name, different format: check the bytes.
- **Keep the ONNX, or better the `.pt`.** Going from OpenVINO back to ONNX needs the third-party `openvino2onnx`. It worked for
  rp_0708_openvino.xml and failed for tongji_yolo11_int8.xml (see `../reference/sp_vision_25/JETSON_ORIN.md`), which is why we went looking for yolo11's original ONNX.
- The TensorRT version that built an engine sits in bytes 24-27 of the file (`runtimes.engine_trt_version()`): `10.6.0.26` for the
  `tup_1208_trt10.6.engine`, `10.3.0.30` for `rp_0708_openvino2onnx_nms_trt10.3.engine` (= JetPack 6.1/6.2). NVIDIA doesn't document this; it held on every engine we checked.

---

## 5. Things we found out the hard way

1. **RobotPilots' README has the colors wrong.** It says columns 9-12 are red, blue, gray, purple. On the demo video (all blue plates)
   column 9 lights up for rp_0708, rp_0526 and rp_0708_openvino: the order is **blue, red, gray, purple**, as the robot's code has it.
2. **Every detector in the zoo uses the same corner order:** TL, BL, BR, TR (counter-clockwise on screen). They come from the same
   community labeling format. AT's README calls this order "clockwise": that's true with the y axis pointing up. Image y points down.
3. **The old NCNN conversion changed the model.** `tup_1208_onnx2ncnn` (onnx2ncnn) finds 16 more plates than `tup_1208` over 60 frames,
   corners move up to 27 px and labels disagree on 7 frames. `tup_1208_pnnx` (pnnx) matches the ONNX to 0.35 px. Converting is
   not automatically safe: compare (m6).
4. **INT8 costs corner precision.** `tongji_yolo11_int8` (INT8) vs `tongji_yolo11` (FP32): same plates, same labels, corners up to ~8 px
   apart. The INT8 file is faster on CPU. For PnP, a few pixels on a far plate is a lot of distance.
5. **rp_0708_openvino (OpenVINO) matches rp_0708 (ONNX) to 0.1 px.** A careful conversion really is lossless.
6. **Metadata can be instructions, not facts.** `reverse_input_channels: YES` and `scale_values: 255` in the Ultralytics files mean
   "YOU must swap to RGB and divide by 255"; they don't mean it's already done.
7. **Old models don't know new plates.** TUP's 2022 `tup_0527` calls the demo's sentry plate an outpost; the four newer models agree
   it's a sentry. Plate stickers change between seasons.
8. **ONNX Runtime's threads spin after every run** and steal the CPU from the next model. `runtimes.py` turns that off; without it,
   side-by-side timings were off by 10x.
9. **Prebuilt engines are a trap.** Seven engines came with the models, all built with TensorRT 10.6. JetPack 6 has 10.3, so none
   of them load on our Jetson. We kept one (`tup_1208_trt10.6`) to show the error. Rebuild from ONNX on the Jetson
   (`jetson/build_engines.py`).
10. **Same shape, different meaning.** `at_c2psa_*` and `ns_yolo26n_v2` both answer with 18 numbers per row. In AT's,
    numbers 6-9 are color scores; in ours, they're the first corner's x, y, visible and the second corner's x.
    A decoder that only checks the shape would run fine and be wrong on every plate.
11. **One export setting changes the whole output.** Exporting `ns_yolo26n_v2.pt` with or without `nms=False`
    gives (1, 300, 18) or (1, 24, 8400). Whoever exports a model decides its contract: write down how.
12. **File names and stored labels can mislead.** `Infantry-v8n-fp16-...onnx` wants float32 input, and its stored class
    names are `class0` ... `class8`. Read the inputs from the file and the meanings from the team's docs or code.
13. **A model can only name what it was trained on.** `ns_yolo26n_v2` never saw a 4, so it calls blue 4s "blue 3",
    unsure (score < 0.3). `rp_v8n_0726` dropped infantry 5 and purple on purpose for 2026. Check the class list
    against the robots you'll actually fight.

---

## 6. Not in the zoo, and why

These came in the original `models/model_awakening_WUSTRM/` and `models/model_sp25/` folders and were taken out of the
repo when it was cleaned up:

| file | what it is | why it's out |
|---|---|---|
| `cb_rune.onnx/.engine` | YOLOv8 pose, `r_target`/`b_target`, 9 keypoints | energy rune (buff), not armor |
| `best2-sim.onnx` | YOLOv8n pose, 1 class, 5 keypoints | energy rune |
| `yolo11_buff_int8.xml/.bin` | YOLO11 pose, (1, 17, 8400) | energy rune |
| `standard_fanblade.jpg` | picture | rune reference image |
| `tiny_resnet`, `mlp`, `mlp_finetuned`, `lenet`, `reborn_number_classifier`, `label.txt` | number classifiers for the light-bar (non-NN) detector | separate topic |
| `0526.engine`, `0708.engine`, `praysky_*.engine` (2), `cb_rune.engine`, `reborn_number_classifier.engine` | TensorRT 10.6 engines | can't load on JetPack 6; rebuild from the ONNX instead |

---

## 7. Exporting yolov8.onnx yourself

`reborn_yolov8_int8` is INT8 and OpenVINO-only, so it can't become a TensorRT engine. Reborn published the training run it came from:
[gaoxinstudent/RM-Armor-Detect](https://github.com/gaoxinstudent/RM-Armor-Detect) has `ultralytics/runs/pose/train/weights/best.pt`
and `last.pt`, and its `args.yaml` matches the `.xml`'s metadata exactly (same `C:\Users\ssq13\...yolov8_large_MobileNetV3-pose.yaml`,
416x416, Ultralytics 8.1.5). One doubt: `results.csv` logs 8 of 1200 planned epochs, so the checkpoint may not be the exact weights in
the `.xml`. Running m6 after the export answers that.

The network uses Reborn's custom MobileNetV3 layers, so it only loads with their modified `ultralytics`, which lives in that repo.
`tools/export_reborn_yolov8.py` has the full steps (separate virtual environment, PyTorch 2.3 CPU, their package first on the import
path). It writes `models/reborn/reborn_yolov8.onnx`, which shows up as `reborn_yolov8`.

Loading a `.pt` runs code (it's a Python pickle). `tools/inspect_pt.py` checks any checkpoint without loading it. We
scanned both of Reborn's that way: they only reference `torch`,
`ultralytics`, `numpy`, `collections` and one `dill` helper that rebuilds Python's `set` type.

---

## 8. Adding a new model

1. Put the file in `models/<team>/`, named `<team>_<model>[_<what changed>]` like the others (or anywhere, with an absolute path).
2. Read its card: add a `ModelFile(...)` line to `zoo.py` and run `python3 m1_io_cards.py --models <name>`.
3. Find out its contract. If the output matches a family table above, give it that family. If not, write a new family module
   (copy the closest of m2-m5b, add it to `FAMILY_MODULES` in `common.py`).
   A `.pt` from Ultralytics becomes an ONNX with `tools/export_ultralytics.py` (it asks for the end-to-end head when
   the model has one).
4. Find the team's inference code and check the input side: color order, 0..1 or 0..255, letterbox position and pad value.
   Then check the output side with Part A of the family script: draw the raw points and see where they land.
5. If it's an ONNX, `jetson/build_engines.py --models <name>` makes an engine; then `m8_benchmark.py` puts it on the scoreboard.

---

## Licenses

Who allows what. This repo is for our team's learning; check this table before making it public or putting a model
on the robot for a competition.

| source | license | what that means for us |
|---|---|---|
| robot code ([sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25), our fork), copied into `../reference/` | MIT | free to use and copy, keep the license notice (it's in `reference/sp_vision_25/LICENSE`) |
| Tongji's [LabelRoboMaster](https://github.com/TongjiSuperPower/LabelRoboMaster) (`tongji_yolo11.onnx`) | MIT | same |
| RobotPilots' [RobotDetectionModel](https://github.com/broalantaps/RobotDetectionModel) (`rp_*`) | no license file | by default "all rights reserved": published for other teams to use, but nothing written. Ask before redistributing |
| Reborn's [inference repo](https://github.com/RebornVision/Reborn-Vision-2024-armor-Inference) (`reborn_yolov8_int8`) | no license file | same as above |
| Reborn's [training repo](https://github.com/gaoxinstudent/RM-Armor-Detect) | AGPL-3.0 | sharing modified versions means sharing their source too |
| TUP's [TUP-InfantryVision-2022](https://github.com/tup-robomaster/TUP-InfantryVision-2022) (`tup_*`) | GPL-3.0 | sharing modified versions means sharing their source too. That's why we link to TUP's code instead of copying it |
| Actor&Thinker's [AT_NN_Detector](https://github.com/PraySky1337/AT_NN_Detector) (`at_*`) | AGPL-3.0 | same as above; also why we link instead of copying |
| RobotPilots' [SZURPVision deployment repo](https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models) (`rp_v8n_0726`) | MIT | free to use and copy, keep their notice |
| our `ns_yolo26n_v2` | ours, but trained with Ultralytics (AGPL-3.0) and partly on Actor&Thinker's dataset (their repo is AGPL-3.0) | treat it like the AGPL models above |
| models trained with Ultralytics (`reborn_*`, `tongji_*`, `at_*`, `rp_v8n_0726`, `ns_*`) | their metadata says AGPL-3.0 | same as above |

This is a summary to help you notice the question, not legal advice.
