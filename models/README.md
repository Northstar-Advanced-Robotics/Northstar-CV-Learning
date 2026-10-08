# Models

Every model file the two tracks use, one folder per team. File names say what's inside:
`<team>_<model>[_<what was done to it>]`. In `model_zoo/`, a file's zoo name (for `--models`) is its file name
without the extension. [model_zoo/MODELS.md](../model_zoo/MODELS.md) has the full story: where each file came
from, its inputs and outputs, and what we found out about it.

| file | what it is | used by | original name |
|---|---|---|---|
| **robotpilots/** | Shenzhen Univ. RobotPilots' 4-point YOLOv5 | | |
| `rp_0708.onnx` | their 2024 model, the original PyTorch export | zoo | `0708.onnx` |
| `rp_0526.onnx` | their 2025 model (float16 input) | zoo | `0526.onnx` |
| `rp_0708_openvino.xml/.bin` | rp_0708 converted to OpenVINO by Tongji | zoo | sp_vision `assets/yolov5.xml` |
| `rp_0708_openvino2onnx.onnx` | that converted back to ONNX | main track (laptop mode), zoo | `yolov5_prefusion_backup.onnx` |
| `rp_0708_openvino2onnx_nms.onnx` | that + a TensorRT NMS node: **the main track's model, the robot's `yolov5.onnx`** | main track, zoo | sp_vision `assets/yolov5.onnx` |
| `rp_0708_openvino2onnx_nms_trt10.3.engine` | engine of the one above, TensorRT 10.3 (JetPack 6.1/6.2) | zoo | `yolov5.engine` |
| **tup/** | Shenyang Aerospace Univ. TUP's YOLOX | | |
| `tup_0527.onnx` | 2022 model | zoo | `opt-0527-001.onnx` |
| `tup_1208.onnx` | later model | zoo | `opt-1208-001.onnx` |
| `tup_1208_onnx2ncnn.param/.bin` | tup_1208 as NCNN, old converter (**changes the model**, see MODELS.md) | zoo | `opt-1208-001.param/.bin` |
| `tup_1208_pnnx.ncnn.param/.bin` | tup_1208 as NCNN, new converter (matches) | zoo | `opt_1208_001.ncnn.param/.bin` |
| `tup_1208_trt10.6.engine` | engine built with TensorRT 10.6: kept to show it **won't load** on JetPack 6 | zoo | `opt-1208-001.engine` |
| **reborn/** | USTB Reborn's YOLOv8-pose with a MobileNetV3 backbone | | |
| `reborn_yolov8_int8.xml/.bin` | INT8 OpenVINO, color only | zoo | `last.xml`; sp_vision `assets/yolov8.xml` |
| `reborn_yolov8.onnx` | not here until you export it (`model_zoo/tools/export_reborn_yolov8.py`) | zoo | - |
| **tongji/** | Tongji SuperPower's YOLO11n-pose, 38 classes | | |
| `tongji_yolo11_int8.xml/.bin` | INT8 OpenVINO | zoo | sp_vision `assets/yolo11.xml` |
| `tongji_yolo11.onnx` | the FP32 original | zoo | LabelRoboMaster `resource/best.onnx` |
| **actor_thinker/** | HEBUST Actor&Thinker's end-to-end YOLO pose | | |
| `at_c2psa_640.onnx` | 640x640 input | zoo | `praysky_c2psa_e2e_0228_640x640.onnx` |
| `at_c2psa_576x768.onnx` | 576x768 input (the camera's 4:3 shape) | zoo | `praysky_c2psa_e2e_0228_576x768.onnx` |

Engines are built for one GPU and one TensorRT version, so build your own on the Jetson: the main track's
exercise 2 builds `exercises/yolov5.engine`, and `model_zoo/jetson/build_engines.py` builds one for every ONNX here
into `model_zoo/engines/`. Neither is committed: they only work on the device that built them.

Not kept: the energy-rune models, the number classifiers and six more TensorRT 10.6 engines that came with the
original `model_awakening_WUSTRM/` and `model_sp25/` folders ([MODELS.md, section 6](../model_zoo/MODELS.md#6-not-in-the-zoo-and-why)).

Git: `.gitattributes` marks these files as binary, so git never rewrites their line endings and they stay
byte-for-byte what their authors published. Every file is under GitHub's 100 MB limit.
