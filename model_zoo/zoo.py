"""Every armor-detection model file we have, where it came from, and which decoder reads it.

One entry = one FILE. The same trained network often exists as several files (ONNX, OpenVINO
.xml/.bin, NCNN .param/.bin, TensorRT .engine), so entries that share a `network` id hold the
same weights in different packaging. MODELS.md tells the full story of each one.

Paths: models live in ../models/<team>/ (override with MODELS_DIR), the demo video in ../videos
(override with VIDEO). Engines you build on the Jetson with jetson/build_engines.py land in
./engines and are picked up automatically.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", HERE.parent / "models")).expanduser()
ENGINES_DIR = Path(os.environ.get("ENGINES_DIR", HERE / "engines")).expanduser()

sys.path.append(str(HERE.parent / "shared"))  # cameras.py, shared with the main track
from cameras import DEMO_VIDEO as VIDEO_PATH  # noqa: E402  (videos/demo.avi, or $VIDEO)

# Calibration of the camera that recorded demo.avi (also in ../configs/demo_camera.yaml), for distances in m8
CAMERA_MATRIX = [[1818.3669452465165, 0, 751.06226574703498], [0, 1822.494494078506, 530.43671556112133], [0, 0, 1]]
DIST_COEFFS = [-0.077944626599568856, 0.15447826031486889, -0.0025714394278524674, 0.00083016311301273629, 0]
SMALL_ARMOR_WIDTH, BIG_ARMOR_WIDTH, LIGHTBAR_LENGTH = 0.135, 0.230, 0.056

FORMATS = {".onnx": "onnx", ".xml": "openvino", ".param": "ncnn", ".engine": "tensorrt"}

RP = "Shenzhen Univ. RobotPilots"
TUP = "Shenyang Aerospace Univ. TUP"
REBORN = "USTB Reborn"
TONGJI = "Tongji SuperPower"
AT = "HEBUST Actor&Thinker"


@dataclass
class ModelFile:
    name: str      # short id for the command line, e.g. --models rp_0708
    path: str      # relative to MODELS_DIR (or absolute)
    family: str    # which decoder reads its output: yolov5 | yolox | ultralytics | end2end
    network: str   # entries with the same network id hold the same trained weights
    team: str
    source: str    # where the file was published
    notes: str = ""
    input_hw: tuple = None  # only for formats that don't record their input size (NCNN)

    @property
    def file(self):
        return Path(self.path) if Path(self.path).is_absolute() else MODELS_DIR / self.path

    @property
    def format(self):
        return FORMATS[self.file.suffix]

    def exists(self):
        return self.file.exists()


RPD, TUPD, REBD, TJD, ATD = "robotpilots/", "tup/", "reborn/", "tongji/", "actor_thinker/"

MODELS = [
    # ---- family "yolov5": 25200 rows (guesses) x 22 values, corners already in pixels
    ModelFile("rp_0708", RPD + "rp_0708.onnx", "yolov5", "rp-0708", RP,
              "https://github.com/broalantaps/RobotDetectionModel (Model/0708.onnx)",
              "The original PyTorch export. Every other rp_0708_* file was converted from this one."),
    ModelFile("rp_0526", RPD + "rp_0526.onnx", "yolov5", "rp-0526", RP,
              "https://github.com/broalantaps/RobotDetectionModel (Model/0526.onnx)",
              "Newer RobotPilots model (2025). Same output layout, but the input is FLOAT16."),
    ModelFile("rp_0708_openvino", RPD + "rp_0708_openvino.xml", "yolov5", "rp-0708", TONGJI,
              "https://github.com/TongjiSuperPower/sp_vision_25 (assets/yolov5.xml)",
              "rp_0708 converted to OpenVINO IR (its rt_info still says input_model = DIR/0708.onnx)."),
    ModelFile("rp_0708_openvino2onnx", RPD + "rp_0708_openvino2onnx.onnx", "yolov5", "rp-0708", "Northstar",
              "https://github.com/Northstar-Advanced-Robotics/Northstar-sp_vision_25 (scripts/onnx/yolov5_prefusion_backup.onnx)",
              "rp_0708_openvino converted back to ONNX with openvino2onnx. The main track's laptop model. "
              "The converter named the output after the XML's Result layer: output/sink_port_0."),
    ModelFile("rp_0708_openvino2onnx_nms", RPD + "rp_0708_openvino2onnx_nms.onnx", "yolov5", "rp-0708", "Northstar",
              "https://github.com/Northstar-Advanced-Robotics/Northstar-sp_vision_25 (assets/yolov5.onnx)",
              "rp_0708_openvino2onnx + an EfficientNMS_TRT node: the main track's model, the robot's yolov5.onnx. "
              "TensorRT only."),
    ModelFile("rp_0708_openvino2onnx_nms_trt10.3", RPD + "rp_0708_openvino2onnx_nms_trt10.3.engine", "yolov5",
              "rp-0708", "Northstar", "built from rp_0708_openvino2onnx_nms with trtexec",
              "Prebuilt engine (TensorRT 10.3 = JetPack 6.x). Only loads on a matching Jetson."),

    # ---- family "yolox": one row per grid cell (3549 x 21), corners in cells, not pixels yet
    ModelFile("tup_0527", TUPD + "tup_0527.onnx", "yolox", "tup-0527", TUP,
              "https://github.com/tup-robomaster/TUP-InfantryVision-2022 (model/opt-0527-001.onnx)",
              "TUP's 2022 armor model. Input is BGR 0..255, no normalization."),
    ModelFile("tup_1208", TUPD + "tup_1208.onnx", "yolox", "tup-1208", TUP,
              "https://github.com/RebornVision/TUP-Vision-2023-Based (copy of TUP's repo, opt-1208-001.onnx)",
              "Same layout as tup_0527, a later training run."),
    ModelFile("tup_1208_onnx2ncnn", TUPD + "tup_1208_onnx2ncnn.param", "yolox", "tup-1208", "unknown (onnx2ncnn)",
              "came with the WUSTRM folder (opt-1208-001.param/.bin)",
              "NCNN, converted with the old onnx2ncnn tool: blob names images/output. Does NOT match tup_1208 (m6).",
              input_hw=(416, 416)),
    ModelFile("tup_1208_pnnx", TUPD + "tup_1208_pnnx.ncnn.param", "yolox", "tup-1208", "unknown (pnnx)",
              "came with the WUSTRM folder (opt_1208_001.ncnn.param/.bin)",
              "NCNN, converted with the newer pnnx tool: blob names in0/out0, fp16 weights.", input_hw=(416, 416)),
    ModelFile("tup_1208_trt10.6", TUPD + "tup_1208_trt10.6.engine", "yolox", "tup-1208", "unknown",
              "came with the WUSTRM folder (opt-1208-001.engine)",
              "Built with TensorRT 10.6: will NOT load on JetPack 6 (10.3). Kept as the example of that."),

    # ---- family "ultralytics": YOLOv8/YOLO11 "pose" models, one COLUMN per guess: box, class scores, 4 corners
    ModelFile("reborn_yolov8_int8", REBD + "reborn_yolov8_int8.xml", "ultralytics", "reborn-v8-mbv3", REBORN,
              "https://github.com/RebornVision/Reborn-Vision-2024-armor-Inference (model/mobilenetv3_last_int_all_new/last.xml)",
              "YOLOv8 pose with a MobileNetV3 backbone, INT8. Only knows color (B/R), not the number. "
              "sp_vision ships it as assets/yolov8.xml."),
    ModelFile("reborn_yolov8", REBD + "reborn_yolov8.onnx", "ultralytics", "reborn-v8-mbv3", REBORN,
              "export it yourself from https://github.com/gaoxinstudent/RM-Armor-Detect (MODELS.md, section 7)",
              "FP32 ONNX of the same training run. Not in the repo until you export it."),
    ModelFile("tongji_yolo11_int8", TJD + "tongji_yolo11_int8.xml", "ultralytics", "tongji-yolo11n", TONGJI,
              "https://github.com/TongjiSuperPower/sp_vision_25 (assets/yolo11.xml)",
              "YOLO11n pose, INT8, 38 classes. Its output tensor has no name."),
    ModelFile("tongji_yolo11", TJD + "tongji_yolo11.onnx", "ultralytics", "tongji-yolo11n", TONGJI,
              "https://github.com/TongjiSuperPower/LabelRoboMaster (resource/best.onnx)",
              "FP32 ONNX of the same network as tongji_yolo11_int8 (before INT8). The one to build an engine from."),

    # ---- family "end2end": NMS is inside the model, 30 finished detections x 18 values
    ModelFile("at_c2psa_640", ATD + "at_c2psa_640.onnx", "end2end", "at-c2psa-640", AT,
              "https://github.com/PraySky1337/AT_NN_Detector (common/praysky_c2psa_e2e_0228_640x640.onnx)",
              "End-to-end YOLO pose with C2PSA attention; outputs final detections, no NMS needed."),
    ModelFile("at_c2psa_576x768", ATD + "at_c2psa_576x768.onnx", "end2end", "at-c2psa-576x768", AT,
              "https://github.com/PraySky1337/AT_NN_Detector (common/praysky_c2psa_e2e_0228_576x768.onnx)",
              "Same as at_c2psa_640 but 576x768: a 1440x1080 camera frame fits with no padding at all."),
]


def _engines_you_built():
    """Engines from jetson/build_engines.py are named after their ONNX model: <name>.engine (FP16)
    or <name>_fp32.engine. They show up as <name>_trt and <name>_trt_fp32."""
    by_name = {m.name: m for m in MODELS}
    found = []
    for f in sorted(ENGINES_DIR.glob("*.engine")) if ENGINES_DIR.exists() else []:
        stem, precision = (f.stem[:-5], "_fp32") if f.stem.endswith("_fp32") else (f.stem, "")
        base = by_name.get(stem)
        if base:
            found.append(ModelFile(f"{stem}_trt{precision}", str(f), base.family, base.network, base.team,
                                   f"built here from {base.name}",
                                   f"Your own {'FP32' if precision else 'FP16'} engine, built on this device."))
    return found


MODELS += _engines_you_built()
BY_NAME = {m.name: m for m in MODELS}


def find(names=None, family=None, formats=None, only_existing=True):
    """Pick models by name (list or comma string), family and/or format ("onnx", "openvino", ...)."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if names:
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise SystemExit(f"Unknown model(s) {unknown}. Known: {', '.join(BY_NAME)}")
        picked = [BY_NAME[n] for n in names]
    else:
        picked = MODELS
    if family:
        picked = [m for m in picked if m.family == family]
    if formats:
        picked = [m for m in picked if m.format in formats]
    return [m for m in picked if m.exists()] if only_existing else picked
