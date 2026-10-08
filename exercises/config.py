"""Shared settings for every exercise. Change paths and numbers here, not in the exercise files."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # the cv_learning folder
sys.path.append(str(REPO / "shared"))  # so `import cameras` works (exercises 6 and 7)

# The model and the footage are in this repo (models/README.md says where each file came from).
# The robot calls the first model assets/yolov5.onnx. The second is the same model before the NMS step
# was added to it, which is what ONNX Runtime can run on a laptop.
ONNX_PATH = REPO / "models/robotpilots/rp_0708_openvino2onnx_nms.onnx"  # model WITH the added NMS step
ONNX_NO_NMS_PATH = REPO / "models/robotpilots/rp_0708_openvino2onnx.onnx"  # same model, without it
VIDEO_PATH = Path(os.environ.get("VIDEO", REPO / "videos/demo.avi")).expanduser()  # footage with armor plates
ENGINE_PATH = Path(__file__).parent / "yolov5.engine"  # you build this in exercise 2

# Which program runs the model:
# "tensorrt"     -> your .engine on the Jetson's GPU (the real thing)
# "onnxruntime"  -> the ONNX file on any computer's CPU (slower, works anywhere)
# Set it with the BACKEND environment variable (README, Setup).
BACKEND = os.environ.get("BACKEND", "tensorrt")

INPUT_SIZE = 640  # the model's input is INPUT_SIZE x INPUT_SIZE pixels
SCORE_THRESHOLD = 0.7  # keep guesses at least this sure, and NMS overlap limit: the robot's values
NMS_IOU_THRESHOLD = 0.3  # (reference/sp_vision_25/tasks/auto_aim/yolos/yolov5.hpp)

# What each color and number column means, in the model's order
# (reference/sp_vision_25/tasks/auto_aim/armor.hpp and armor.cpp)
COLORS = ["blue", "red", "gray", "purple"]
NUMBERS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base", "not_armor"]

# Calibration of the camera that recorded demo.avi (also in configs/demo_camera.yaml).
# Use your own camera's numbers before trusting distances measured with it (BASICS.md, section 9).
CAMERA_MATRIX = [
    [1818.3669452465165, 0, 751.06226574703498],
    [0, 1822.494494078506, 530.43671556112133],
    [0, 0, 1],
]
DIST_COEFFS = [-0.077944626599568856, 0.15447826031486889, -0.0025714394278524674, 0.00083016311301273629, 0]

# Armor plate size in meters: distance between the light bars, and light-bar height
# (reference/sp_vision_25/tasks/auto_aim/solver.cpp)
SMALL_ARMOR_WIDTH = 0.135
BIG_ARMOR_WIDTH = 0.230
LIGHTBAR_LENGTH = 0.056
