"""Export Reborn's YOLOv8-MobileNetV3-pose checkpoint (.pt) to ONNX, so it can become an engine.

yolov8.xml is INT8 and OpenVINO-only. Reborn published the training run it came from, including
the .pt checkpoints, at https://github.com/gaoxinstudent/RM-Armor-Detect. Their network uses custom
MobileNetV3 layers, so it only loads with THEIR modified copy of ultralytics (8.1.5), which lives
in that repo. This script puts their copy first on the import path and exports from there.

Do this ONCE, on a laptop (not the Jetson), in a separate virtual environment:

    git clone --depth 1 https://github.com/gaoxinstudent/RM-Armor-Detect ~/RM-Armor-Detect
    python3 -m venv ~/venv-export && source ~/venv-export/bin/activate     (Windows: venv-export\\Scripts\\activate)
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
    pip install "numpy<2" opencv-python pyyaml requests scipy tqdm psutil py-cpuinfo thop pandas seaborn matplotlib pillow dill onnx==1.16.1
    python3 tools/export_reborn_yolov8.py ~/RM-Armor-Detect

Then check it IS the same network as reborn_yolov8_int8.xml:  python3 m6_same_network.py
(reborn_yolov8_int8 vs reborn_yolov8: expect the same plates, corners a few px apart from INT8).

A .pt file is a Python pickle: loading one runs code. Only load checkpoints from people you trust,
in an environment you can throw away. (We checked these two: they only reference torch,
ultralytics, numpy and the standard library.)
"""
import argparse
import shutil
import sys
from pathlib import Path

ZOO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ZOO))
import zoo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="your clone of https://github.com/gaoxinstudent/RM-Armor-Detect")
    ap.add_argument("--weights", choices=["last", "best"], default="last",
                    help="which checkpoint (Reborn's OpenVINO folder is named after 'last')")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    pt = repo / "ultralytics/runs/pose/train/weights" / f"{args.weights}.pt"
    if not pt.exists():
        raise SystemExit(f"{pt} not found. Is {repo} a clone of RM-Armor-Detect?")
    sys.path.insert(0, str(repo))  # their ultralytics/ folder must win over any installed one
    import ultralytics
    from ultralytics import YOLO

    if not Path(ultralytics.__file__).resolve().is_relative_to(repo):
        raise SystemExit(f"Imported ultralytics from {ultralytics.__file__}, not from {repo}. "
                         "Uninstall ultralytics from this environment and try again.")
    model = YOLO(str(pt))
    print(f"{pt.name}: task {model.task}, classes {model.names}, input 416x416")
    onnx_file = model.export(format="onnx", imgsz=416, opset=12, simplify=False, dynamic=False)
    target = zoo.BY_NAME["reborn_yolov8"].file
    shutil.copy(onnx_file, target)
    print(f"\nSaved {target}. It's now in the zoo as reborn_yolov8.")


if __name__ == "__main__":
    main()
